"""Where the test database lives, and whether it is actually there.

⭐ ONE module owns the DSN. ``tests/conftest.py``, ``tests/test_db.py`` and
``tests/test_alembic_does_not_silence_logging.py`` each used to answer "which
database does this suite talk to?" separately — conftest resolved
``TEST_DATABASE_URL > DATABASE_URL > localhost:5451`` while ``test_db.py``
re-implemented a two-level version of the same precedence over its own second
copy of the default DSN. Two copies of one rule drift, and they had already
begun to: ``test_db.py`` never consulted ``DATABASE_URL`` at all, so under the
CI archetype — which exports ``DATABASE_URL`` and never sets
``TEST_DATABASE_URL`` — it resolved to ``localhost:5451``, a port no CI runner
has ever had open, and its whole module skipped on a port probe while the rest
of the suite ran against the real CI database. All three now call
:func:`resolve_test_database_url`, and the ``createdb`` remedy is rendered from
that same DSN, so the thing that connects and the thing that creates cannot
disagree.

## Why the precondition is not a port probe

``test_db.py`` guarded itself with ``socket.create_connection((host, port))``
and a module-level ``pytest.skip(allow_module_level=True)``.

⚠️ That was SOUND at the default. ``localhost:5451`` is this repo's OWN
docker-compose Postgres (``docker-compose.override.yml``, ``5451:5432``), a
REPO-PRIVATE port: it is open iff that container — and therefore the database
inside it — is up. The probe is a valid proxy for exactly as long as that stays
true.

⛔ It stops being true the moment ``TEST_DATABASE_URL`` or ``DATABASE_URL``
points at a SHARED or host-native PostgreSQL, which is what the fleet has been
moving to (``hirag_service``, ``kag_graph_service``). A shared 5432 is open
whether or not this repo's database was ever created, whether or not the role
exists, and whether or not the password is right — so the probe passes, the
module does not skip, and the tests fail for a reason that names the wrong
cause. Worse, at a pool open there is no error to read at all:
``AsyncConnectionPool.open()`` defaults to ``wait=False`` and returns without
raising, so the first symptom is a ``PoolTimeout`` from inside a test body.

So the question the suite asks is the only one that answers it: can we CONNECT
to THAT DATABASE and get an answer out of it.

## Three states, because the two failures have different remedies

``PRESENT``     connected and ``SELECT 1`` answered. The suite can run.
``ABSENT``      the SERVER answered and this DATABASE is not there. One
                documented command fixes it (:func:`createdb_command`).
``UNREACHABLE`` no server, wrong port, wrong credentials, missing role,
                blackholed host. A different remedy entirely.

Collapsing them would hand a developer whose server is down a ``createdb`` they
cannot run, and — the direction that actually matters here — would let a
missing database keep hiding behind the friendly no-Postgres skip this repo
deliberately keeps (see ``conftest.decide_on_probe``).

Mirrors ``hirag_service/tests/db_url.py`` and
``kag_graph_service/tests/local_db.py``, which closed the same defect in the two
sibling repos. Do not invent a fourth mechanism.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from urllib.parse import quote, unquote, urlsplit, urlunsplit

# This repo's own docker-compose Postgres (docker-compose.override.yml maps
# 5451:5432). Throwaway local-dev placeholders, not credentials — detect-secrets
# flags the `user:pass@host` shape alone.
DEFAULT_TEST_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5451/data_collector"  # pragma: allowlist secret

#: Env vars consulted, in precedence order. ``TEST_DATABASE_URL`` sits ABOVE
#: ``DATABASE_URL`` deliberately: it is how one lane points itself at its own
#: database without editing a shared file. CI exports ``DATABASE_URL`` (the
#: python-service archetype's per-job ephemeral ``postgres:18``) and never sets
#: ``TEST_DATABASE_URL``, so CI resolves to the sidecar exactly as before.
PRECEDENCE = ("TEST_DATABASE_URL", "DATABASE_URL")

#: Probe outcomes.
PRESENT = "present"
ABSENT = "absent"
UNREACHABLE = "unreachable"

# ⛔ WHY CLASSIFICATION CANNOT BE BY EXCEPTION CLASS.
#
# Measured on psycopg 3.3.4 against PostgreSQL 18.4: every CONNECT-time failure
# arrives as a bare `OperationalError` whose `.sqlstate` AND `.diag.sqlstate` are
# both `None`. A missing database, a missing role and a dead server are
# indistinguishable by class — and `issubclass(InvalidCatalogName,
# OperationalError)` is False, so `except InvalidCatalogName` around a
# `connect()` is dead code. (It is still caught below, because the class DOES
# arrive when a STATEMENT, rather than a connection, hits SQLSTATE 3D000.)
#
# ⚠️ The message text is therefore consulted — but libpq translates it, so a
# non-English `lc_messages` matches nothing. An unmatched message is treated as
# UNKNOWN and escalated to the structural probe below; it is never guessed into
# one of the two failure states. Anchoring on the word `database` also keeps a
# missing ROLE (`role "x" does not exist`, a completely different remedy) from
# being reported as a missing database.
_MISSING_DATABASE = re.compile(r'database "[^"]*" does not exist', re.IGNORECASE)
_MISSING_ROLE = re.compile(r'role "[^"]*" does not exist', re.IGNORECASE)

# ⛔ NOT a hardcoded `postgres`. `CREATE DATABASE`-style bookkeeping has to be
# asked of SOME other database on the same server, and a scoped role often
# cannot open `postgres`. Each candidate is tried in turn and a server that
# answers none of them yields "inconclusive" — reported as UNREACHABLE with the
# reason spelled out, never silently converted into ABSENT.
MAINTENANCE_DATABASES = ("postgres", "template1")

#: Memoised per (dsn, timeout). This is a PRECONDITION: one answer per database
#: per session is the correct number of answers.
_PROBE_CACHE: dict[tuple[str, int], tuple[str, str]] = {}


def resolve_test_database_url(env: Mapping[str, str] | None = None) -> str:
    """Resolve the DSN the suite will connect to.

    An empty or whitespace-only value is treated as unset: ``VAR=`` in a shell
    profile or a CI matrix is a far likelier accident than a deliberate request
    to connect to the empty string, and passing it through produces a much worse
    error than the default would.
    """
    source = os.environ if env is None else env
    for name in PRECEDENCE:
        value = (source.get(name) or "").strip()
        if value:
            return value
    return DEFAULT_TEST_DATABASE_URL


def resolved_source(env: Mapping[str, str] | None = None) -> str:
    """Name of the env var that supplied the DSN, or ``"built-in default"``."""
    source = os.environ if env is None else env
    for name in PRECEDENCE:
        if (source.get(name) or "").strip():
            return f"${name}"
    return "built-in default"


def running_in_ci(env: Mapping[str, str] | None = None) -> bool:
    """True inside GitHub Actions (or any CI setting ``CI=true``).

    GitHub Actions sets both ``CI`` and ``GITHUB_ACTIONS`` in every job, so the
    fleet's python-service archetype needs no extra wiring. Same predicate as
    ``kag_graph_service/tests/conftest.py`` — do NOT invent a new env var.
    """
    source = os.environ if env is None else env
    return source.get("CI") == "true" or source.get("GITHUB_ACTIONS") == "true"


def libpq_dsn(dsn: str) -> str:
    """Strip SQLAlchemy's ``+psycopg`` driver suffix, which libpq cannot parse."""
    return dsn.replace("postgresql+psycopg://", "postgresql://", 1)


def database_name(dsn: str) -> str:
    """The database component of ``dsn``."""
    return unquote(urlsplit(libpq_dsn(dsn)).path.lstrip("/"))


def mask(dsn: str) -> str:
    """``dsn`` with its password replaced — safe to print in a gate message.

    Every message this module produces goes through here. The DSN carries the
    database password in its userinfo, and a precondition that leaks it into CI
    logs has traded one defect for a worse one.
    """
    parts = urlsplit(dsn)
    if not parts.password:
        return dsn
    userinfo = quote(unquote(parts.username or ""), safe="") + ":***"
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit(parts._replace(netloc=f"{userinfo}@{host}"))


def maintenance_dsns(dsn: str) -> list[str]:
    """The same server as ``dsn``, addressed at databases that ought to exist."""
    parts = urlsplit(libpq_dsn(dsn))
    target = database_name(dsn)
    candidates = (*MAINTENANCE_DATABASES, unquote(parts.username or ""))
    out: list[str] = []
    seen: set[str] = set()
    for name in candidates:
        if not name or name == target or name in seen:
            continue
        seen.add(name)
        out.append(urlunsplit(parts._replace(path="/" + quote(name))))
    return out


def createdb_command(dsn: str) -> str:
    """The exact shell command that provisions this DSN's database.

    ⛔ Single source. The remedy is rendered from the DSN the suite actually
    resolved, so a remedy quoted by hand cannot rot when the DSN moves.
    """
    parts = urlsplit(libpq_dsn(dsn))
    user = unquote(parts.username or "")
    flag = f" -U {user}" if user else ""
    host = parts.hostname or "127.0.0.1"
    return f"createdb -h {host} -p {parts.port or 5432}{flag} {database_name(dsn)}"


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())


def _database_exists(dsn: str, connect_timeout: int) -> bool | None:
    """Ask the SERVER, structurally. ``None`` when no maintenance DB answered."""
    import psycopg

    target = database_name(dsn)
    for admin in maintenance_dsns(dsn):
        try:
            with psycopg.connect(libpq_dsn(admin), connect_timeout=connect_timeout) as conn:
                row = conn.execute(
                    "select 1 from pg_database where datname = %s", (target,)
                ).fetchone()
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
        return row is not None
    return None


def classify_connect_failure(
    dsn: str, exc: BaseException, connect_timeout: int = 3
) -> tuple[str, str]:
    """Turn a connect-time exception into ``(state, detail)``.

    Two stages, deliberately in this order:

    1. libpq's own wording, when we recognise it — cheap, and exact.
    2. a STRUCTURAL question to the server (``pg_database``) when we do not.

    Stage 2 is what makes stage 1's locale-dependence safe: an unrecognised
    message never becomes ABSENT by assumption, it becomes a question we ask the
    server, and if the server will not answer it stays UNREACHABLE with the
    reason attached.
    """
    detail = f"{type(exc).__name__}: {_one_line(exc)}"
    name = database_name(dsn)

    if _MISSING_DATABASE.search(str(exc)):
        return (ABSENT, detail)
    if _MISSING_ROLE.search(str(exc)):
        return (UNREACHABLE, f"{detail} — the ROLE is missing, not the database")

    exists = _database_exists(dsn, connect_timeout)
    if exists is False:
        return (ABSENT, f"{detail} — and pg_database on that server holds no {name!r}")
    if exists is True:
        return (
            UNREACHABLE,
            f"{detail} — the server is up and {name!r} exists, so this is a "
            f"connection or credential problem",
        )
    return (
        UNREACHABLE,
        f"{detail} — no maintenance database answered, so whether {name!r} exists is "
        f"UNKNOWN; reporting UNREACHABLE rather than guessing ABSENT",
    )


def probe_database(
    dsn: str, *, connect_timeout: int = 3, use_cache: bool = True
) -> tuple[str, str]:
    """Return ``(state, detail)`` for ``dsn`` — fast, and never raising.

    ⚠️ ``connect_timeout`` is load-bearing. Without an explicit deadline a
    blackholed host hangs for the OS default (~2 min on Linux) per call, which
    is how a precondition turns into the slowest thing in the run.
    """
    key = (dsn, connect_timeout)
    if use_cache and key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - psycopg is a hard dependency
        return (UNREACHABLE, f"psycopg is not importable: {exc}")

    try:
        with psycopg.connect(libpq_dsn(dsn), connect_timeout=connect_timeout) as conn:
            row = conn.execute("select 1").fetchone()
        result = (
            (PRESENT, "")
            if row == (1,)
            else (UNREACHABLE, "connected, but `select 1` did not return 1")
        )
    except psycopg.errors.InvalidCatalogName as exc:
        # SQLSTATE 3D000, class-typed only when a STATEMENT hits it.
        result = (ABSENT, _one_line(exc))
    except Exception as exc:  # noqa: BLE001 - every other shape is a connect failure
        result = classify_connect_failure(dsn, exc, connect_timeout)

    if use_cache:
        _PROBE_CACHE[key] = result
    return result


def absent_message(dsn: str, detail: str, what: str) -> str:
    """The refusal for ABSENT — loud in CI **and** locally. See conftest."""
    return f"""
The test database does not exist, so {what} did not run.

  DSN ({resolved_source()}): {mask(dsn)}
  database:                  {database_name(dsn)}
  server:                    REACHABLE — it answered: {detail}

This is NOT "there is no PostgreSQL here". The server is up and simply does not
have this database, and exactly one command fixes that:

    {createdb_command(dsn)}

Or bring up this repo's own local Postgres, which creates it as a side effect of
starting:

    docker compose up -d db          # docker-compose.override.yml, host port 5451

Running a second lane on this box? Point TEST_DATABASE_URL at your own database
and create it with the command above — it is rendered from the DSN this check
resolved, so the two cannot disagree.

⚠️ Why this FAILS rather than skipping, while an unreachable server still skips
locally: those are different situations with different remedies. "No PostgreSQL
here" is a developer running a reduced suite on purpose, and this repo's
CLAUDE.md keeps that deliberately. "The server is up and the database was never
created" is a provisioning gap that one command closes — and a skip is exactly
how such a gap stays invisible while the run reports success.
"""


def unreachable_message(dsn: str, detail: str, what: str) -> str:
    """The refusal for UNREACHABLE — CI only; locally this stays a skip."""
    return f"""
Test database UNREACHABLE, so {what} did not run: {detail}

  DSN ({resolved_source()}): {mask(dsn)}

A run that could not execute must never look like one that passed — failing
loudly instead of skipping. In CI the ephemeral postgres:18 container (fleet
python-service archetype, `has_postgres: true`) is a hard prerequisite: check the
"Prepare CI database" step and the DATABASE_URL it exports.

Locally this same state is a clean skip, on purpose — see tests/conftest.py.
"""


# ---------------------------------------------------------------------------
# The verdict guard
# ---------------------------------------------------------------------------
#
# The gate above fails on ABSENT everywhere and on UNREACHABLE in CI. This is the
# backstop for the next database-shaped skip somebody adds:
# ``conftest.pytest_sessionfinish`` reddens a CI run carrying one, so it can
# never ride out as `exit 0`.
#
# ⚠️ TWO patterns, because English puts the negation on either side of the noun
# and a single ordered pattern can only see one of them. `database ... not
# available` is the trailing shape (and is verbatim what conftest's old
# `pytest.skip(f"PostgreSQL not available: {e}")` produced); `no test database
# configured` is the leading shape, which the trailing pattern misses because
# what follows "database" is "configured", never "not configured".
_DATABASE_SKIP_REASON_TRAILING = re.compile(
    r"\b(database|postgres|postgresql)\b[^\n]*?"
    r"\b(not reachable|unreachable|not available|unavailable|does not exist|is down|"
    r"not configured)\b",
    re.IGNORECASE,
)
# ⚠️ The negator has to be ADJACENT (at most two words in between), not merely
# somewhere on the line. `\bno\b[^\n]*\bdatabase\b` also matches "no rows in the
# database yet", which is a skip about DATA and would redden a run for the wrong
# reason — a guard that fires on healthy runs is a guard that gets deleted.
_DATABASE_SKIP_REASON_LEADING = re.compile(
    r"\b(no|without|missing)\s+(?:\w+\s+){0,2}(database|postgres|postgresql)\b",
    re.IGNORECASE,
)


def is_database_skip_reason(reason: str) -> bool:
    """True when ``reason`` reads as "skipped because there was no database"."""
    text = reason or ""
    return bool(
        _DATABASE_SKIP_REASON_TRAILING.search(text) or _DATABASE_SKIP_REASON_LEADING.search(text)
    )
