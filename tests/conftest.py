"""Shared fixtures for data-collector tests.

Requires PostgreSQL. The DSN is resolved by ``tests/db_url.py`` —
``TEST_DATABASE_URL`` > ``DATABASE_URL`` > ``localhost:5451/data_collector`` —
which is now the ONLY place that answers that question for this suite. Read that
module's header for why the precondition is no longer a port probe.

====================================================================
TEST-GATE HONESTY — what fails, what skips, and why they differ
====================================================================

⚠️ A run that could not execute must never be indistinguishable from one that
passed. But "could not execute" has two shapes here, and this repo's CLAUDE.md
deliberately keeps a friendly outcome for one of them:

  PRESENT      run.
  ABSENT       ⛔ FAIL — in CI *and* locally. The server answered and simply does
               not have this database. One documented command fixes it, and a
               skip is precisely how such a provisioning gap stays invisible
               while the suite reports success.
  UNREACHABLE  FAIL in CI, SKIP locally. "There is no PostgreSQL here" is a
               developer running a reduced suite on purpose — the documented
               contract ("a no-DB pre-push doesn't hard-fail") — whereas in CI
               the archetype's ephemeral postgres:18 is a hard prerequisite, so
               its absence is an infrastructure failure that must say so.

⭐ This split is only possible because the probe distinguishes the two states.
The old guards could not: ``test_db.py`` used a port probe, and this file's
``engine`` fixture caught every connect failure into one
``pytest.skip(f"PostgreSQL not available: {e}")``. Both collapsed "no server"
and "no database" into the single outcome that looks like success.

Same shape as ``kag_graph_service/tests/conftest.py`` and
``hirag_service/tests/conftest.py`` — do NOT invent a new env var, and do NOT
route capability skips (a missing extension, a lost CREATE ROLE privilege)
through these: those are about what the environment can DO, not whether the
database exists, and failing them in CI would turn a legitimate conditional skip
into a false red.
"""

import os
from collections.abc import AsyncGenerator, Callable
from typing import Any
from uuid import uuid4

from tests.db_url import (
    ABSENT,
    PRESENT,
    UNREACHABLE,
    absent_message,
    is_database_skip_reason,
    mask,
    probe_database,
    resolve_test_database_url,
    running_in_ci,
    unreachable_message,
)

#: ⭐ ONE resolver. ``tests/test_db.py`` and
#: ``tests/test_alembic_does_not_silence_logging.py`` import this name rather
#: than re-deriving it, so the DSN the precondition proved and the DSN the tests
#: connect to are the same string by construction.
TEST_DATABASE_URL = resolve_test_database_url()

# Production Settings fail-fast requires concrete URLs; tests run in "test".
os.environ.setdefault("DATA_COLLECTOR_ENVIRONMENT", "test")

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from db import Base, get_owner_session, get_session  # noqa: E402
from main import app  # noqa: E402
from models.db import InferenceSampleDB  # noqa: E402, F401 -- register model


def decide_on_probe(state: str, detail: str, dsn: str, what: str) -> None:
    """Turn a :func:`probe_database` outcome into fail / skip / proceed.

    Returns only for ``PRESENT``; otherwise raises via ``pytest.fail`` or
    ``pytest.skip``.

    ⭐ The probe result is an ARGUMENT, not something this function measures.
    That is what makes the guard testable: every branch can be driven from a
    unit test with no PostgreSQL anywhere. A guard whose branches can only be
    observed in the situation nobody notices is not a guard.
    """
    if state == PRESENT:
        return

    if state == ABSENT:
        pytest.fail(absent_message(dsn, detail, what), pytrace=False)

    # ⛔ Fail closed on an unrecognised state. Without this an unknown value (a
    # typo, a fourth state added later) would fall through to the local-skip
    # branch below — the one outcome that makes a run look fine when nothing
    # ran, which is the defect this whole file exists to prevent.
    if state != UNREACHABLE:  # pragma: no cover - defensive
        raise AssertionError(f"unknown database-probe state {state!r} for {mask(dsn)}")

    if running_in_ci():
        pytest.fail(unreachable_message(dsn, detail, what), pytrace=False)
    pytest.skip(f"PostgreSQL not reachable at {mask(dsn)} ({detail}); skipping {what}")


def require_database(what: str = "this test", dsn: str | None = None) -> str:
    """Prove the DSN usable, then return it. Never returns on a bad state.

    ``what`` names the suite so a failure says WHICH tests did not run.
    """
    dsn = TEST_DATABASE_URL if dsn is None else dsn
    state, detail = probe_database(dsn)
    decide_on_probe(state, detail, dsn, what)
    return dsn


# ---------------------------------------------------------------------------
# The backstop: a database-shaped skip must not ride out of CI as exit 0
# ---------------------------------------------------------------------------
#
# ``decide_on_probe`` cannot produce one in CI. This catches the next one
# somebody adds — and it is deliberately CI-only, because locally such a skip is
# the documented contract, not a defect.
_DATABASE_SKIPS: list[str] = []


def _skip_reason(report: pytest.TestReport) -> str:
    """The human reason out of a skipped report, whatever shape it arrived in."""
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr or "")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped and is_database_skip_reason(_skip_reason(report)):
        _DATABASE_SKIPS.append(f"{report.nodeid}: {_skip_reason(report)}")


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ANN001, ARG001
    """Put the database state IN THE VERDICT, every run.

    Positive evidence rather than an absence: the summary says which DSN the run
    used and what state it was in, so "0 failed" can be read against "did the DB
    tests actually have a database".
    """
    state, detail = probe_database(TEST_DATABASE_URL)
    line = f"test database {mask(TEST_DATABASE_URL)}: {state.upper()}"
    terminalreporter.write_sep("-", line + (f" — {detail}" if detail else ""))
    if _DATABASE_SKIPS:
        terminalreporter.write_sep(
            "!",
            "TESTS WERE SKIPPED FOR WANT OF A DATABASE"
            + (" — this run is NOT a pass" if running_in_ci() else ""),
        )
        for entry in _DATABASE_SKIPS:
            terminalreporter.write_line(f"  {entry}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """In CI, a database-shaped skip reddens the verdict. It cannot exit 0.

    ⚠️ CI-only on purpose. Locally the same skip is this repo's documented
    contract — reddening it would break every no-DB pre-push, which is the
    behaviour CLAUDE.md keeps deliberately.
    """
    if _DATABASE_SKIPS and running_in_ci():
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply the ``unit`` marker to every non-integration test so
    ``pytest -m unit`` selects only the no-DB-needed tests."""
    unit_marker = pytest.mark.unit
    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "unit" not in markers and "integration" not in markers:
            item.add_marker(unit_marker)


# ---------------------------------------------------------------------------
# Engine & session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def engine():
    """Create a fresh PostgreSQL engine per test with brand-new tables."""
    # ⚠️ The precondition comes FIRST and is re-derived from a direct probe
    # rather than read off whatever SQLAlchemy raised. This used to be a single
    # `except Exception: pytest.skip(f"PostgreSQL not available: {e}")`, which
    # collapsed "no server" (a legitimate local skip) into the same outcome as
    # "the server is up and this database was never created" (a provisioning gap
    # that must be loud). SQLAlchemy, psycopg and psycopg_pool each wrap
    # `FATAL: database "…" does not exist` differently — and the pool does not
    # surface it at all, it retries and hands back a PoolTimeout with the cause
    # discarded — so asking the server ourselves is the only reading that cannot
    # be defeated by whichever layer happened to raise.
    require_database("the DB-backed tests")
    eng = create_async_engine(TEST_DATABASE_URL, echo=False)
    try:
        async with eng.begin() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        await eng.dispose()
        # The probe above said PRESENT, so this is a genuine failure of
        # something other than existence or reachability (privileges, an
        # extension, schema drift). Keep the original posture — loud in CI,
        # reduced run locally — rather than guessing.
        if running_in_ci():
            pytest.fail(
                f"Test database at {mask(TEST_DATABASE_URL)} answered a probe, but this "
                f"test could not use it: {e}\n\n"
                "A run that could not execute must never look like one that passed. The "
                "server IS up, so this is not the ephemeral postgres:18 container being "
                "absent — look at the error above.\n",
                pytrace=False,
            )
        pytest.skip(f"cannot use PostgreSQL at {mask(TEST_DATABASE_URL)}: {e}")

    async with eng.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest.fixture()
async def session(engine) -> AsyncGenerator[AsyncSession]:
    """Yield a fresh async session for each test."""
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as sess:
        yield sess


@pytest.fixture()
async def client(session: AsyncSession) -> AsyncGenerator[AsyncClient]:
    """HTTPX async client with the FastAPI app, using the test session."""

    async def _override_session() -> AsyncGenerator[AsyncSession]:
        yield session

    # Both the runtime and owner session dependencies resolve to the single test
    # session (one engine, connected as the owner — RLS is bypassed). Dedicated
    # enforce-mode tests (test_rls_enforce.py) exercise the two-role split with a
    # real non-owner role.
    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_owner_session] = _override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers -- no global mutable state
# ---------------------------------------------------------------------------

SAMPLE_PAYLOAD: dict[str, Any] = {
    "schema_id": "ks_EntitySpotterAgent",
    "job_id": "job-abc-123",
    "inference_id": "inf-default-001",
    "messages": [
        {"role": "system", "content": "You are an entity spotter."},
        {"role": "user", "content": "Find entities in: Apple announced new iPhones."},
    ],
    "response_content": '{"entities": [{"text": "Apple", "type": "ORG"}]}',
    "provider": "Qwen3-4B-AWQ",
    "validation_passed": True,
    "fallback_used": False,
    "fallback_reason": None,
    "latency_ms": 142,
    "failover_chain_id": "chain-001",
}


def _make_unique_payload(**overrides: Any) -> dict[str, Any]:
    """Return a sample payload with unique response_content and inference_id to avoid dedup.

    Uses uuid4 for uniqueness instead of a global counter, eliminating
    cross-test ordering dependencies.
    """
    unique_id = uuid4().hex[:12]
    p = SAMPLE_PAYLOAD.copy()
    p["response_content"] = (
        f'{{"entities": [{{"text": "Apple", "type": "ORG", "uid": "{unique_id}"}}]}}'
    )
    p["inference_id"] = f"inf-{uuid4().hex[:12]}"
    p.update(overrides)
    return p


@pytest.fixture()
def sample_payload() -> dict[str, Any]:
    """Return a valid sample creation payload with unique content."""
    return _make_unique_payload()


@pytest.fixture()
def make_sample() -> Callable[..., dict[str, Any]]:
    """Factory fixture that returns unique sample payloads each call."""
    return _make_unique_payload
