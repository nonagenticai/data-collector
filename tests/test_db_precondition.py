"""The DB precondition must distinguish "no database" from "no server".

`tests/test_db.py` guarded itself with a module-level
`socket.create_connection((host, port))` + `pytest.skip(allow_module_level=True)`,
and `conftest.engine` caught every connect failure into one
`pytest.skip(f"PostgreSQL not available: {e}")`.

Against this repo's OWN docker-compose Postgres (`docker-compose.override.yml`,
`5451:5432`) the port probe was SOUND: a repo-private port is open iff that
container — and so the database inside it — exists. Against a SHARED or
host-native PostgreSQL it is a guard that cannot fail, because the port is open
with no database, no tables and no valid credentials. That is the future this
file closes; today it is future-proofing, not a live bug.

⭐ The part that is NOT merely future-proofing is the collapse. Both old guards
gave "no server" and "server up, database never created" the SAME outcome — a
skip — and those two have different remedies. This repo's CLAUDE.md deliberately
keeps the friendly skip for the first; the second is a provisioning gap that one
command closes, and a skip is how such a gap hides while the run reports
success.

⚠️ These tests deliberately need no CREATE DATABASE privilege. A name that was
never created answers the question just as well as one we made and dropped, and
requiring the privilege would put a new precondition on CI's role — i.e. it
would reintroduce the class of defect this file exists to close.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import pytest
from _pytest.outcomes import Failed, Skipped

from tests.conftest import decide_on_probe, require_database
from tests.db_url import (
    ABSENT,
    DEFAULT_TEST_DATABASE_URL,
    MAINTENANCE_DATABASES,
    PRESENT,
    UNREACHABLE,
    absent_message,
    classify_connect_failure,
    createdb_command,
    database_name,
    is_database_skip_reason,
    libpq_dsn,
    maintenance_dsns,
    mask,
    probe_database,
    resolve_test_database_url,
    resolved_source,
    running_in_ci,
    unreachable_message,
)

CONFIGURED_DSN = resolve_test_database_url()

# Synthetic. A database name nothing has ever created, on the very server this
# suite is configured to use — so "the port is open" and "the database exists"
# come apart under measurement rather than in prose.
#
# ⚠️ Built from CONFIGURED_DSN, NOT from `libpq_dsn(CONFIGURED_DSN)`. That
# matters because `TestTheGateIsActuallyWiredIn` feeds this DSN to a real pytest
# run: stripping `+psycopg` leaves a bare `postgresql://`, SQLAlchemy then reaches
# for psycopg2, and `create_async_engine` dies on `ModuleNotFoundError` before any
# database question is asked. Measured — with the driver stripped, a mutant that
# deletes the precondition still produced a red run, for a reason that had nothing
# to do with the guard, and SURVIVED a check that was only looking at the colour.
_parts = urlsplit(CONFIGURED_DSN)
ABSENT_DSN = urlunsplit(_parts._replace(path=f"/dc_absent_{uuid.uuid4().hex}"))

# A port nothing listens on, on loopback.
DEAD_DSN = urlunsplit(
    _parts._replace(netloc=f"{_parts.username or 'postgres'}@127.0.0.1:1", path="/whatever")
)


def _hostport(dsn: str) -> tuple[str, int]:
    parts = urlsplit(libpq_dsn(dsn))
    return (parts.hostname or "127.0.0.1", parts.port or 5432)


def _summary_line(output: str) -> str:
    r"""pytest's final counts line, isolated from everything above it.

    ⛔ Never grep the whole output for "failed" / "error". libpq's own message
    for an unreachable server reads `connection to server at "127.0.0.1", port 1
    failed: ...` — so a substring check for " failed" fires on a perfectly clean
    skip, and even the digit-anchored `\d+ (failed|error)` matches the literal
    "1 failed" inside "port 1 failed". Both produced a false red here, on a run
    whose real summary was "2 passed, 4 skipped".

    The counts line is the assertion surface; the body is prose written by
    somebody else's error formatter.
    """
    lines = [ln.strip(" =") for ln in output.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


@pytest.fixture
def live_server() -> str:
    """Some tests below need a real PostgreSQL. Gate them with the SAME contract
    this repo applies to every other DB-backed test.

    ⛔ Without this the new module broke the contract it exists to defend.
    CLAUDE.md defines the local gate as the WHOLE-SUITE `pytest`; measured on a
    box with no server at all, `TEST_DATABASE_URL=...@127.0.0.1:1/data_collector`:

        before this PR   71 passed, 207 skipped                exit 0
        with the module   7 failed, 132 passed, 262 skipped    exit 1

    All 7 were here. The PR's own arms table said "UNREACHABLE, local -> exit 0,
    contract preserved" — true of the file subset it measured, false of the gate
    the contract actually names. A subset measurement is not a suite property.

    Routed through `require_database` rather than a bespoke reachability check on
    purpose: the gate under test is then the gate guarding its own tests, so
    these skip locally with no server, fail in CI, and fail everywhere when the
    server is up and the database is missing.
    """
    return require_database("tests/test_db_precondition.py")


class TestTheResolverIsTheOnlyResolver:
    """One DSN, or the precondition and the tests can disagree about what it proved."""

    def test_precedence_is_test_database_url_then_database_url_then_default(self) -> None:
        assert (
            resolve_test_database_url(
                {"TEST_DATABASE_URL": "postgresql://a/x", "DATABASE_URL": "postgresql://b/y"}
            )
            == "postgresql://a/x"
        )
        assert resolve_test_database_url({"DATABASE_URL": "postgresql://b/y"}) == "postgresql://b/y"
        assert resolve_test_database_url({}) == DEFAULT_TEST_DATABASE_URL

    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_a_blank_value_is_treated_as_unset(self, value: str) -> None:
        """`VAR=` in a shell profile or a CI matrix is a far likelier accident
        than a deliberate request to connect to the empty string."""
        assert resolve_test_database_url({"TEST_DATABASE_URL": value}) == DEFAULT_TEST_DATABASE_URL
        assert (
            resolve_test_database_url(
                {"TEST_DATABASE_URL": value, "DATABASE_URL": "postgresql://b/y"}
            )
            == "postgresql://b/y"
        )

    def test_the_ci_archetypes_database_url_is_honoured(self) -> None:
        """⭐ The drift that had already happened. `tests/test_db.py` read only
        `TEST_DATABASE_URL`, so under the archetype — which exports
        `DATABASE_URL` and never sets `TEST_DATABASE_URL` — it resolved to
        `localhost:5451`, a port no CI runner has open, and the whole module
        skipped on the port probe while the rest of the suite ran against the
        real CI database."""
        ci_env = {"DATABASE_URL": "postgresql+psycopg://ci_test@127.0.0.1:5432/data_collector_ci"}
        assert resolve_test_database_url(ci_env) == ci_env["DATABASE_URL"]
        assert resolved_source(ci_env) == "$DATABASE_URL"

    def test_every_consumer_imports_the_same_name(self) -> None:
        """A second copy is how the drift above happened. Assert there is one."""
        import tests.conftest as ct
        import tests.test_alembic_does_not_silence_logging as alembic_test
        import tests.test_db as test_db

        assert test_db.TEST_DATABASE_URL is ct.TEST_DATABASE_URL
        assert alembic_test.SYNC_DSN is ct.TEST_DATABASE_URL


class TestTheThreeStates:
    """PRESENT / ABSENT / UNREACHABLE have different remedies and must not collapse."""

    @pytest.mark.usefixtures("live_server")
    def test_the_configured_database_is_present(self) -> None:
        """Positive control. Without this every assertion below is vacuous: an
        instrument that reported ABSENT for everything would pass them all."""
        state, detail = probe_database(CONFIGURED_DSN)
        assert state == PRESENT, f"{mask(CONFIGURED_DSN)} -> {state}: {detail}"

    @pytest.mark.usefixtures("live_server")
    def test_a_database_that_was_never_created_is_absent_not_unreachable(self) -> None:
        state, detail = probe_database(ABSENT_DSN)
        assert state == ABSENT, f"{mask(ABSENT_DSN)} -> {state}: {detail}"

    @pytest.mark.usefixtures("live_server")
    def test_the_open_port_alone_would_have_said_yes(self) -> None:
        """⭐ The defect, reproduced. The replaced probe passes on the ABSENT DSN.

        Same host, same port, a database that does not exist. `test_db.py`'s
        `_postgres_reachable` returned True here and the module did not skip.
        """
        host, port = _hostport(ABSENT_DSN)
        with socket.create_connection((host, port), timeout=3) as sock:
            assert sock is not None  # the old `_postgres_reachable` returned True

        state, _ = probe_database(ABSENT_DSN)
        assert state == ABSENT

    def test_a_dead_server_is_unreachable(self) -> None:
        state, detail = probe_database(DEAD_DSN, connect_timeout=2)
        assert state == UNREACHABLE, f"{state}: {detail}"

    @pytest.mark.usefixtures("live_server")
    def test_a_missing_role_is_not_reported_as_a_missing_database(self) -> None:
        """Different remedy. Telling someone to `createdb` a database they
        already have, because their credentials are wrong, is worse than silence.
        """
        parts = urlsplit(libpq_dsn(CONFIGURED_DSN))
        netloc = (
            f"dc_no_such_role_{uuid.uuid4().hex[:12]}@"
            f"{parts.hostname or '127.0.0.1'}:{parts.port or 5432}"
        )
        state, detail = probe_database(urlunsplit(parts._replace(netloc=netloc)))
        assert state == UNREACHABLE, f"{state}: {detail}"

    def test_libpqs_missing_role_wording_is_classified_as_unreachable(self) -> None:
        """⚠️ The test above does NOT reach the missing-ROLE branch on every
        server, and a mutation run proved it: against a container configured for
        scram-sha-256 the server answers `password authentication failed`, never
        `role "…" does not exist`, so classification falls through to the
        structural probe and the branch is never executed. A mutant flipping that
        branch to ABSENT therefore SURVIVED. Drive the branch with the message
        directly instead of hoping the environment produces it."""
        state, detail = classify_connect_failure(
            CONFIGURED_DSN,
            OSError('connection failed: FATAL:  role "dc_no_such_role" does not exist'),
        )
        assert state == UNREACHABLE, detail
        assert "ROLE" in detail


class TestClassificationIsNotAGuess:
    """An unrecognised libpq message is a question for the server, not a guess."""

    @pytest.mark.usefixtures("live_server")
    def test_an_unrecognised_message_over_an_existing_database_is_unreachable(self) -> None:
        """libpq translates its messages; a non-English `lc_messages` matches
        nothing. The structural `pg_database` lookup is what keeps that safe."""
        state, detail = classify_connect_failure(
            CONFIGURED_DSN, OSError("FATAL: la base de données est introuvable")
        )
        assert state == UNREACHABLE, detail
        assert "exists" in detail

    @pytest.mark.usefixtures("live_server")
    def test_an_unrecognised_message_over_a_missing_database_is_still_absent(self) -> None:
        state, detail = classify_connect_failure(
            ABSENT_DSN, OSError("FATAL: la base de données est introuvable")
        )
        assert state == ABSENT, detail
        assert "pg_database" in detail

    def test_an_unanswerable_server_stays_unreachable_rather_than_guessing(self) -> None:
        """No maintenance database answered ⇒ existence is UNKNOWN. UNKNOWN is
        reported as UNREACHABLE with the reason attached — never as ABSENT."""
        state, detail = classify_connect_failure(
            DEAD_DSN, OSError("something untranslatable"), connect_timeout=2
        )
        assert state == UNREACHABLE
        assert "UNKNOWN" in detail

    def test_the_maintenance_database_is_not_hardcoded(self) -> None:
        """A scoped role often cannot open `postgres`, so several maintenance
        databases are tried and the target is never one of them.

        ⛔ Asserted against a SYNTHETIC DSN, and on the intersection with
        `MAINTENANCE_DATABASES` — not `len(candidates) > 1` over the configured
        DSN. The candidate list also appends the DSN's OWN USERNAME as a
        fallback, so that bound is satisfied by the fallback alone whenever the
        username differs from a maintenance name. Measured on this module with
        the mutant `MAINTENANCE_DATABASES = ("postgres",)`:

            DSN user `postgres` -> ['postgres']               len 1  -> KILLED
            DSN user `ci_test`  -> ['postgres', 'ci_test']    len 2  -> SURVIVED

        The throwaway container used to verify #43 runs as `postgres`; the CI
        archetype runs as `ci_test`. So the mutant died under the instance it was
        graded on and would have lived in the instance that matters — the matrix
        reported the site covered because of who the DSN happened to log in as.

        Same class as `test_libpqs_missing_role_wording_...` above, and found the
        same way: by re-running the matrix against a second environment. See
        discovery-engine#76.
        """
        synthetic = "postgresql+psycopg://appuser@127.0.0.1:5432/some_ci"
        candidates = [database_name(dsn) for dsn in maintenance_dsns(synthetic)]
        assert len(set(candidates) & set(MAINTENANCE_DATABASES)) > 1, candidates
        assert "appuser" in candidates, f"the username fallback was dropped: {candidates}"
        assert database_name(synthetic) not in candidates, candidates
        # And the real DSN still never probes itself.
        assert database_name(CONFIGURED_DSN) not in [
            database_name(dsn) for dsn in maintenance_dsns(CONFIGURED_DSN)
        ]


class TestTheDecisionSplitsTheTwoFailures:
    """⭐ The whole point. A missing DATABASE and a missing SERVER are not the
    same event, and this repo answers them differently on purpose."""

    @pytest.fixture
    def not_ci(self, monkeypatch: pytest.MonkeyPatch):
        import tests.conftest as ct

        monkeypatch.setattr(ct, "running_in_ci", lambda *a, **k: False)

    @pytest.fixture
    def in_ci(self, monkeypatch: pytest.MonkeyPatch):
        import tests.conftest as ct

        monkeypatch.setattr(ct, "running_in_ci", lambda *a, **k: True)

    @staticmethod
    def _raised(state: str) -> BaseException:
        """Run the decision and hand back what it raised.

        ⚠️ `pytest.raises(pytest.fail.Exception)` would let a `Skipped` escape
        uncaught, so a mutant downgrading a hard failure to a skip would read as
        `exit 0` rather than red. Catch BaseException; the TYPE is the assertion.
        """
        with pytest.raises(BaseException) as excinfo:  # noqa: PT011 - the type IS the assertion
            decide_on_probe(state, "synthetic", ABSENT_DSN, "the proof")
        return excinfo.value

    def test_an_absent_database_fails_locally(self, not_ci) -> None:
        """⛔ The one branch that is NOT "fail in CI, skip locally". The server
        answered; one documented command fixes it, and a skip is how that gap
        stayed invisible."""
        raised = self._raised(ABSENT)
        assert isinstance(raised, Failed), f"expected Failed, got {type(raised).__name__}"
        assert not isinstance(raised, Skipped)

    def test_an_absent_database_fails_in_ci(self, in_ci) -> None:
        raised = self._raised(ABSENT)
        assert isinstance(raised, Failed)
        assert not isinstance(raised, Skipped)

    def test_an_unreachable_server_skips_locally(self, not_ci) -> None:
        """The documented contract (CLAUDE.md): a no-DB pre-push does not
        hard-fail. Preserved deliberately — this is the negative control that
        stops the change from being "make everything red"."""
        raised = self._raised(UNREACHABLE)
        assert isinstance(raised, Skipped), f"expected Skipped, got {type(raised).__name__}"

    def test_an_unreachable_server_fails_in_ci(self, in_ci) -> None:
        """In CI the ephemeral postgres:18 is a hard prerequisite; its absence is
        an infrastructure failure, and "N skipped, exit 0" is byte-for-byte a
        green run."""
        raised = self._raised(UNREACHABLE)
        assert isinstance(raised, Failed), f"expected Failed, got {type(raised).__name__}"
        assert not isinstance(raised, Skipped)

    @pytest.mark.parametrize("ci", [True, False])
    def test_a_present_database_proceeds(self, monkeypatch: pytest.MonkeyPatch, ci: bool) -> None:
        """Negative control: the gate is not simply always-fail."""
        import tests.conftest as ct

        monkeypatch.setattr(ct, "running_in_ci", lambda *a, **k: ci)
        assert decide_on_probe(PRESENT, "", CONFIGURED_DSN, "the proof") is None

    def test_an_unknown_state_fails_closed(self, not_ci) -> None:
        """⛔ Without this an unrecognised state falls through to the local-skip
        branch — the one outcome that makes a run look fine when nothing ran.

        ⚠️ `pytest.raises(AssertionError)` alone is NOT enough here, and a
        mutation run proved it: `Skipped` derives from `BaseException`, not
        `Exception`, so with the guard deleted it escapes `pytest.raises`
        uncaught, the test reports as SKIPPED, and the mutant reads as `exit 0`.
        Catch BaseException; the TYPE is the assertion.
        """
        with pytest.raises(BaseException) as excinfo:  # noqa: PT011 - the type IS the assertion
            decide_on_probe("wat", "synthetic", ABSENT_DSN, "the proof")

        assert isinstance(excinfo.value, AssertionError), (
            f"expected AssertionError, got {type(excinfo.value).__name__}"
        )
        assert not isinstance(excinfo.value, Skipped)
        assert "unknown database-probe state" in str(excinfo.value)

    @pytest.mark.usefixtures("live_server")
    def test_require_database_returns_the_dsn_it_proved(self) -> None:
        """A probe that can disagree with the tests it gates is worse than no
        probe.

        ⚠️ Gated on `live_server`, NOT on the `not_ci` monkeypatch it used to
        carry. This test really calls `require_database`, so with no server it
        needs an outcome — and forcing `running_in_ci()` to False gave it a
        LOCAL skip even when running IN CI. Measured: a CI run with no server
        came back `130 passed, 1 skipped, 270 errors`, and that 1 was this test
        producing exactly the database-shaped skip the backstop exists to catch.
        The run was already red and the backstop did flag it, so nothing was
        hidden — but a test that manufactures the very signal its own module
        polices is noise in the one report that must stay legible.
        """
        import tests.conftest as ct

        assert require_database("the proof") == ct.TEST_DATABASE_URL

    def test_require_database_actually_gates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """⚠️ The assertion above passes for a `require_database` that never
        probes at all — measured: a mutant replacing its body with `pass`
        SURVIVED it, because the DSN it returns is right either way. Prove it
        consults the probe and honours the answer."""
        import tests.conftest as ct

        calls: list[str] = []

        def _probe(dsn, **kwargs):  # noqa: ANN001, ANN202, ARG001
            calls.append(dsn)
            return (ABSENT, "synthetic")

        monkeypatch.setattr(ct, "probe_database", _probe)
        monkeypatch.setattr(ct, "running_in_ci", lambda *a, **k: False)

        with pytest.raises(BaseException) as excinfo:  # noqa: PT011 - the type IS the assertion
            ct.require_database("the proof")

        assert isinstance(excinfo.value, Failed), (
            f"expected Failed, got {type(excinfo.value).__name__}"
        )
        assert calls == [ct.TEST_DATABASE_URL], "the DSN probed is not the DSN returned"


class TestRunningInCi:
    def test_github_actions_is_detected_without_a_bespoke_variable(self) -> None:
        assert running_in_ci({"CI": "true"})
        assert running_in_ci({"GITHUB_ACTIONS": "true"})

    @pytest.mark.parametrize("env", [{}, {"CI": "false"}, {"CI": "1"}, {"GITHUB_ACTIONS": ""}])
    def test_a_developer_box_is_not_ci(self, env: dict) -> None:
        """GitHub Actions sets these to the literal string "true"; anything else
        is a local shell and must keep the friendly skip."""
        assert not running_in_ci(env)


class TestTheMessagesAreActionable:
    def test_the_absent_message_names_the_dsn_the_database_and_the_remedy(self) -> None:
        message = absent_message(ABSENT_DSN, "synthetic", "the proof")
        assert database_name(ABSENT_DSN) in message
        assert "the proof" in message
        assert createdb_command(ABSENT_DSN) in message

    def test_the_remedy_provisions_the_database_the_suite_will_connect_to(self) -> None:
        """⚠️ Asserting `createdb_command(dsn) in message` grades a COPY: both
        sides move together, so a mutant hardcoding a database name into it
        satisfies them both. Measured on the sibling repo — it survived. Derive
        the expected host, port and name from the DSN instead."""
        command = createdb_command(ABSENT_DSN)
        parts = urlsplit(libpq_dsn(ABSENT_DSN))
        assert command.split()[-1] == database_name(ABSENT_DSN)
        assert f"-h {parts.hostname}" in command
        assert f"-p {parts.port}" in command

    def test_the_unreachable_remedy_is_a_different_remedy(self) -> None:
        """Collapsing the two states would hand a developer whose server is down
        a `createdb` they cannot run."""
        message = unreachable_message(ABSENT_DSN, "synthetic", "the proof")
        assert "UNREACHABLE" in message
        assert createdb_command(ABSENT_DSN) not in message

    @pytest.mark.parametrize("render", [absent_message, unreachable_message])
    def test_the_password_never_reaches_the_message(self, render) -> None:
        leaky = "postgresql+psycopg://u:n0t-a-real-pw-dc@127.0.0.1:5432/data_collector"  # pragma: allowlist secret
        message = render(leaky, "synthetic", "the proof")
        assert "n0t-a-real-pw-dc" not in message
        assert "127.0.0.1:5432/data_collector" in message  # the line was emitted at all


class TestTheVerdictGuard:
    """A database-shaped skip must redden a CI run, not ride out as exit 0."""

    def test_the_exact_reasons_these_guards_produced_are_caught(self) -> None:
        assert is_database_skip_reason("PostgreSQL not available: connection refused")
        assert is_database_skip_reason(
            "Postgres at postgresql://x:5451/data_collector not reachable"
        )

    @pytest.mark.parametrize(
        "reason",
        [
            'database "data_collector" does not exist',
            "no test database configured",
            "PostgreSQL not reachable at postgresql://x (OperationalError)",
        ],
    )
    def test_other_database_shaped_skips_are_caught(self, reason: str) -> None:
        assert is_database_skip_reason(reason)

    @pytest.mark.parametrize(
        "reason",
        [
            "requires network access",
            "async backend not installed",
            "",
            "skipped on purpose: the database fixture is intentionally unused here",
            # ⚠️ The leading-negation pattern must keep its negator ADJACENT to
            # the noun. Both of these carry "no" AND "database" on one line and
            # are about DATA, not about the database being absent.
            "no rows in the database yet, nothing to assert",
            "no samples were returned, so the database assertions are moot",
        ],
    )
    def test_unrelated_skips_are_left_alone(self, reason: str) -> None:
        """Without this the guard would redden every run with any skip in it,
        which is a guard nobody keeps."""
        assert not is_database_skip_reason(reason)


class TestTheBackstopMovesTheVerdict:
    """⭐ Recording a skip is not the same as reddening the run.

    `is_database_skip_reason` only CLASSIFIES; `pytest_sessionfinish` is the only
    thing that turns a classification into a non-zero exit code. Without a test
    here a mutant deleting the `session.exitstatus` assignment survives every
    other assertion in this file — the registry still fills, the summary still
    prints, and CI still exits 0.
    """

    @pytest.fixture
    def conftest_module(self):
        import tests.conftest as ct

        saved = list(ct._DATABASE_SKIPS)
        ct._DATABASE_SKIPS.clear()
        yield ct
        ct._DATABASE_SKIPS[:] = saved

    @staticmethod
    def _skipped(reason: str) -> SimpleNamespace:
        # pytest hands a skip's reason through as the third element of the
        # (path, lineno, message) longrepr tuple.
        return SimpleNamespace(
            skipped=True, nodeid="tests/fake.py::t", longrepr=("fake.py", 1, reason)
        )

    @staticmethod
    def _session() -> SimpleNamespace:
        return SimpleNamespace(config=SimpleNamespace(), exitstatus=pytest.ExitCode.OK)

    def test_in_ci_a_database_shaped_skip_forces_a_failing_exit_code(
        self, conftest_module, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ct = conftest_module
        monkeypatch.setattr(ct, "running_in_ci", lambda *a, **k: True)
        ct.pytest_runtest_logreport(self._skipped("Skipped: PostgreSQL not available: boom"))
        assert ct._DATABASE_SKIPS, "the skip was not even recorded"

        session = self._session()
        ct.pytest_sessionfinish(session, 0)
        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED

    def test_locally_the_same_skip_leaves_the_verdict_alone(
        self, conftest_module, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """⚠️ CI-only on purpose. Locally this skip is the documented contract;
        reddening it would break every no-DB pre-push."""
        ct = conftest_module
        monkeypatch.setattr(ct, "running_in_ci", lambda *a, **k: False)
        ct.pytest_runtest_logreport(self._skipped("Skipped: PostgreSQL not available: boom"))
        assert ct._DATABASE_SKIPS

        session = self._session()
        ct.pytest_sessionfinish(session, 0)
        assert session.exitstatus == pytest.ExitCode.OK

    def test_an_unrelated_skip_is_not_even_recorded(
        self, conftest_module, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative control: the backstop is not simply always-fail in CI."""
        ct = conftest_module
        monkeypatch.setattr(ct, "running_in_ci", lambda *a, **k: True)
        ct.pytest_runtest_logreport(self._skipped("Skipped: requires network access"))
        assert not ct._DATABASE_SKIPS

        session = self._session()
        ct.pytest_sessionfinish(session, 0)
        assert session.exitstatus == pytest.ExitCode.OK


class _FakeReporter:
    """The two methods `pytest_terminal_summary` uses, and a record of what it said."""

    def __init__(self, stats: dict | None = None) -> None:
        self.stats = stats or {}
        self.lines: list[str] = []

    def write_sep(self, sep: str, title: str = "", **kwargs) -> None:  # noqa: ARG002
        self.lines.append(title)

    def write_line(self, line: str, **kwargs) -> None:  # noqa: ARG002
        self.lines.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class TestTheVerdictLineIsInTheVerdict:
    """The state of the database has to appear in the summary of every run."""

    @pytest.mark.usefixtures("live_server")
    def test_it_names_the_dsn_and_the_state(self) -> None:
        import tests.conftest as ct

        reporter = _FakeReporter()
        ct.pytest_terminal_summary(reporter, 0, None)
        assert "PRESENT" in reporter.text
        assert database_name(CONFIGURED_DSN) in reporter.text

    def test_the_password_never_reaches_the_verdict_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tests.conftest as ct

        leaky = "postgresql+psycopg://u:n0t-a-real-pw-dc@127.0.0.1:5432/data_collector"  # pragma: allowlist secret
        monkeypatch.setattr(ct, "TEST_DATABASE_URL", leaky)
        monkeypatch.setattr(ct, "probe_database", lambda *a, **k: (PRESENT, ""))
        reporter = _FakeReporter()
        ct.pytest_terminal_summary(reporter, 0, None)
        assert "n0t-a-real-pw-dc" not in reporter.text
        assert "data_collector" in reporter.text


class TestTheGateIsActuallyWiredIn:
    """⭐ Every assertion above grades `decide_on_probe` / `require_database` in
    isolation. None of them notices if a FIXTURE stops calling them — measured:
    mutants that replaced `conftest.engine`'s and `test_db.py`'s precondition
    calls with `pass` both SURVIVED the whole file.

    So this runs a real pytest, in a subprocess, against a database that does
    not exist on a server that is up, with CI unset — the exact situation the
    old guards turned into a silent skip — and asserts the three DB-backed entry
    points come back RED with the remedy attached.
    """

    #: One node id per distinct precondition call site, so a mutant that removes
    #: any single one of them is caught rather than masked by its siblings.
    ENTRY_POINTS = (
        # conftest.engine — the `pytest.skip(f"PostgreSQL not available: {e}")`
        "tests/test_export.py::TestExport::test_export_empty",
        # test_db.py._reset_db_globals — the module-level port probe
        "tests/test_db.py::TestInitDb::test_init_db_creates_tables",
        # the alembic test's own _requires_postgres
        "tests/test_alembic_does_not_silence_logging.py::test_warning_and_above_survived_the_defect",
        # test_rls_enforce.enforce_env — the RLS tenant-isolation proof, whose
        # own `pytest.skip(f"PostgreSQL not available: {e}")` was a THIRD guard
        # of the same class. Measured before the fix, database ABSENT on a live
        # server: `14 skipped, exit 0`.
        "tests/test_rls_enforce.py::TestAntiDrift::test_rls_enabled_not_forced",
    )

    @staticmethod
    def _run_against(dsn: str) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items() if k not in ("CI", "GITHUB_ACTIONS")}
        env["TEST_DATABASE_URL"] = dsn
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *TestTheGateIsActuallyWiredIn.ENTRY_POINTS,
                "-p",
                "no:cacheprovider",
                "--no-cov",
                "-q",
                "--no-header",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    def test_the_absent_dsn_is_one_the_app_can_actually_build_an_engine_from(self) -> None:
        """⛔ Guard on the instrument, not the subject. If ABSENT_DSN loses the
        `+psycopg` driver, `create_async_engine` dies on `ModuleNotFoundError`
        before the precondition is ever consulted — the run below is then red
        for a reason that has nothing to do with the guard, and a mutant that
        deletes the guard survives a check that only looks at the colour."""
        assert ABSENT_DSN.startswith("postgresql+psycopg://"), ABSENT_DSN
        assert DEAD_DSN.startswith("postgresql+psycopg://"), DEAD_DSN

    @pytest.mark.slow
    @pytest.mark.usefixtures("live_server")
    def test_an_absent_database_reddens_every_entry_point(self) -> None:
        result = self._run_against(ABSENT_DSN)
        output = result.stdout + result.stderr

        assert "ModuleNotFoundError" not in output, (
            f"the run failed before reaching the precondition:\n{output}"
        )
        assert result.returncode != 0, f"exit 0 against an absent database:\n{output}"
        assert createdb_command(ABSENT_DSN) in output, output
        # ⛔ The defect was a SKIP, not a failure. "N skipped" and exit 0 is
        # byte-for-byte a green run, so assert none of them vanished — read off
        # the counts line, never the body (see `_summary_line`).
        assert "skipped" not in _summary_line(output), f"something still skipped:\n{output}"
        for node in self.ENTRY_POINTS:
            assert node.split("::")[-1] in output, f"{node} never reported:\n{output}"

        # ⭐ COUNT the refusals, one per entry point — do not merely assert the
        # phrase is present somewhere. Red is not the property; red WITH THE
        # DIAGNOSIS is. A call site that loses its precondition still goes red,
        # on a raw `sqlalchemy.exc.OperationalError` naming neither the DSN
        # resolution order nor the remedy — and a presence check is satisfied by
        # its two surviving siblings. Measured: the mutant that deletes
        # `test_db.py`'s call SURVIVED exactly that presence check.
        refusals = output.count("The test database does not exist")
        assert refusals == len(self.ENTRY_POINTS), (
            f"expected one named refusal per entry point "
            f"({len(self.ENTRY_POINTS)}), got {refusals} — a call site is bypassing "
            f"the precondition and failing raw:\n{output}"
        )
        assert "sqlalchemy.exc.OperationalError" not in output, (
            f"a raw driver traceback escaped the gate:\n{output}"
        )

    @pytest.mark.slow
    @pytest.mark.usefixtures("live_server")
    def test_the_same_entry_points_pass_against_the_real_database(self) -> None:
        """Positive control. Without it the assertion above is satisfied by a
        suite that is simply broken."""
        result = self._run_against(CONFIGURED_DSN)
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert f"{len(self.ENTRY_POINTS)} passed" in output, output


class TestThisModuleHonoursTheContractItDefends:
    """⭐ The new module must not break the gate CLAUDE.md actually names.

    That gate is the WHOLE-SUITE `pytest`, and these tests need a live server —
    so without the `live_server` fixture routing them through `require_database`
    they hard-fail on a box with no PostgreSQL. Measured, `CI` unset,
    `TEST_DATABASE_URL=…@127.0.0.1:1/data_collector`:

        before this PR    71 passed, 207 skipped              exit 0
        module ungated     7 failed, 132 passed, 262 skipped  exit 1
        module gated     130 passed, 271 skipped              exit 0

    ⚠️ The PR's arms table claimed "contract preserved" off three OTHER files.
    A subset measurement is not a suite property, and this class is the suite
    property — asserted, not asserted-about.

    ⛔ Deliberately runs `TestTheThreeStates`, which spawns no subprocess of its
    own. Pointing it at this whole module would recurse.
    """

    SUBSET = "tests/test_db_precondition.py::TestTheThreeStates"
    NO_SERVER_DSN = DEAD_DSN

    @staticmethod
    def _run(dsn: str, *, in_ci: bool) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items() if k not in ("CI", "GITHUB_ACTIONS")}
        env["TEST_DATABASE_URL"] = dsn
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if in_ci:
            env["CI"] = "true"
            env["GITHUB_ACTIONS"] = "true"
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                TestThisModuleHonoursTheContractItDefends.SUBSET,
                "-p",
                "no:cacheprovider",
                "--no-cov",
                "-q",
                "--no-header",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    @pytest.mark.slow
    def test_with_no_server_these_tests_skip_and_the_run_stays_green(self) -> None:
        """The documented contract: a no-DB pre-push does not hard-fail."""
        result = self._run(self.NO_SERVER_DSN, in_ci=False)
        output = result.stdout + result.stderr
        assert result.returncode == 0, f"the no-DB contract is broken:\n{output}"
        summary = _summary_line(output)
        assert re.search(r"\d+ skipped", summary), output
        assert not re.search(r"\d+ (failed|error)", summary), output

    @pytest.mark.slow
    def test_with_no_server_in_ci_the_same_tests_are_loud(self) -> None:
        """Negative control. If they skipped here too, the gate would be
        satisfied by never running anywhere — which is the original defect."""
        result = self._run(self.NO_SERVER_DSN, in_ci=True)
        output = result.stdout + result.stderr
        assert result.returncode != 0, f"CI went green with no database:\n{output}"
