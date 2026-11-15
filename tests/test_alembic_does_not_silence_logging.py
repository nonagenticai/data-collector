"""In-process ``alembic upgrade`` must not reconfigure the service's logging.

THE DEFECT
==========
``alembic/env.py`` called ``logging.config.fileConfig(alembic.ini)``
unconditionally. ``fileConfig`` REPLACES the process-wide logging configuration
rather than extending it, and this repo's ``alembic.ini`` declares::

    [logger_root]
    level = WARN
    ...
    [handler_console]
    args = (sys.stderr,)
    [formatter_generic]
    format = %(levelname)-5.5s [%(name)s] %(message)s

``db._run_migrations`` runs migrations IN-PROCESS during lifespan startup, so
that call demoted the whole service to WARNING permanently.

MEASURED IN PRODUCTION (pod data-collector-7b6fcc5bcc-2zk29, started
2026-08-04T12:01:52Z):

    kubectl logs <pod> | wc -l                                  ->    27
    ... | grep -oE '"level": *"[a-z]+' | sort | uniq -c          ->  2 info, 13 error
    ... | grep -cE '^(WARNI|INFO |ERROR) \\['                     ->    17   <- alembic's formatter
    ... | grep -c sample_created                                 ->     0

    select count(*) from inference_samples
    where created_at >= '2026-08-04 12:01:52+00';               ->  4982

**4,982 samples ingested, 0 ``sample_created`` lines.** ``create_sample`` logs
one INFO per insert; the per-sample audit trail was 100% absent for a week. The
2 surviving info lines are both startup, emitted BEFORE migrations ran.

The tell is the FORMAT: 17 of 27 lines are alembic's ``LEVEL [name]`` layout
rather than this service's JSON renderer. The configuration was swapped, not
merely filtered.

WHY THE REAL MIGRATION PATH
===========================
These call ``db._run_alembic_upgrade`` itself — the exact function the lifespan
calls — against the test Postgres. Alembic's OFFLINE mode is not usable here:
``001_baseline`` calls ``has_table()`` so the pre-Alembic database is a no-op,
and ``has_table`` rejects the ``MockConnection`` that ``sql=True`` supplies.
Running it for real is the stronger test anyway: it exercises the production
call site, not a reconstruction of it.
"""

from __future__ import annotations

import logging
import logging.config

import pytest
from alembic import command
from alembic.config import Config

from db import _ALEMBIC_INI, _run_alembic_upgrade
from tests.conftest import TEST_DATABASE_URL, require_database

# ``_run_alembic_upgrade`` takes the DSN verbatim; psycopg is sync-capable, so
# the app's own URL works unchanged.
SYNC_DSN = TEST_DATABASE_URL


@pytest.fixture
def _requires_postgres():
    """Prove the database before running. Skips only for an ABSENT SERVER.

    The repo contract (CLAUDE.md) is that a no-DB pre-push skips cleanly instead
    of hard-failing, and that is preserved — but only for the state it was
    written for. This used to catch EVERY connection problem into one
    ``pytest.skip(f"PostgreSQL not available: {exc}")``, which also swallowed
    "the server is up and this database was never created": a provisioning gap
    with a one-command remedy, disappearing into the outcome that reads as
    success. ``conftest.require_database`` splits the two — see that module's
    header. The mention of "a developer machine whose 5451 happens to be some
    other service's container" is exactly the case the old guard got wrong: that
    port being open is not evidence the right database is behind it.
    """
    require_database("tests/test_alembic_does_not_silence_logging.py")


@pytest.fixture
def _restore_logging():
    """Save and restore global logging state.

    ``fileConfig`` mutates process-wide state; without this a failure here would
    silently break every test that runs after it in the same worker.
    """
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_disabled = {
        name: lg.disabled
        for name, lg in logging.root.manager.loggerDict.items()
        if isinstance(lg, logging.Logger)
    }
    try:
        yield
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers
        for name, was_disabled in saved_disabled.items():
            lg = logging.root.manager.loggerDict.get(name)
            if isinstance(lg, logging.Logger):
                lg.disabled = was_disabled


def _upgrade_the_way_alembic_cli_would() -> None:
    """Reproduce the DEFECT: the CLI path, which still applies alembic.ini."""
    cfg = Config(_ALEMBIC_INI)
    cfg.set_main_option("sqlalchemy.url", SYNC_DSN)
    command.upgrade(cfg, "head")


def test_in_process_upgrade_leaves_info_logging_intact(
    _requires_postgres, _restore_logging
) -> None:
    """RED BEFORE GREEN: pre-fix this fails — env.py ignored the opt-out.

    The pre-fix ``env.py`` called ``fileConfig`` unconditionally, so the root
    logger came back at WARNING (30) and every ``sample_created`` was dropped.
    """
    logging.getLogger().setLevel(logging.INFO)

    _run_alembic_upgrade(SYNC_DSN)  # the exact call the lifespan makes

    root = logging.getLogger()
    assert root.level == logging.INFO, (
        f"in-process alembic upgrade demoted the root logger to "
        f"{logging.getLevelName(root.level)}; every logger.info in the request "
        f"path is now silently dropped (4,982 samples, 0 log lines)"
    )
    assert root.isEnabledFor(logging.INFO)


def test_cli_default_still_applies_alembic_ini_logging(
    _requires_postgres, _restore_logging
) -> None:
    """NEGATIVE CONTROL: the opt-out must be opt-in, not the new default.

    Without the attribute, ``env.py`` must still honour ``alembic.ini`` so the
    standalone ``alembic`` CLI is unchanged. Passes on BOTH trees — it exists to
    show the fix narrowed behaviour for the in-process caller only.
    """
    logging.getLogger().setLevel(logging.INFO)

    _upgrade_the_way_alembic_cli_would()

    assert logging.getLogger().level == logging.WARNING, (
        "alembic.ini sets [logger_root] level = WARN; the CLI path must still apply it"
    )


def test_warning_and_above_survived_the_defect(_requires_postgres, _restore_logging) -> None:
    """CHARACTERISATION: the silencing was INFO-only, never WARNING+.

    This is why 13 error lines survived while 4,982 sample_created lines did
    not, and why any WARNING-level reading taken during the silent window is
    still trustworthy. Passes on both trees; it pins the blast radius.
    """
    logging.getLogger().setLevel(logging.INFO)

    _upgrade_the_way_alembic_cli_would()  # reproduce the defect deliberately

    root = logging.getLogger()
    assert not root.isEnabledFor(logging.INFO), "expected INFO to be suppressed"
    assert root.isEnabledFor(logging.WARNING), "WARNING must still be emitted"
    assert root.isEnabledFor(logging.ERROR), "ERROR must still be emitted"
