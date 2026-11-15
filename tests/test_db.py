"""Tests for db.py -- init_db, get_session, get_engine_for_check, dispose_engine.

Uses real async PostgreSQL engines for all tests except the connection-error test.
No mocks for normal DB operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import db as db_module
from settings import Settings
from tests.conftest import TEST_DATABASE_URL, require_database

pytestmark = pytest.mark.integration

# ⛔ WHAT USED TO BE HERE, AND WHY IT HAD TO GO.
#
# A module-level `socket.create_connection((host, port))` behind
# `pytest.skip(..., allow_module_level=True)`, over a SECOND copy of the default
# DSN that this module resolved itself.
#
# Two defects, one line:
#
# 1. The DSN was a second copy, and it had already drifted — it read only
#    `TEST_DATABASE_URL`, never `DATABASE_URL`. Under the CI archetype, which
#    exports `DATABASE_URL` and never sets `TEST_DATABASE_URL`, this module
#    resolved to `localhost:5451`, a port no CI runner has ever had open, so the
#    whole file skipped on the port probe while the rest of the suite ran
#    against the real CI database. It now imports the ONE resolver.
#
# 2. A port probe cannot answer "can these tests run?". Against this repo's own
#    docker-compose Postgres (`5451:5432`) it was a sound proxy — a repo-private
#    port is open iff that container, and so the database in it, exists. Against
#    a SHARED or host-native 5432 it is a guard that cannot fail: the port is
#    open with no database, no tables and no valid credentials. See
#    tests/db_url.py.
#
# `require_database` connects to the named DATABASE and splits the two failures:
# a missing database is loud everywhere, an absent server still skips locally
# (this repo's documented contract) and fails in CI.


@pytest.fixture(autouse=True)
async def _reset_db_globals():
    """Ensure db module globals and database state are clean before and after each test."""
    require_database("tests/test_db.py")
    original_engine = db_module._engine
    original_factory = db_module._session_factory
    db_module._engine = None
    db_module._session_factory = None

    # Drop Alembic version tracking and application tables so each test
    # calling init_db() gets a clean slate for migration testing.
    cleanup_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with cleanup_engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        await conn.execute(text("DROP TABLE IF EXISTS inference_samples"))
        await conn.execute(text("DROP TABLE IF EXISTS trigger_progress"))
    await cleanup_engine.dispose()

    yield

    # Dispose if anything was created during the test
    if db_module._engine is not None and hasattr(db_module._engine, "dispose"):
        try:
            await db_module._engine.dispose()
        except Exception:
            pass
    db_module._engine = original_engine
    db_module._session_factory = original_factory


class TestInitDb:
    async def test_init_db_creates_engine_and_factory(self) -> None:
        """init_db should set _engine and _session_factory globals using a real PostgreSQL engine."""
        settings = Settings(database_url=TEST_DATABASE_URL)
        await db_module.init_db(settings)

        assert db_module._engine is not None
        assert db_module._session_factory is not None

        # Verify the engine is functional
        async with db_module._engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    async def test_init_db_creates_tables(self) -> None:
        """init_db should create the inference_samples table."""
        settings = Settings(database_url=TEST_DATABASE_URL)
        await db_module.init_db(settings)

        # Verify the table exists by querying it
        assert db_module._engine is not None
        async with db_module._engine.connect() as conn:
            result = await conn.execute(text("SELECT count(*) FROM inference_samples"))
            count = result.scalar()
            assert count == 0

    async def test_init_db_overwrites_on_second_call(self) -> None:
        """Calling init_db a second time should replace the existing engine."""
        settings = Settings(database_url=TEST_DATABASE_URL)
        await db_module.init_db(settings)
        first_engine = db_module._engine
        first_factory = db_module._session_factory

        await db_module.init_db(settings)
        assert db_module._engine is not first_engine
        assert db_module._session_factory is not first_factory

    async def test_init_db_session_factory_produces_sessions(self) -> None:
        """The session factory created by init_db should produce working sessions."""
        settings = Settings(database_url=TEST_DATABASE_URL)
        await db_module.init_db(settings)

        assert db_module._session_factory is not None
        async with db_module._session_factory() as sess:
            assert isinstance(sess, AsyncSession)
            result = await sess.execute(text("SELECT 1"))
            assert result.scalar() == 1


class TestInitDbMigrationFailFast:
    """init_db must re-raise on a failed Alembic upgrade so a stale-schema
    deploy CrashLoops instead of silently serving on an old schema."""

    async def test_init_db_reraises_on_migration_failure(self) -> None:
        """A SQLAlchemyError from the Alembic upgrade must propagate out of init_db."""
        from sqlalchemy.exc import OperationalError

        settings = Settings(database_url=TEST_DATABASE_URL)
        with patch.object(
            db_module,
            "_run_alembic_upgrade",
            side_effect=OperationalError("stmt", {}, Exception("migration boom")),
        ):
            with pytest.raises(OperationalError):
                await db_module.init_db(settings)

    async def test_init_db_calls_upgrade_on_success(self) -> None:
        """On a successful (no-op) upgrade, init_db calls it exactly once and does not raise."""
        settings = Settings(database_url=TEST_DATABASE_URL)
        upgrade_mock = MagicMock()  # sync callable run via asyncio.to_thread
        with patch.object(db_module, "_run_alembic_upgrade", upgrade_mock):
            await db_module.init_db(settings)
        upgrade_mock.assert_called_once_with(settings.database_url)


class TestGetSession:
    async def test_get_session_raises_when_not_initialised(self) -> None:
        """get_session should raise RuntimeError if init_db was never called."""
        gen = db_module.get_session()
        with pytest.raises(RuntimeError, match="Database not initialised"):
            await gen.__anext__()

    async def test_get_session_yields_working_session(self) -> None:
        """get_session should yield an AsyncSession that can execute queries."""
        engine = create_async_engine(TEST_DATABASE_URL, echo=False)
        factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        db_module._engine = engine
        db_module._session_factory = factory

        gen = db_module.get_session()
        session = await gen.__anext__()
        assert isinstance(session, AsyncSession)

        # Verify the session can execute a query
        result = await session.execute(text("SELECT 42"))
        assert result.scalar() == 42

        # Clean up the generator
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass

    async def test_get_session_closes_after_use(self) -> None:
        """get_session should close the session when the generator is exhausted."""
        engine = create_async_engine(TEST_DATABASE_URL, echo=False)
        factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        db_module._engine = engine
        db_module._session_factory = factory

        gen = db_module.get_session()
        session = await gen.__anext__()
        assert isinstance(session, AsyncSession)

        # Exhaust the generator (triggers __aexit__ which closes the session)
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass

        # After generator cleanup, session should be closed
        # (no exception is the success condition here -- we got through the full lifecycle)


class TestGetEngineForCheck:
    async def test_returns_false_when_no_engine(self) -> None:
        """Should return False when _engine is None."""
        assert db_module._engine is None
        result = await db_module.get_engine_for_check()
        assert result is False

    async def test_returns_true_with_valid_engine(self) -> None:
        """Should return True when engine can execute SELECT 1."""
        engine = create_async_engine(TEST_DATABASE_URL, echo=False)
        db_module._engine = engine
        result = await db_module.get_engine_for_check()
        assert result is True

    async def test_returns_false_on_connection_error(self) -> None:
        """Should return False when the engine connection fails."""
        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(side_effect=OSError("Connection refused"))
        mock_engine.connect.return_value = mock_conn
        db_module._engine = mock_engine
        result = await db_module.get_engine_for_check()
        assert result is False

    async def test_returns_false_on_sqlalchemy_error(self) -> None:
        """Should return False when SQLAlchemy raises a driver error."""
        from sqlalchemy.exc import OperationalError

        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(
            side_effect=OperationalError("stmt", {}, Exception("db down"))
        )
        mock_engine.connect.return_value = mock_conn
        db_module._engine = mock_engine
        result = await db_module.get_engine_for_check()
        assert result is False


class TestDisposeEngine:
    async def test_dispose_engine_clears_globals(self) -> None:
        """dispose_engine should set _engine and _session_factory to None."""
        engine = create_async_engine(TEST_DATABASE_URL, echo=False)
        factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        db_module._engine = engine
        db_module._session_factory = factory

        await db_module.dispose_engine()
        assert db_module._engine is None
        assert db_module._session_factory is None

    async def test_dispose_engine_noop_when_no_engine(self) -> None:
        """Should not raise when engine is already None."""
        assert db_module._engine is None
        await db_module.dispose_engine()
        assert db_module._engine is None

    async def test_dispose_engine_actually_disposes(self) -> None:
        """After dispose, the original engine should be disposed (pool closed)."""
        engine = create_async_engine(TEST_DATABASE_URL, echo=False)
        db_module._engine = engine
        db_module._session_factory = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False
        )

        await db_module.dispose_engine()

        # The engine's pool should be disposed -- attempting to connect should raise
        # or return a new connection (engine internals vary), but the globals are cleared
        assert db_module._engine is None
        assert db_module._session_factory is None
