"""SQLAlchemy async engine and session factory."""

import asyncio
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from pathlib import Path

import structlog
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from rls import register_tenant_guc
from settings import Settings

logger = structlog.stdlib.get_logger(__name__)

# Resolve alembic.ini relative to the project root (one level above src/).
_ALEMBIC_INI = str(Path(__file__).resolve().parent.parent / "alembic.ini")


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

# Owner (table-owner / RLS-bypass) engine + session factory. Distinct from the
# runtime engine ONLY when DATA_COLLECTOR_OWNER_DATABASE_URL is set and differs
# (the enforce cutover): then the runtime connects as a restricted non-owner role
# (bound by RLS) while cross-tenant flywheel reads/counts + global (NULL-tenant)
# writes run on this owner connection (which bypasses ENABLE'd RLS). When unset
# (dormant / local / tests) both point at the same engine, so nothing changes.
_owner_engine: AsyncEngine | None = None
_owner_session_factory: async_sessionmaker[AsyncSession] | None = None

# Request-scoped tenant, set by the X-Tenant-Id middleware (main.py). Read at
# transaction-begin by the RLS binding to set the app.current_tenant
# GUC. Inert until TENANT_RLS_MODE=enforce (register_tenant_guc no-ops otherwise),
# so this is pure plumbing for the future enforce cutover.
tenant_ctx: ContextVar[str | None] = ContextVar("tenant_ctx", default=None)


def _run_alembic_upgrade(database_url: str) -> None:
    """Run Alembic migrations synchronously up to head.

    Called via ``asyncio.to_thread`` so the blocking Alembic machinery
    (which internally calls ``asyncio.run``) executes in an OS thread
    without conflicting with the running event loop.

    For existing databases that pre-date Alembic (they already have the
    inference_samples table but no alembic_version table), the baseline
    migration detects the table and becomes a no-op, so this is safe to
    call on every startup.
    """
    cfg = Config(_ALEMBIC_INI)
    # Keep alembic's env.py from calling ``logging.config.fileConfig`` on
    # alembic.ini. That call REPLACES this process's logging configuration with
    # alembic's own (``[logger_root] level = WARN``, stderr handler), which would
    # silently drop every ``logger.info`` for the remaining life of the pod. See
    # the long note in ``alembic/env.py``.
    cfg.attributes["configure_logger"] = False
    # Override the URL so Alembic uses the same DSN as the application.
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")


async def init_db(settings: Settings) -> None:
    """Create the async engine and session factory.

    Must be called once during application startup (inside the lifespan
    context manager).
    """
    global _engine, _session_factory, _owner_engine, _owner_session_factory  # noqa: PLW0603

    # NullPool: pgbouncer (transaction-mode, in-cluster) does the pooling.
    # App-side pooling would double-pool and exhaust pgbouncer's
    # default_pool_size=2. Memory: feedback-pgbouncer-nullpool.
    engine_kwargs: dict[str, object] = {
        "echo": False,
        "poolclass": NullPool,
        "pool_pre_ping": True,
    }

    _engine = create_async_engine(settings.database_url, **engine_kwargs)
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Bind app.current_tenant on every transaction of the engine (covers get_session
    # plus any direct sessionmaker use — correct-by-construction). No-op while
    # TENANT_RLS_MODE is unset/off, so this changes nothing until the enforce cutover.
    register_tenant_guc(_engine, lambda: tenant_ctx.get(None))

    # Owner engine for migrations + cross-tenant flywheel sweeps + global writes.
    # A distinct engine only when an owner DSN is configured AND differs from the
    # runtime DSN (the enforce cutover). It is deliberately NOT bound with
    # register_tenant_guc: it is the RLS-bypass path (the owner bypasses ENABLE'd
    # RLS regardless of the GUC). When unset it reuses the runtime engine, so the
    # dormant/local/test path is byte-for-byte unchanged.
    owner_url = settings.owner_database_url
    if owner_url and owner_url != settings.database_url:
        _owner_engine = create_async_engine(owner_url, **engine_kwargs)
        _owner_session_factory = async_sessionmaker(
            bind=_owner_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    else:
        _owner_engine = _engine
        _owner_session_factory = _session_factory

    # Import ORM models so metadata is populated before Alembic runs.
    import models.db  # noqa: F401

    # Run Alembic migrations in a worker thread (Alembic env.py uses asyncio.run
    # internally, which cannot be called from within a running event loop).
    # Migrations run on the OWNER DSN: under enforce the runtime role is a
    # restricted non-owner with NO DDL grant, so migrating on settings.database_url
    # would CrashLoop. owner_url is settings.database_url when no owner is
    # configured, so this is unchanged while dormant.
    migrate_url = owner_url or settings.database_url
    try:
        await asyncio.to_thread(_run_alembic_upgrade, migrate_url)
    except SQLAlchemyError, OSError:
        logger.error("alembic_upgrade_failed", exc_info=True)
        raise


async def get_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency that yields a runtime (tenant-bound) session.

    Under enforce this connects as the restricted non-owner role: writes are
    GUC-scoped by the request's X-Tenant-Id. Used only for tenant-attributed
    writes; all reads and global/NULL writes use ``get_owner_session``.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised - call init_db() first")
    async with _session_factory() as session:
        yield session


async def get_owner_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency that yields an owner (RLS-bypass) session.

    Backs the system-level cross-tenant surface — every read/export/count/stats/
    mark-used/purge and global (NULL-tenant) writes + the flywheel threshold
    counts. Under enforce the owner bypasses ENABLE'd RLS so these see all
    tenants' rows; while dormant it is the same engine as ``get_session``.
    """
    if _owner_session_factory is None:
        raise RuntimeError("Database not initialised - call init_db() first")
    async with _owner_session_factory() as session:
        yield session


async def get_engine_for_check() -> bool:
    """Execute a trivial query to verify the database connection is alive."""
    if _engine is None:
        return False
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError, OSError:
        logger.error("health_check_failed", exc_info=True)
        return False


async def dispose_engine() -> None:
    """Dispose of the engine pool(s) on shutdown."""
    global _engine, _session_factory, _owner_engine, _owner_session_factory  # noqa: PLW0603
    # Dispose the owner engine first, but only when it is a distinct engine
    # (when dormant it aliases _engine, which is disposed just below).
    if _owner_engine is not None and _owner_engine is not _engine:
        await _owner_engine.dispose()
    _owner_engine = None
    _owner_session_factory = None
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
