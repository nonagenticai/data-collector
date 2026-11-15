"""
Alembic configuration for asynchronous migrations.

This file sets up an async engine for running migrations, using the
DATABASE_URL environment variable when available.
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

import models.db  # noqa: F401  # ensures models are imported for metadata
from db import Base

# Load logging configuration
config = context.config
# ⚠️ ``fileConfig`` REPLACES the process-wide logging configuration; it does not
# extend it. ``alembic.ini`` here carries ``[logger_root] level = WARN``, a
# stderr handler, and the ``%(levelname)-5.5s [%(name)s]`` formatter.
#
# Fine for the short-lived ``alembic`` CLI. NOT fine for ``db._run_migrations``,
# which runs migrations IN-PROCESS during lifespan startup: every
# ``logger.info`` for the rest of the pod's life is silently dropped, and the
# service's JSON renderer is swapped for alembic's.
#
# Verified on the sibling service (inference-router#101): a pod with 24 h uptime
# serving hundreds of requests emitted 24 log lines, 3 of them info, all from
# startup BEFORE migrations ran. The tell is the FORMAT — production lines
# looked like ``WARNI [collector_client] {...}`` rather than the service's JSON.
# WARNING and above keep flowing, so the damage is INFO-only.
#
# The in-process caller opts out via ``cfg.attributes["configure_logger"]``
# (Alembic's documented idiom for programmatic use); the default stays True so
# the CLI is unchanged.
try:
    if config.config_file_name is not None and config.attributes.get("configure_logger", True):
        fileConfig(config.config_file_name)
except KeyError:
    pass

# Use the metadata from our models
target_metadata = Base.metadata


def _resolve_db_url() -> str | None:
    """Pick the DB URL from the same env var the running service uses.

    Prefer DATA_COLLECTOR_DATABASE_URL (what the deployment sets) and fall back
    to DATABASE_URL for local / CI use. The alembic.ini sqlalchemy.url only
    applies when neither env var is set.
    """
    return os.getenv("DATA_COLLECTOR_DATABASE_URL") or os.getenv("DATABASE_URL")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = _resolve_db_url() or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section) or {}
    db_url = _resolve_db_url() or configuration.get("sqlalchemy.url")

    # Convert postgresql:// to postgresql+psycopg:// for async driver
    if db_url and db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    configuration["sqlalchemy.url"] = db_url or ""

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
