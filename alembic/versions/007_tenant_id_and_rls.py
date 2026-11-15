"""Add nullable tenant_id + per-tenant dedup + dormant RLS (tenant hard-gate).

Revision ID: 007
Revises: 006
Create Date: 2026-07-19

Lands the multi-tenancy HARD GATE (data-collector/CLAUDE.md) in DORMANT form:

- ``tenant_id`` (nullable UUID) on ``inference_samples``. NULL = fleet-global
  sample (shared cross-tenant — every pre-gate row + any untagged writer); a real
  tenant's samples carry its UUID.
- Dedup becomes per-tenant: the global ``UNIQUE(content_hash)`` (mig 005) is
  replaced by ``UNIQUE(tenant_id, content_hash) NULLS NOT DISTINCT WHERE
  content_hash IS NOT NULL``. NULLS NOT DISTINCT keeps global (NULL-tenant) rows
  deduped on content_hash — without it PG treats NULLs as distinct and mig 005's
  dedup would silently regress. The partial WHERE keeps NULL-content_hash rows out
  (a NULL hash is "no dedup key", matching 005).
- Row-Level-Security policies ("global-or-own": NULL rows readable by all,
  writes GUC-scoped) are applied but INERT: they only bind once the runtime
  connects as a restricted non-owner role with TENANT_RLS_MODE=enforce. Migrations
  and the current owner connection bypass RLS (ENABLE-not-FORCE), so runtime
  behavior is unchanged until the enforce cutover. Rendered from the same
  ``rls`` helpers as the create_all path so the two cannot drift.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

import models.db  # noqa: F401  # populate Base.metadata before rendering RLS DDL
from db import Base
from rls import downgrade_statements, upgrade_statements

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    columns = {c["name"] for c in sa.inspect(conn).get_columns("inference_samples")}

    # 1. Add nullable tenant_id + its lookup index (idempotent).
    if "tenant_id" not in columns:
        op.add_column(
            "inference_samples",
            sa.Column("tenant_id", pg.UUID(as_uuid=True), nullable=True),
        )
    op.execute("DROP INDEX IF EXISTS ix_samples_tenant_id")
    op.create_index("ix_samples_tenant_id", "inference_samples", ["tenant_id"])

    # 2. Replace the global content_hash unique index with the per-tenant one.
    #    Raw DDL: NULLS NOT DISTINCT + a partial predicate are not expressible via
    #    op.create_index kwargs across all versions, so emit the exact SQL.
    op.execute("DROP INDEX IF EXISTS ix_samples_content_hash_unique")
    op.execute("DROP INDEX IF EXISTS ix_samples_tenant_content_hash_unique")
    op.execute(
        "CREATE UNIQUE INDEX ix_samples_tenant_content_hash_unique "
        "ON inference_samples (tenant_id, content_hash) "
        "NULLS NOT DISTINCT WHERE content_hash IS NOT NULL"
    )

    # 3. Apply RLS policies (inert until enforce). inference_samples is the only
    #    tenant_id table; trigger_progress has none and is auto-excluded, so the
    #    flywheel's aggregate/global training counts stay cross-tenant.
    for stmt in upgrade_statements(Base.metadata):
        op.execute(stmt)


def downgrade() -> None:
    # Reverse RLS, then the per-tenant index, then restore the global unique index,
    # then drop tenant_id.
    for stmt in downgrade_statements(Base.metadata):
        op.execute(stmt)

    op.execute("DROP INDEX IF EXISTS ix_samples_tenant_content_hash_unique")
    op.execute("DROP INDEX IF EXISTS ix_samples_tenant_id")

    op.execute("DROP INDEX IF EXISTS ix_samples_content_hash_unique")
    op.create_index(
        "ix_samples_content_hash_unique",
        "inference_samples",
        ["content_hash"],
        unique=True,
    )

    op.drop_column("inference_samples", "tenant_id")
