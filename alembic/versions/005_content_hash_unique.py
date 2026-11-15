"""Restore UNIQUE index on content_hash so the DB enforces dedup.

Revision ID: 005
Revises: 004
Create Date: 2026-07-19

Migration 003 relaxed the content_hash index to non-unique (alongside the
job_id-in-hash change), which left dedup enforced only in application code —
so a concurrent duplicate insert could still land two rows. This re-enforces
global uniqueness at the DB level: first collapse any existing duplicate
content_hash rows (keeping the oldest by created_at), then create the unique
index. The application-side IntegrityError fallback in create_sample goes live
once this index exists.

Global uniqueness is correct while this service is tenant-less (every row is
fleet-global); the future tenant hard-gate needs UNIQUE(tenant_id, content_hash).
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Delete duplicate content_hash rows, keeping the oldest (earliest
    #    created_at, then id as a deterministic tie-break). NULL content_hash
    #    rows are never deduplicated (NULLs are distinct under a unique index).
    op.execute(
        """
        DELETE FROM inference_samples
        WHERE id NOT IN (
            SELECT DISTINCT ON (content_hash) id
            FROM inference_samples
            WHERE content_hash IS NOT NULL
            ORDER BY content_hash, created_at ASC, id ASC
        )
        AND content_hash IS NOT NULL
        """
    )

    # 2. Drop any existing (non-unique or stale) content_hash indexes.
    op.execute("DROP INDEX IF EXISTS ix_samples_content_hash")
    op.execute("DROP INDEX IF EXISTS ix_samples_content_hash_unique")

    # 3. Create the unique index.
    op.create_index(
        "ix_samples_content_hash_unique",
        "inference_samples",
        ["content_hash"],
        unique=True,
    )


def downgrade() -> None:
    # Drop the unique index and restore the prior non-unique index.
    op.execute("DROP INDEX IF EXISTS ix_samples_content_hash_unique")
    op.create_index(
        "ix_samples_content_hash",
        "inference_samples",
        ["content_hash"],
        unique=False,
    )
