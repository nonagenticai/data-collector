"""Deduplicate content_hash rows and re-enforce unique index.

Revision ID: 002
Revises: 001
Create Date: 2026-03-27

Deletes duplicate content_hash rows (keeping oldest by created_at) using
PostgreSQL DISTINCT ON, then drops and recreates the unique index.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Delete duplicate content_hash rows, keeping the oldest (earliest created_at).
    # The subquery finds the id of the row to keep per content_hash using DISTINCT ON.
    op.execute(
        """
        DELETE FROM inference_samples
        WHERE id NOT IN (
            SELECT DISTINCT ON (content_hash) id
            FROM inference_samples
            WHERE content_hash IS NOT NULL
            ORDER BY content_hash, created_at ASC
        )
        AND content_hash IS NOT NULL
        """
    )

    # Drop existing indexes on content_hash (if any) and create a unique one.
    # The index may not exist if this is the first migration run on a pre-Alembic DB.
    op.execute("DROP INDEX IF EXISTS ix_samples_content_hash_unique")
    op.execute("DROP INDEX IF EXISTS ix_inference_samples_content_hash")
    op.create_index(
        "ix_samples_content_hash_unique",
        "inference_samples",
        ["content_hash"],
        unique=True,
    )


def downgrade() -> None:
    # Downgrade is a no-op: deleted rows cannot be restored, and the index
    # already existed before this migration.
    pass
