"""Add inference_id column, include job_id in dedup, fix index uniqueness.

Revision ID: 003
Revises: 002
Create Date: 2026-04-14

Changes:
- Add inference_id column (NOT NULL, with default for existing rows)
- Drop unique constraint on content_hash (replace with non-unique index)
- Add index on inference_id
- Add composite index on (job_id, schema_id)
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add inference_id column as nullable first, backfill, then make NOT NULL.
    conn = op.get_bind()
    columns = {c["name"] for c in sa.inspect(conn).get_columns("inference_samples")}

    if "inference_id" not in columns:
        op.add_column(
            "inference_samples",
            sa.Column("inference_id", sa.String(), nullable=True),
        )

        # Backfill existing rows with a unique inference_id per row.
        # Use the existing id column as a reasonable default.
        op.execute("UPDATE inference_samples SET inference_id = id WHERE inference_id IS NULL")

        # Now make it NOT NULL.
        op.alter_column(
            "inference_samples",
            "inference_id",
            nullable=False,
        )

    # 2. Drop the unique content_hash index and replace with non-unique.
    op.execute("DROP INDEX IF EXISTS ix_samples_content_hash_unique")
    op.execute("DROP INDEX IF EXISTS ix_inference_samples_content_hash")
    op.execute("DROP INDEX IF EXISTS ix_samples_content_hash")

    op.create_index(
        "ix_samples_content_hash",
        "inference_samples",
        ["content_hash"],
        unique=False,
    )

    # 3. Add inference_id index.
    op.execute("DROP INDEX IF EXISTS ix_samples_inference_id")
    op.create_index(
        "ix_samples_inference_id",
        "inference_samples",
        ["inference_id"],
    )

    # 4. Add composite job_id + schema_id index.
    op.execute("DROP INDEX IF EXISTS ix_samples_job_schema")
    op.create_index(
        "ix_samples_job_schema",
        "inference_samples",
        ["job_id", "schema_id"],
    )


def downgrade() -> None:
    # Remove new indexes.
    op.execute("DROP INDEX IF EXISTS ix_samples_job_schema")
    op.execute("DROP INDEX IF EXISTS ix_samples_inference_id")

    # Restore unique content_hash index.
    op.execute("DROP INDEX IF EXISTS ix_samples_content_hash")
    op.create_index(
        "ix_samples_content_hash_unique",
        "inference_samples",
        ["content_hash"],
        unique=True,
    )

    # Remove inference_id column.
    op.drop_column("inference_samples", "inference_id")
