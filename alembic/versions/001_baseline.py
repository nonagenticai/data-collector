"""Baseline: create inference_samples table with all columns and indexes.

Revision ID: 001
Revises:
Create Date: 2026-03-24

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.has_table(conn, "inference_samples"):
        # Table already exists (pre-Alembic deployment). Ensure content_hash column
        # is present (the old raw-SQL migration may or may not have run).
        columns = {c["name"] for c in sa.inspect(conn).get_columns("inference_samples")}
        if "content_hash" not in columns:
            op.add_column(
                "inference_samples",
                sa.Column("content_hash", sa.String(64), nullable=True),
            )
            op.create_index(
                "ix_inference_samples_content_hash",
                "inference_samples",
                ["content_hash"],
            )
            op.create_index(
                "ix_samples_content_hash_unique",
                "inference_samples",
                ["content_hash"],
                unique=True,
            )
        return

    op.create_table(
        "inference_samples",
        sa.Column(
            "id",
            sa.String(36).with_variant(postgresql.UUID(as_uuid=True), "postgresql"),
            primary_key=True,
        ),
        sa.Column("schema_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column(
            "messages", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False
        ),
        sa.Column("response_content", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("validation_passed", sa.Boolean(), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), default=False),
        sa.Column("fallback_reason", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("failover_chain_id", sa.String(), nullable=False),
        sa.Column("used_in_training", sa.Boolean(), default=False),
        sa.Column("training_batch_id", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # Single-column indexes
    op.create_index("ix_inference_samples_schema_id", "inference_samples", ["schema_id"])
    op.create_index("ix_inference_samples_job_id", "inference_samples", ["job_id"])
    op.create_index("ix_inference_samples_content_hash", "inference_samples", ["content_hash"])

    # Composite / named indexes
    op.create_index(
        "ix_samples_schema_validation_training",
        "inference_samples",
        ["schema_id", "validation_passed", "used_in_training"],
    )
    op.create_index("ix_samples_created_at", "inference_samples", ["created_at"])
    op.create_index(
        "ix_samples_content_hash_unique",
        "inference_samples",
        ["content_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("inference_samples")
