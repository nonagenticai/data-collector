"""Add trigger_progress marker table for rising-edge training triggers.

Revision ID: 006
Revises: 005
Create Date: 2026-07-19

The training triggers used exact-equality on the threshold (``available ==
threshold``), which silently missed the crossing whenever a single insert made
the count overshoot (e.g. a concurrent insert lands two rows, jumping from
threshold-1 to threshold+1). This table persists the last observed
validated/unused count per trigger key so the trigger fires exactly once on the
rising edge (``prev < threshold <= available``).

On first deploy, seed the markers to the CURRENT counts so an already-past
threshold does not spuriously fire a catch-up trigger on the next insert.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Idempotent create: like the baseline, tolerate a DB where the table
    # already exists (e.g. alembic_version was reset but app tables were not).
    conn = op.get_bind()
    if "trigger_progress" not in set(sa.inspect(conn).get_table_names()):
        op.create_table(
            "trigger_progress",
            sa.Column("key", sa.Text(), nullable=False),
            sa.Column(
                "last_count",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.PrimaryKeyConstraint("key"),
        )

    # Seed markers to current counts so a deploy against an already-past
    # threshold does not fire a spurious catch-up trigger on the next insert.
    op.execute(
        """
        INSERT INTO trigger_progress (key, last_count)
        SELECT '__global__', count(*)
        FROM inference_samples
        WHERE validation_passed IS TRUE AND used_in_training IS FALSE
        ON CONFLICT (key) DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO trigger_progress (key, last_count)
        SELECT 'schema:' || schema_id, count(*)
        FROM inference_samples
        WHERE validation_passed IS TRUE AND used_in_training IS FALSE
        GROUP BY schema_id
        ON CONFLICT (key) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_table("trigger_progress")
