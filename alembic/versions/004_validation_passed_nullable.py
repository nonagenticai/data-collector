"""Relax validation_passed to nullable for tri-state semantics.

Revision ID: 004
Revises: 003
Create Date: 2026-05-21

inference-router emits validation_passed=NULL when a routing rule has no
json_schema attached (the validation step is skipped, not "passed"). Storing
that as a real NULL — rather than coercing to True — keeps downstream pass-rate
metrics honest. See plan A5 in analyze-the-architecture-eager-sunset.md.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "inference_samples",
        "validation_passed",
        nullable=True,
    )


def downgrade() -> None:
    # Coerce any NULLs back to False so the column can be NOT NULL again.
    # False (not True) is the safe choice: it keeps "skipped" rows OUT of
    # pass-rate / available-for-training aggregates instead of inflating them.
    op.execute(
        "UPDATE inference_samples SET validation_passed = FALSE WHERE validation_passed IS NULL"
    )
    op.alter_column(
        "inference_samples",
        "validation_passed",
        nullable=False,
    )
