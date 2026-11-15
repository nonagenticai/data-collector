"""Add outcome + finish_reason + tool_calls to inference_samples.

Revision ID: 008
Revises: 007
Create Date: 2026-08-25

``response_content`` is ``TEXT NOT NULL``, so three completely different
upstream realities all land in this table as ``''``:

1. a healthy **tool-call reply** -- the answer is in upstream's ``tool_calls``
   and ``content`` is legitimately ``""`` (~97% of the blank rows);
2. a **breaker rejection** -- never dispatched at all (``latency_ms = 0``);
3. a **genuine blank completion** -- ``finish_reason = 'length'``, budget burned.

No column separated them, and this table IS the training corpus
(``used_in_training`` / ``training_batch_id``), so the ambiguity propagates
straight into training data. Three nullable columns fix that:

- ``outcome``       -- inference-router's exhaustive partition (ok | empty |
  empty_result | error | invalid | aborted). Deliberately ``String`` and NOT a
  PG enum: the router owns the vocabulary, and adding a label must never require
  a migration plus a deploy-ordering dance.
- ``finish_reason`` -- upstream's own value (stop | length | tool_calls | ...),
  which turns an ``empty`` outcome into an actual diagnosis.
- ``tool_calls``    -- the upstream tool-call payload (JSONB). For tool-calling
  traffic this is the ANSWER; it was previously written nowhere.

All three are NULLABLE with NO server_default and there is NO backfill: NULL
means "written by a pre-R22 inference-router", which is a fact worth keeping
rather than papering over with a guessed value. No index -- nothing filters on
them yet, and an unused index is only write cost.

Deliberately NOT touched:
- ``response_content`` nullability, and every other existing column.
- ``_compute_content_hash`` inputs. The hash feeds the UNIQUE index
  ``ix_samples_tenant_content_hash_unique``; adding a field would silently
  change dedup volume and therefore training volume.
- Migration 007's RLS policies. This migration emits no POLICY / SECURITY /
  raw ALTER TABLE statement of its own. 007's policies are rendered from
  ``rls`` and predicate solely on ``tenant_id``; ``ADD
  COLUMN`` neither drops nor re-renders them, so they survive untouched (and
  keep applying to the new columns automatically, since a policy scopes rows,
  not columns).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql as pg

# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _existing_columns() -> set[str]:
    """Columns already on ``inference_samples``, or an empty set when offline.

    The idempotency guard exists because ``db._run_alembic_upgrade`` also runs
    against databases built by ``Base.metadata.create_all`` (which already has
    these columns) and then stamped -- the same reason 007 guards ``tenant_id``.

    Offline (``alembic upgrade --sql``) there is no connection to inspect.
    Returning an empty set makes the generated script emit the real ``ADD
    COLUMN`` statements. Reflecting instead would either raise (what 007 does)
    or, worse, silently emit an EMPTY script that looks like a successful
    no-op migration.
    """
    if context.is_offline_mode():
        return set()
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns("inference_samples")}


def upgrade() -> None:
    columns = _existing_columns()

    if "outcome" not in columns:
        op.add_column(
            "inference_samples",
            sa.Column("outcome", sa.String(), nullable=True),
        )
    if "finish_reason" not in columns:
        op.add_column(
            "inference_samples",
            sa.Column("finish_reason", sa.String(), nullable=True),
        )
    if "tool_calls" not in columns:
        op.add_column(
            "inference_samples",
            sa.Column("tool_calls", pg.JSONB(astext_type=sa.Text()), nullable=True),
        )


def downgrade() -> None:
    # Pure drops, reverse order. Nothing else to unwind: no index, no default,
    # no backfill, and no RLS change was made on the way up. Offline, all three
    # are assumed present so the generated script emits the real DROPs.
    columns = _existing_columns() or {"outcome", "finish_reason", "tool_calls"}

    if "tool_calls" in columns:
        op.drop_column("inference_samples", "tool_calls")
    if "finish_reason" in columns:
        op.drop_column("inference_samples", "finish_reason")
    if "outcome" in columns:
        op.drop_column("inference_samples", "outcome")
