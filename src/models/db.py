"""SQLAlchemy ORM models for the data-collector service."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from db import Base
from rls import register_rls_ddl

# Postgres-native column types. This service is postgres-only (house rule: no
# sqlite); tests run against a real Postgres, so no cross-backend variants.
_JsonColumn = JSONB
# ⛔ NULLABLE JSON NEEDS ``none_as_null=True``. SQLAlchemy's JSON/JSONB default is
# ``none_as_null=False``, which writes Python ``None`` as the JSON scalar ``'null'``
# rather than SQL NULL. The ORM then deserialises that back to ``None``, so every
# Python-level assertion still passes while the DATABASE holds two different
# representations of "absent" -- and ``WHERE tool_calls IS NOT NULL``, which is the
# whole point of the column, matches EVERY row written by this code. Pre-cutover
# rows are genuine SQL NULL, so the two would coexist in the corpus forever.
# ``messages`` above is ``nullable=False`` and never sees ``None``, which is why
# this surface did not exist before the first nullable JSON column.
_NullableJsonColumn = JSONB(none_as_null=True)
_UuidColumn = PG_UUID(as_uuid=True)


class InferenceSampleDB(Base):
    """Persisted inference sample row."""

    __tablename__ = "inference_samples"

    id: Mapped[str] = mapped_column(_UuidColumn, primary_key=True, default=uuid7)
    # Nullable by design: NULL = fleet-global sample (shared cross-tenant, e.g.
    # every pre-gate row + any untagged writer). A real tenant's samples carry
    # its UUID. Under the "global-or-own" RLS policy this makes NULL
    # rows readable by every tenant (the flywheel's cross-tenant training export)
    # while tagged writes are GUC-scoped. Inert until TENANT_RLS_MODE=enforce.
    tenant_id: Mapped[UUID | None] = mapped_column(_UuidColumn, nullable=True, index=True)
    schema_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    inference_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    messages: Mapped[list[dict[str, Any]]] = mapped_column(_JsonColumn, nullable=False)
    response_content: Mapped[str] = mapped_column(Text, nullable=False)
    # --- What actually happened upstream -------------------------------------
    # ``response_content = ''`` is ambiguous by itself: a healthy tool-call reply
    # (the answer lives in ``tool_calls``, ~97% of blank rows), a breaker
    # rejection that was never dispatched (latency_ms = 0), and a genuine blank
    # completion (finish_reason = 'length', budget burned) are three completely
    # different realities that all land as ''. Since this table IS the training
    # corpus (used_in_training / training_batch_id), it needs to tell them apart.
    #
    # ``outcome`` is a plain String, NOT a PG enum, on purpose: inference-router
    # owns the partition (ok / empty / empty_result / error / invalid / aborted)
    # and a new label must never require a migration + deploy ordering dance.
    outcome: Mapped[str | None] = mapped_column(String, nullable=True)
    # Upstream's own finish_reason (stop | length | tool_calls | ...). This is
    # what turns an ``empty`` outcome into an actual diagnosis.
    finish_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    # The upstream tool-call payload. For tool-calling traffic this IS the
    # answer, and it was previously written nowhere at all.
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(
        _NullableJsonColumn, nullable=True
    )
    provider: Mapped[str] = mapped_column(String, nullable=False)
    validation_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    fallback_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    failover_chain_id: Mapped[str] = mapped_column(String, nullable=False)
    used_in_training: Mapped[bool] = mapped_column(Boolean, default=False)
    training_batch_id: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "ix_samples_schema_validation_training",
            "schema_id",
            "validation_passed",
            "used_in_training",
        ),
        Index("ix_samples_created_at", "created_at"),
        # Dedup is per-tenant: UNIQUE(tenant_id, content_hash). NULLS NOT DISTINCT
        # keeps global (NULL-tenant) rows deduped on content_hash — without it PG
        # treats NULLs as distinct and mig 005's dedup would silently regress.
        # Partial WHERE keeps NULL-content_hash rows out of the constraint (matches
        # 005: a NULL hash is "no dedup key", not a colliding value).
        Index(
            "ix_samples_tenant_content_hash_unique",
            "tenant_id",
            "content_hash",
            unique=True,
            postgresql_nulls_not_distinct=True,
            postgresql_where=text("content_hash IS NOT NULL"),
        ),
        Index("ix_samples_inference_id", "inference_id"),
        Index("ix_samples_job_schema", "job_id", "schema_id"),
    )


class TriggerProgressDB(Base):
    """Persisted high-water mark for rising-edge training triggers.

    One row per trigger key (``"__global__"`` for the aggregate flywheel,
    ``"schema:<schema_id>"`` per schema). ``last_count`` records the most
    recently observed validated/unused count so a trigger fires exactly once
    on the rising edge across the threshold — even when a concurrent insert
    makes the count overshoot the threshold in a single step (which the old
    exact-equality check silently missed).
    """

    __tablename__ = "trigger_progress"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    last_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# Apply the SAME RLS policies on ``create_all`` (tests/dev, which build the schema
# without Alembic) that migration 007 applies in prod — both render from
# ``rls`` so they cannot drift. Only ``inference_samples``
# carries ``tenant_id``; ``trigger_progress`` has none and is auto-excluded, so the
# flywheel's aggregate/global training counts stay cross-tenant (correct). The
# policies are inert until the runtime connects as a restricted role with
# ``TENANT_RLS_MODE=enforce`` (see CLAUDE.md enforce-cutover).
register_rls_ddl(Base.metadata)
