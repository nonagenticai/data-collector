"""Business logic for inference sample CRUD, export, and stats."""

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid7

import httpx
import structlog
from sqlalchemy import CursorResult, case, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from models.db import InferenceSampleDB, TriggerProgressDB
from models.domain import (
    ExportParams,
    InferenceSample,
    MarkUsedRequest,
    PurgeResponse,
    SampleCountResponse,
    SampleCreate,
    SampleFilter,
    SampleStats,
)
from settings import get_settings

logger = structlog.stdlib.get_logger(__name__)


def _compute_content_hash(
    schema_id: str, job_id: str, messages: list[dict[str, Any]], response_content: str
) -> str:
    """Compute SHA-256 of schema_id + job_id + messages + response_content for deduplication.

    Including job_id ensures that identical content from different jobs is stored
    separately, preventing cross-job deduplication that breaks correlation.

    ⛔ DO NOT ADD FIELDS HERE. This hash feeds the UNIQUE index
    ``ix_samples_tenant_content_hash_unique``, so any new input silently changes
    how much traffic dedups away -- and therefore how much training data this
    corpus holds. ``outcome`` / ``finish_reason`` / ``tool_calls`` are excluded
    on purpose: two identical completions that merely differ in outcome
    labelling are still ONE sample. Pinned by
    ``tests/test_sample_service.py::TestContentHashIgnoresOutcomeFields``.
    """
    canonical = (
        schema_id
        + "\0"
        + job_id
        + "\0"
        + json.dumps(messages, sort_keys=True, separators=(",", ":"))
        + "\0"
        + response_content
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _fire_training_trigger(url: str, schema_id: str, available_samples: int) -> None:
    """POST a training trigger notification. Runs as a fire-and-forget task.

    All exceptions are caught so a webhook failure never impacts sample ingestion.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json={"schema_id": schema_id, "available_samples": available_samples},
            )
            resp.raise_for_status()
        logger.info(
            "training_trigger_sent",
            schema_id=schema_id,
            status_code=resp.status_code,
        )
    except httpx.HTTPError:
        logger.warning(
            "training_trigger_failed",
            schema_id=schema_id,
            exc_info=True,
        )


async def _fire_global_training_trigger(url: str, available_samples: int) -> None:
    """POST a system-wide single-LoRA retrain trigger. Fire-and-forget."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={"available_samples": available_samples})
            resp.raise_for_status()
        logger.info("global_training_trigger_sent", status_code=resp.status_code)
    except httpx.HTTPError:
        logger.warning("global_training_trigger_failed", exc_info=True)


async def _check_training_threshold(session: AsyncSession, schema_id: str) -> None:
    """Fire the webhook on the rising edge across the per-schema threshold.

    Fires exactly once when the validated/unused count crosses the threshold —
    i.e. when the previously observed count was below it and the current count
    is at or above it (``prev < threshold <= available``). A persisted
    ``trigger_progress`` marker (locked ``FOR UPDATE``) records the last observed
    count, so a concurrent insert that makes the count overshoot the threshold in
    a single step still fires (the old exact-equality check silently missed that).

    Re-arming needs an observed count strictly below the threshold: after
    training consumes samples (dropping the count), a subsequent refill back up
    to the threshold fires again. A partial refill that only reaches exactly the
    threshold without the count first being observed below it will not refire,
    which is correct for real batch consumption.

    The DB query runs inline (fast), and only the HTTP POST is fire-and-forget.
    All exceptions are caught so this never impacts sample ingestion.
    """
    try:
        settings = get_settings()
        if not settings.training_trigger_enabled:
            return

        threshold = settings.training_threshold

        count_stmt = (
            select(func.count())
            .select_from(InferenceSampleDB)
            .where(InferenceSampleDB.schema_id == schema_id)
            .where(InferenceSampleDB.validation_passed.is_(True))
            .where(InferenceSampleDB.used_in_training.is_(False))
        )
        result = await session.execute(count_stmt)
        available = result.scalar_one()

        key = f"schema:{schema_id}"
        marker = await session.get(TriggerProgressDB, key, with_for_update=True)
        prev = marker.last_count if marker is not None else 0
        if marker is None:
            session.add(TriggerProgressDB(key=key, last_count=available))
        else:
            marker.last_count = available
        await session.commit()

        if not (prev < threshold <= available):
            return

        logger.info(
            "training_threshold_reached",
            schema_id=schema_id,
            available_samples=available,
            threshold=threshold,
            trigger_url=settings.training_trigger_url,
        )

        # Fire-and-forget: schedule the HTTP POST without blocking sample creation
        asyncio.create_task(
            _fire_training_trigger(settings.training_trigger_url, schema_id, available),
            name=f"training-trigger-{schema_id}",
        )
    except SQLAlchemyError:
        await session.rollback()
        logger.warning(
            "training_threshold_check_failed",
            schema_id=schema_id,
            exc_info=True,
        )


async def _check_global_training_threshold(session: AsyncSession) -> None:
    """Fire on the rising edge across the AGGREGATE (system-wide) global threshold.

    The flywheel for the system-wide single-LoRA adapter is event-based, NOT
    time-based: when the aggregated count of validated/unused samples (across
    every schema) crosses ``global_training_threshold``, fire one HTTP POST to
    training-data-generator's `/api/triggers/fire-global` and let it run a
    `mode=global` batch + training job in the background.

    Fires exactly once on the rising-edge crossing
    (``prev < threshold <= available``), using the persisted ``__global__``
    ``trigger_progress`` marker (locked ``FOR UPDATE``). A concurrent insert that
    makes the aggregate count overshoot the threshold in a single step still
    fires — the old exact-equality check silently missed that. Re-arming needs an
    observed count strictly below the threshold (training marking samples used
    drops the count); a partial refill to exactly the threshold won't refire,
    which is fine for real batch consumption.
    """
    try:
        settings = get_settings()
        if not settings.global_training_trigger_enabled:
            return
        if not settings.global_training_trigger_url:
            return

        threshold = settings.global_training_threshold

        count_stmt = (
            select(func.count())
            .select_from(InferenceSampleDB)
            .where(InferenceSampleDB.validation_passed.is_(True))
            .where(InferenceSampleDB.used_in_training.is_(False))
        )
        result = await session.execute(count_stmt)
        available = result.scalar_one()

        key = "__global__"
        marker = await session.get(TriggerProgressDB, key, with_for_update=True)
        prev = marker.last_count if marker is not None else 0
        if marker is None:
            session.add(TriggerProgressDB(key=key, last_count=available))
        else:
            marker.last_count = available
        await session.commit()

        if not (prev < threshold <= available):
            return

        logger.info(
            "global_training_threshold_reached",
            available_samples=available,
            threshold=threshold,
            trigger_url=settings.global_training_trigger_url,
        )

        asyncio.create_task(
            _fire_global_training_trigger(settings.global_training_trigger_url, available),
            name="global-training-trigger",
        )
    except SQLAlchemyError:
        await session.rollback()
        logger.warning("global_training_threshold_check_failed", exc_info=True)


async def create_sample(
    session: AsyncSession,
    payload: SampleCreate,
    tenant_id: UUID | None = None,
    *,
    owner_session: AsyncSession | None = None,
) -> InferenceSample:
    """Persist a new inference sample and return the domain model.

    Deduplicates by (tenant_id, content hash) — if a sample with the same
    messages + response_content already exists for the same tenant, return the
    existing row instead. Dedup is per-tenant so two tenants can independently
    hold identical content; ``tenant_id`` NULL (fleet-global) rows dedup among
    themselves. Matches the ``UNIQUE(tenant_id, content_hash) NULLS NOT DISTINCT``
    index.

    Engine routing (enforce cutover): ``owner_session`` is the RLS-bypass
    connection. A **global** (``tenant_id is None``) write goes on it, because
    under enforce the restricted runtime role's INSERT ``WITH CHECK
    (tenant_id = current_setting('app.current_tenant'))`` rejects a NULL row.
    A **tenant-attributed** write goes on ``session`` (the runtime role) so the
    row is GUC-scoped and isolated. The flywheel threshold counts always run on
    ``owner_session`` so the aggregate stays cross-tenant (the runtime role would
    scope them to one tenant and silently under-count). When ``owner_session`` is
    None (dormant / local / tests, one engine) everything uses ``session`` — no
    behavior change.
    """
    owner = owner_session if owner_session is not None else session
    # Global (unattributed) rows can only be written by the owner under enforce;
    # tenant rows go on the runtime role where WITH CHECK enforces isolation.
    write_session = owner if tenant_id is None else session

    content_hash = _compute_content_hash(
        payload.schema_id, payload.job_id, payload.messages, payload.response_content
    )

    # Check for duplicate (use .first() instead of scalar_one_or_none to tolerate
    # pre-existing duplicate rows until the dedup migration has run).
    # ``is_not_distinct_from`` renders NULL-safe ``IS NOT DISTINCT FROM`` so a
    # NULL tenant matches a NULL tenant (the global bucket).
    dup_stmt = (
        select(InferenceSampleDB)
        .where(InferenceSampleDB.content_hash == content_hash)
        .where(InferenceSampleDB.tenant_id.is_not_distinct_from(tenant_id))
        .order_by(InferenceSampleDB.created_at.asc())
        .limit(1)
    )
    dup_result = await write_session.execute(dup_stmt)
    existing = dup_result.scalars().first()
    if existing is not None:
        logger.info(
            "sample_deduplicated",
            existing_id=str(existing.id),
            schema_id=payload.schema_id,
            provider=payload.provider,
        )
        return _row_to_domain(existing)

    sample_id = uuid7()

    row = InferenceSampleDB(
        id=str(sample_id),
        tenant_id=tenant_id,
        schema_id=payload.schema_id,
        job_id=payload.job_id,
        inference_id=payload.inference_id,
        messages=payload.messages,
        response_content=payload.response_content,
        # Deliberately NOT part of content_hash -- see _compute_content_hash.
        outcome=payload.outcome,
        finish_reason=payload.finish_reason,
        tool_calls=payload.tool_calls,
        provider=payload.provider,
        validation_passed=payload.validation_passed,
        fallback_used=payload.fallback_used,
        fallback_reason=payload.fallback_reason,
        latency_ms=payload.latency_ms,
        failover_chain_id=payload.failover_chain_id,
        used_in_training=False,
        training_batch_id=None,
        content_hash=content_hash,
    )

    try:
        write_session.add(row)
        await write_session.commit()
        await write_session.refresh(row)
    except IntegrityError:
        await write_session.rollback()
        dup_result = await write_session.execute(dup_stmt)
        existing = dup_result.scalars().first()
        if existing is not None:
            logger.info(
                "sample_deduplicated_on_conflict",
                existing_id=str(existing.id),
                schema_id=payload.schema_id,
            )
            return _row_to_domain(existing)
        raise

    logger.info(
        "sample_created",
        sample_id=str(sample_id),
        schema_id=payload.schema_id,
        provider=payload.provider,
    )

    # Check training thresholds for new validated samples (DB queries inline, HTTP fire-and-forget).
    # Per-schema trigger and the system-wide single-LoRA trigger are independent;
    # both can fire on the same insert if the per-schema and aggregate counts both
    # land exactly on their thresholds.
    if payload.validation_passed is True:
        await _check_training_threshold(owner, payload.schema_id)
        await _check_global_training_threshold(owner)

    return _row_to_domain(row)


def _scope_reads_to_tenant(stmt, tenant_id: UUID | None):  # noqa: ANN001, ANN202
    """Narrow a read to the caller's own rows plus the fleet-global (NULL) bucket.

    This is the application-layer half of the tenant close. It renders the SAME
    predicate as the ``tenant_isolation_read`` RLS policy installed by migration 007
    (``tenant_id IS NULL OR tenant_id = current_setting('app.current_tenant')``), on
    purpose: reads that carry a tenant run on the restricted ``data_collector_app``
    role where that policy already binds, so the two agree and RLS remains the
    backstop if this predicate is ever dropped.

    NULL rows are deliberately included. 81,974 rows are fleet-global training data
    (96 more written in the 24h to 2026-09-07) and every caller has always been able
    to see them; excluding them here would silently delete 81,974 rows from every
    caller's view, which is a bigger break than the one being fixed.

    ⛔ Do not "simplify" this to ``tenant_id == tenant_id`` or to
    ``is_not_distinct_from``. Strict equality drops the global bucket;
    ``is_not_distinct_from(None)`` would, for an untenanted caller, match ONLY the
    global bucket and is the write-path's dedup predicate, not this one.
    """
    return stmt.where(
        or_(InferenceSampleDB.tenant_id.is_(None), InferenceSampleDB.tenant_id == tenant_id)
    )


async def list_samples(
    session: AsyncSession,
    filters: SampleFilter,
    *,
    tenant_id: UUID | None = None,
    all_tenants: bool = False,
) -> list[InferenceSample]:
    """Return a filtered, paginated list of inference samples.

    Scoped to ``tenant_id`` + the global bucket unless ``all_tenants`` is set. This
    endpoint returns full ``messages`` and ``response_content`` — the customer content
    itself — which is why its untenanted arm fails closed at the route.
    """
    stmt = select(InferenceSampleDB)
    if not all_tenants:
        stmt = _scope_reads_to_tenant(stmt, tenant_id)

    if filters.schema_id is not None:
        stmt = stmt.where(InferenceSampleDB.schema_id == filters.schema_id)
    if filters.job_id is not None:
        stmt = stmt.where(InferenceSampleDB.job_id == filters.job_id)
    if filters.inference_id is not None:
        stmt = stmt.where(InferenceSampleDB.inference_id == filters.inference_id)
    if filters.failover_chain_id is not None:
        stmt = stmt.where(InferenceSampleDB.failover_chain_id == filters.failover_chain_id)
    if filters.provider is not None:
        stmt = stmt.where(InferenceSampleDB.provider == filters.provider)
    if filters.validation_passed is not None:
        stmt = stmt.where(InferenceSampleDB.validation_passed == filters.validation_passed)
    if filters.used_in_training is not None:
        stmt = stmt.where(InferenceSampleDB.used_in_training == filters.used_in_training)
    if filters.start_date is not None:
        stmt = stmt.where(InferenceSampleDB.created_at >= filters.start_date)
    if filters.end_date is not None:
        stmt = stmt.where(InferenceSampleDB.created_at <= filters.end_date)

    stmt = stmt.order_by(InferenceSampleDB.created_at.desc())
    stmt = stmt.offset(filters.offset).limit(filters.limit)

    result = await session.execute(stmt)
    rows = result.scalars().all()
    return [_row_to_domain(r) for r in rows]


async def get_count(
    session: AsyncSession,
    schema_id: str | None = None,
    only_validated: bool = False,
    only_unused: bool = False,
    *,
    tenant_id: UUID | None = None,
    all_tenants: bool = False,
) -> SampleCountResponse:
    """Return the count of samples matching the given filters.

    When ``schema_id`` is omitted the count aggregates across all schemas —
    used by the flywheel CronJob to decide whether to trigger a global retrain.

    ⚠️ This is the HTTP count only. The flywheel's own rising-edge threshold counts in
    ``_check_training_threshold`` / ``_check_global_training_threshold`` are separate
    call sites that keep running on the owner session with no predicate, because they
    must stay cross-tenant — scoping them would silently under-count and the trigger
    would stop firing (``tests/test_rls_enforce.py::TestFlywheelCountNeedsOwner``).
    """
    stmt = select(func.count()).select_from(InferenceSampleDB)
    if not all_tenants:
        stmt = _scope_reads_to_tenant(stmt, tenant_id)
    if schema_id is not None:
        stmt = stmt.where(InferenceSampleDB.schema_id == schema_id)
    if only_validated:
        stmt = stmt.where(InferenceSampleDB.validation_passed.is_(True))
    if only_unused:
        stmt = stmt.where(InferenceSampleDB.used_in_training.is_(False))

    result = await session.execute(stmt)
    count = result.scalar_one()

    return SampleCountResponse(count=count, schema_id=schema_id)


async def list_schema_ids(
    session: AsyncSession,
    *,
    tenant_id: UUID | None = None,
    all_tenants: bool = False,
) -> list[str]:
    """Return a deduplicated list of all schema_ids visible to the caller.

    A schema id is itself cross-tenant information: measured 2026-09-07, 47 of 236
    schema ids carry rows from more than one tenant, so an unscoped list told every
    caller which schemas every other tenant runs.
    """
    stmt = select(InferenceSampleDB.schema_id).distinct()
    if not all_tenants:
        stmt = _scope_reads_to_tenant(stmt, tenant_id)
    stmt = stmt.order_by(InferenceSampleDB.schema_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def export_samples_jsonl(
    session: AsyncSession,
    params: ExportParams,
    *,
    tenant_id: UUID | None = None,
    all_tenants: bool = False,
) -> AsyncGenerator[str]:
    """Stream inference samples as newline-delimited JSON.

    Each line is a complete JSON object suitable for training pipelines.
    Uses server-side cursors via ``stream_scalars`` to keep memory bounded.
    """
    stmt = select(InferenceSampleDB).where(InferenceSampleDB.schema_id == params.schema_id)
    if not all_tenants:
        stmt = _scope_reads_to_tenant(stmt, tenant_id)

    if params.only_validated:
        stmt = stmt.where(InferenceSampleDB.validation_passed.is_(True))
    if params.only_unused:
        stmt = stmt.where(InferenceSampleDB.used_in_training.is_(False))
    if params.provider is not None:
        stmt = stmt.where(InferenceSampleDB.provider == params.provider)
    if params.fallback_used is not None:
        stmt = stmt.where(InferenceSampleDB.fallback_used == params.fallback_used)

    stmt = stmt.order_by(InferenceSampleDB.created_at.asc())

    if params.limit is not None:
        stmt = stmt.limit(params.limit)

    result = await session.stream_scalars(stmt)
    async for row in result:
        domain = _row_to_domain(row)
        yield domain.model_dump_json() + "\n"


async def get_stats(
    session: AsyncSession,
    schema_id: str | None = None,
    *,
    tenant_id: UUID | None = None,
    all_tenants: bool = False,
) -> list[SampleStats]:
    """Return aggregation statistics grouped by schema_id.

    Both the outer aggregate and the per-schema provider breakdown are scoped — the
    breakdown is a second query and would otherwise leak the very counts the outer
    one just hid.
    """
    stmt = select(
        InferenceSampleDB.schema_id,
        func.count().label("total_samples"),
        func.sum(case((InferenceSampleDB.validation_passed.is_(True), 1), else_=0)).label(
            "validated_samples"
        ),
        func.sum(case((InferenceSampleDB.validation_passed.is_(False), 1), else_=0)).label(
            "failed_samples"
        ),
        func.sum(case((InferenceSampleDB.used_in_training.is_(True), 1), else_=0)).label(
            "used_in_training"
        ),
        func.sum(
            case(
                (
                    (
                        InferenceSampleDB.validation_passed.is_(True)
                        & InferenceSampleDB.used_in_training.is_(False)
                    ),
                    1,
                ),
                else_=0,
            )
        ).label("available_for_training"),
        func.avg(InferenceSampleDB.latency_ms).label("avg_latency_ms"),
        func.avg(case((InferenceSampleDB.fallback_used.is_(True), 1.0), else_=0.0)).label(
            "fallback_rate"
        ),
    ).group_by(InferenceSampleDB.schema_id)

    if not all_tenants:
        stmt = _scope_reads_to_tenant(stmt, tenant_id)
    if schema_id is not None:
        stmt = stmt.where(InferenceSampleDB.schema_id == schema_id)

    result = await session.execute(stmt)
    rows = result.all()

    stats_list: list[SampleStats] = []
    for row in rows:
        # Fetch provider breakdown for this schema_id
        provider_stmt = (
            select(
                InferenceSampleDB.provider,
                func.count().label("cnt"),
            )
            .where(InferenceSampleDB.schema_id == row.schema_id)
            .group_by(InferenceSampleDB.provider)
        )
        if not all_tenants:
            provider_stmt = _scope_reads_to_tenant(provider_stmt, tenant_id)
        provider_result = await session.execute(provider_stmt)
        provider_breakdown = {p.provider: p.cnt for p in provider_result.all()}

        stats_list.append(
            SampleStats(
                schema_id=row.schema_id,
                total_samples=int(row.total_samples),
                validated_samples=int(row.validated_samples),
                failed_samples=int(row.failed_samples),
                used_in_training=int(row.used_in_training),
                available_for_training=int(row.available_for_training),
                provider_breakdown=provider_breakdown,
                avg_latency_ms=float(row.avg_latency_ms or 0),
                fallback_rate=float(row.fallback_rate or 0),
            )
        )

    return stats_list


async def mark_used(
    session: AsyncSession,
    request: MarkUsedRequest,
) -> int:
    """Mark a batch of samples as used in training.

    Returns the number of rows actually updated.
    """
    # Normalise incoming ids to str; psycopg casts text to the uuid column.
    str_ids = [str(sid) for sid in request.sample_ids]
    stmt = (
        update(InferenceSampleDB)
        .where(InferenceSampleDB.id.in_(str_ids))
        .values(
            used_in_training=True,
            training_batch_id=request.training_batch_id,
        )
    )
    result = cast(CursorResult[tuple[()]], await session.execute(stmt))
    await session.commit()

    rowcount = result.rowcount
    updated = rowcount if rowcount is not None else 0
    logger.info(
        "samples_marked_used",
        count=updated,
        training_batch_id=request.training_batch_id,
    )
    return updated


async def purge_samples(
    session: AsyncSession,
    *,
    tenant_id: UUID | None = None,
    all_tenants: bool = False,
    before_date: datetime | None = None,
    schema_id: str | None = None,
    provider: str | None = None,
    validation_passed: bool | None = None,
) -> PurgeResponse:
    """Delete samples matching the given filters. At least one filter is required.

    TENANT SCOPE (the caller must pick exactly one, and the route enforces it):

      * ``tenant_id`` set — delete ONLY that tenant's rows. The NULL/global
        bucket is deliberately NOT included: a tenant-scoped purge that also
        swept the fleet-global rows would still be destroying data it does not
        own, which is the defect this exists to close.
      * ``all_tenants=True`` — the explicit, named cross-tenant admin path. It
        is never a default; it must be asked for by name.

    This runs on the OWNER (RLS-bypass) session, so the ``_GLOBAL_DEL`` policy
    from migration 007 cannot constrain it — RLS is ENABLE-not-FORCE and the
    owner is the table owner. The predicate below is therefore the ONLY thing
    separating one tenant's rows from another's on this path.
    """
    stmt = delete(InferenceSampleDB)
    filters_applied: dict[str, str] = {}

    if all_tenants:
        filters_applied["tenant_scope"] = "all_tenants"
    else:
        # Strict equality, not is_not_distinct_from: NULL/global rows belong to
        # the fleet, not to this tenant.
        stmt = stmt.where(InferenceSampleDB.tenant_id == tenant_id)
        filters_applied["tenant_scope"] = str(tenant_id)

    if before_date is not None:
        stmt = stmt.where(InferenceSampleDB.created_at < before_date)
        filters_applied["before_date"] = before_date.isoformat()
    if schema_id is not None:
        stmt = stmt.where(InferenceSampleDB.schema_id == schema_id)
        filters_applied["schema_id"] = schema_id
    if provider is not None:
        stmt = stmt.where(InferenceSampleDB.provider == provider)
        filters_applied["provider"] = provider
    if validation_passed is not None:
        stmt = stmt.where(InferenceSampleDB.validation_passed == validation_passed)
        filters_applied["validation_passed"] = str(validation_passed)

    result = cast(CursorResult[tuple[()]], await session.execute(stmt))
    await session.commit()

    purged = result.rowcount if result.rowcount is not None else 0
    # WARNING, not INFO: this is an irreversible delete on an RLS-bypassing
    # session and there is no audit table in this service, so this line is the
    # only record that it happened. The cross-tenant arm is louder still.
    if all_tenants:
        logger.warning(
            "samples_purged_all_tenants",
            count=purged,
            filters=filters_applied,
            tenant_scope="all_tenants",
        )
    else:
        logger.warning(
            "samples_purged",
            count=purged,
            filters=filters_applied,
            tenant_scope=str(tenant_id),
        )

    return PurgeResponse(purged_count=purged, filters_applied=filters_applied)


def _row_to_domain(row: InferenceSampleDB) -> InferenceSample:
    """Convert an ORM row to a Pydantic domain model."""
    return InferenceSample(
        id=UUID(str(row.id)),
        tenant_id=UUID(str(row.tenant_id)) if row.tenant_id is not None else None,
        schema_id=row.schema_id,
        job_id=row.job_id,
        inference_id=row.inference_id,
        messages=row.messages,
        response_content=row.response_content,
        outcome=row.outcome,
        finish_reason=row.finish_reason,
        tool_calls=row.tool_calls,
        provider=row.provider,
        validation_passed=row.validation_passed,
        fallback_used=row.fallback_used,
        fallback_reason=row.fallback_reason,
        latency_ms=row.latency_ms,
        failover_chain_id=row.failover_chain_id,
        used_in_training=row.used_in_training,
        training_batch_id=row.training_batch_id,
        created_at=row.created_at,
    )
