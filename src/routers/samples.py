"""Sample endpoints for ingesting, querying, exporting, and managing inference samples."""

from datetime import UTC, datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_owner_session, get_session, tenant_ctx
from models.domain import (
    ExportParams,
    InferenceSample,
    MarkUsedRequest,
    MarkUsedResponse,
    PurgeResponse,
    SampleCountResponse,
    SampleCreate,
    SampleFilter,
    SampleStats,
)
from read_scope import ReadScope, UntenantedReads, resolve_read_scope
from services import sample_service
from settings import get_settings

logger = structlog.stdlib.get_logger(__name__)

router = APIRouter(prefix="/api/samples", tags=["samples"])

_ALL_TENANTS_QUERY = Query(
    default=False,
    description=(
        "Read across EVERY tenant instead of just the caller's rows plus the "
        "fleet-global (NULL-tenant) bucket. This is the platform/flywheel path; it is "
        "never a default and cannot be combined with an X-Tenant-Id caller."
    ),
)


def _pick_session(scope: ReadScope, runtime: AsyncSession, owner: AsyncSession) -> AsyncSession:
    """Return the connection the resolved scope must run on.

    ``TENANT`` goes to the restricted ``data_collector_app`` role so the
    ``tenant_isolation_read`` policy binds underneath the explicit predicate;
    ``ALL_TENANTS`` goes to the owner, which bypasses RLS by design (ENABLE-not-FORCE)
    and is the only connection that can legitimately see every tenant.

    Both sessions are injected on every read handler. That costs nothing when one is
    unused: an ``AsyncSession`` does not acquire a connection until a statement runs,
    which is the same property ``ingest_sample`` already relies on.
    """
    return owner if scope is ReadScope.ALL_TENANTS else runtime


def _coerce_utc(dt: datetime | None) -> datetime | None:
    """Treat a naive datetime bound as UTC for comparison against timestamptz."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _parse_tenant(value: str | None) -> UUID | None:
    """Parse a tenant header value, treating anything unparseable as absent."""
    if not value:
        return None
    raw = value.removeprefix("Bearer ").strip()
    try:
        return UUID(raw)
    except ValueError:
        return None


async def resolve_tenant(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_effective_tenant_id: str | None = Header(default=None, alias="X-Effective-Tenant-Id"),
) -> UUID | None:
    """Resolve the owning tenant, and keep the RLS GUC in agreement with it.

    ⚠️ **This dependency must stay ``async``.** FastAPI runs a *sync* dependency in a
    threadpool via ``run_in_threadpool``, which copies the context into the worker
    thread — a ``ContextVar.set`` there mutates only that copy and is invisible to the
    endpoint and to the DB session that reads it. A unit test calling this function
    directly cannot see that difference; ``tests/test_tenant_binding.py``'s
    ``TestGucVisibleToTheWrite`` captures the value at ``create_sample`` and is the
    test that actually holds this property.

    ⚠️ The ``bind_tenant_context`` middleware sets ``app.current_tenant`` from the raw
    ``X-Tenant-Id`` alone — it knows nothing about ``X-Effective-Tenant-Id`` or the
    delegation allowlist. So on a **delegated** write the two disagreed: the row was
    stamped with the *effective* tenant while the GUC still held the *caller's*, and
    under ``TENANT_RLS_MODE=enforce`` (live in production) the tenant-attributed INSERT
    runs on the restricted role, where ``tenant_isolation_ins``'s ``WITH CHECK
    (tenant_id = current_setting('app.current_tenant'))`` rejects exactly that row.
    Every delegated write would have failed with a 500.

    It has never fired because ``DATA_COLLECTOR_DELEGATING_TENANTS`` is unset on the
    production Deployment (verified on the running pod 2026-09-07), so delegation is a
    no-op fleet-wide and the break-set for this correction is empty. Re-binding the
    context to the *resolved* tenant here — dependencies resolve before the endpoint
    opens its transaction, so the GUC is set before ``register_tenant_guc`` reads it —
    makes the GUC and the row agree on every path, which is also a precondition for
    ever failing this binding closed.
    """
    tenant = _resolve_tenant(x_tenant_id, x_effective_tenant_id)
    tenant_ctx.set(str(tenant) if tenant is not None else None)
    return tenant


def _resolve_tenant(
    x_tenant_id: str | None,
    x_effective_tenant_id: str | None,
) -> UUID | None:
    """Resolve the owning tenant from the ``X-Tenant-Id`` header.

    License-authz introspects the license token at the mesh edge and injects the
    resolved tenant as a **bare** ``X-Tenant-Id`` (see inference-router
    ``src/security.py``), so this service does not introspect Bearer tokens. A
    value that does not parse as a UUID (e.g. an unresolved token) is treated as
    absent → global sample; while the gate is dormant this never fails a write.

    **Delegation.** A background run reaches this service through a service
    account, so the mesh resolves ``X-Tenant-Id`` to the *service account's*
    tenant and the customer's real tenant is lost — every such sample lands
    under the SA. A caller whose own tenant is listed in
    ``DATA_COLLECTOR_DELEGATING_TENANTS`` may therefore carry the run's real
    tenant in ``X-Effective-Tenant-Id``, mirroring ``_apply_effective_tenant``
    in ``w2v_common/auth/middleware.py``.

    The gate is keyed on ``X-Tenant-Id`` because that is the only value here the
    caller cannot choose — the mesh overwrites it. Honouring the effective
    header on any other basis would let any client write into any tenant, which
    is strictly worse than the mis-attribution it fixes. A caller that sends no
    ``X-Tenant-Id`` has no established identity and can never delegate, and an
    empty allowlist makes the whole path a no-op.
    """
    caller_tenant = _parse_tenant(x_tenant_id)
    effective_tenant = _parse_tenant(x_effective_tenant_id)

    if caller_tenant is None:
        if effective_tenant is not None:
            logger.warning(
                "effective_tenant_rejected_no_caller_tenant",
                effective_tenant=str(effective_tenant),
            )
        if x_tenant_id:
            logger.debug("x_tenant_id_not_a_uuid_treating_as_global")
        return None

    if effective_tenant is None or effective_tenant == caller_tenant:
        return caller_tenant

    if str(caller_tenant) not in get_settings().delegating_tenants_set:
        # Either a spoof attempt or a receiver missing its allowlist config.
        # Surface it rather than silently dropping the header — and keep the
        # caller's own tenant, never the asserted one.
        logger.warning(
            "effective_tenant_rejected_caller_not_delegating",
            caller_tenant=str(caller_tenant),
            effective_tenant=str(effective_tenant),
        )
        return caller_tenant

    logger.info(
        "effective_tenant_delegation",
        caller_tenant=str(caller_tenant),
        effective_tenant=str(effective_tenant),
    )
    return effective_tenant


@router.post(
    "",
    response_model=InferenceSample,
    status_code=201,
    summary="Record inference sample",
    description="Persist a single inference interaction (request + response pair) from the inference pipeline for future training data generation.",
)
async def ingest_sample(
    payload: SampleCreate,
    session: AsyncSession = Depends(get_session),
    owner_session: AsyncSession = Depends(get_owner_session),
    header_tenant: UUID | None = Depends(resolve_tenant),
) -> InferenceSample:
    """Ingest a request+response pair from the inference pipeline.

    Tenant precedence: ``X-Effective-Tenant-Id`` (only from a caller listed in
    ``DATA_COLLECTOR_DELEGATING_TENANTS``) → gateway-injected ``X-Tenant-Id``
    header → body ``tenant_id`` → None (fleet-global). The headers win so an
    untrusted body cannot override the mesh-resolved tenant.

    Both sessions are injected: the runtime (tenant-bound) session for
    tenant-attributed writes and the owner (RLS-bypass) session for global writes
    + the cross-tenant flywheel threshold counts. create_sample routes between
    them; while dormant they are the same engine.
    """
    tenant_id = header_tenant if header_tenant is not None else payload.tenant_id
    return await sample_service.create_sample(
        session, payload, tenant_id, owner_session=owner_session
    )


@router.get(
    "",
    response_model=list[InferenceSample],
    summary="List inference samples",
    description="Retrieve samples with optional filtering by schema, provider, validation status, training usage, and date range. Supports pagination via limit and offset.",
)
async def list_samples(
    schema_id: str | None = Query(default=None),
    job_id: str | None = Query(default=None),
    inference_id: str | None = Query(default=None),
    failover_chain_id: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    validation_passed: bool | None = Query(default=None),
    used_in_training: bool | None = Query(default=None),
    start_date: datetime | None = Query(default=None),
    end_date: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    all_tenants: bool = _ALL_TENANTS_QUERY,
    session: AsyncSession = Depends(get_session),
    owner_session: AsyncSession = Depends(get_owner_session),
    tenant: UUID | None = Depends(resolve_tenant),
) -> list[InferenceSample]:
    """List samples visible to the caller, with optional filtering and pagination.

    This handler returns full ``messages`` + ``response_content`` — the customer
    content itself — and was the widest read on the service. Its untenanted arm
    therefore fails CLOSED: the only source-level consumers are ``model-evaluator``
    (a client, a manifest and its own AuthorizationPolicy, but zero traffic in 28
    days), ``fr``'s super-admin lens (6 connections in 28 days, and it already
    forwards X-Tenant-Id), and the CI harness. None of them is the flywheel.
    """
    scope = resolve_read_scope(
        tenant=tenant,
        all_tenants=all_tenants,
        endpoint="GET /api/samples",
        untenanted=UntenantedReads.REJECT,
    )
    filters = SampleFilter(
        schema_id=schema_id,
        job_id=job_id,
        inference_id=inference_id,
        failover_chain_id=failover_chain_id,
        provider=provider,
        validation_passed=validation_passed,
        used_in_training=used_in_training,
        start_date=_coerce_utc(start_date),
        end_date=_coerce_utc(end_date),
        limit=limit,
        offset=offset,
    )
    return await sample_service.list_samples(
        _pick_session(scope, session, owner_session),
        filters,
        tenant_id=tenant,
        all_tenants=scope is ReadScope.ALL_TENANTS,
    )


@router.get(
    "/count",
    response_model=SampleCountResponse,
    summary="Count samples",
    description="Return the number of samples, optionally filtered to a single schema and/or to validated/unused samples. With no schema_id the count aggregates across all schemas (flywheel global-retrain trigger).",
)
async def get_sample_count(
    schema_id: str | None = Query(default=None),
    only_validated: bool = Query(default=False),
    only_unused: bool = Query(default=False),
    all_tenants: bool = _ALL_TENANTS_QUERY,
    session: AsyncSession = Depends(get_session),
    owner_session: AsyncSession = Depends(get_owner_session),
    tenant: UUID | None = Depends(resolve_tenant),
) -> SampleCountResponse:
    """Return the count of samples visible to the caller.

    ⚠️ Untenanted callers still get the cross-tenant count.
    ``training-data-generator`` polls this per schema to decide whether to fire a
    training batch (``trigger_service.py:244``) and sends no tenant header on any
    call, so rejecting an untenanted count is a flywheel outage, not a fix. The
    branch is counted and warned instead — see ``read_scope``.
    """
    scope = resolve_read_scope(
        tenant=tenant,
        all_tenants=all_tenants,
        endpoint="GET /api/samples/count",
        untenanted=UntenantedReads.CROSS_TENANT,
    )
    return await sample_service.get_count(
        _pick_session(scope, session, owner_session),
        schema_id=schema_id,
        only_validated=only_validated,
        only_unused=only_unused,
        tenant_id=tenant,
        all_tenants=scope is ReadScope.ALL_TENANTS,
    )


@router.get(
    "/export",
    summary="Export samples as JSONL",
    description="Stream samples as newline-delimited JSON for training pipelines. The response is streamed so arbitrarily large datasets can be exported without loading everything into memory.",
)
async def export_samples(
    schema_id: str = Query(...),
    only_validated: bool = Query(default=True),
    only_unused: bool = Query(default=True),
    provider: str | None = Query(default=None),
    fallback_used: bool | None = Query(default=None),
    limit: int | None = Query(default=None),
    all_tenants: bool = _ALL_TENANTS_QUERY,
    session: AsyncSession = Depends(get_session),
    owner_session: AsyncSession = Depends(get_owner_session),
    tenant: UUID | None = Depends(resolve_tenant),
) -> StreamingResponse:
    """Export samples as newline-delimited JSON (JSONL) for training pipelines.

    Returns a streaming response so that arbitrarily large datasets can be
    exported without loading everything into memory.

    ⚠️ Untenanted callers still get the cross-tenant export, for the same reason as
    ``/count``: this is ``training-data-generator``'s batch-build path
    (``batch_service.py:183,359,536``) and it sends no tenant header.
    """
    scope = resolve_read_scope(
        tenant=tenant,
        all_tenants=all_tenants,
        endpoint="GET /api/samples/export",
        untenanted=UntenantedReads.CROSS_TENANT,
    )
    params = ExportParams(
        schema_id=schema_id,
        only_validated=only_validated,
        only_unused=only_unused,
        provider=provider,
        fallback_used=fallback_used,
        limit=limit,
    )
    return StreamingResponse(
        sample_service.export_samples_jsonl(
            _pick_session(scope, session, owner_session),
            params,
            tenant_id=tenant,
            all_tenants=scope is ReadScope.ALL_TENANTS,
        ),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{schema_id}_samples.jsonl"',
        },
    )


@router.get(
    "/stats",
    response_model=list[SampleStats],
    summary="Get sample statistics",
    description="Return aggregation statistics (total, validated, used counts) grouped by schema_id. Optionally filter to a single schema.",
)
async def get_stats(
    schema_id: str | None = Query(default=None),
    all_tenants: bool = _ALL_TENANTS_QUERY,
    session: AsyncSession = Depends(get_session),
    owner_session: AsyncSession = Depends(get_owner_session),
    tenant: UUID | None = Depends(resolve_tenant),
) -> list[SampleStats]:
    """Return aggregation statistics for the schemas visible to the caller.

    Fails closed when untenanted: the only source-level consumer is ``fr``'s
    ``sampleStats`` binding, which has no in-repo caller at all, so there is no live
    untenanted traffic to preserve.
    """
    scope = resolve_read_scope(
        tenant=tenant,
        all_tenants=all_tenants,
        endpoint="GET /api/samples/stats",
        untenanted=UntenantedReads.REJECT,
    )
    return await sample_service.get_stats(
        _pick_session(scope, session, owner_session),
        schema_id,
        tenant_id=tenant,
        all_tenants=scope is ReadScope.ALL_TENANTS,
    )


@router.patch(
    "/mark-used",
    response_model=MarkUsedResponse,
    summary="Mark samples as used",
    description="Flag a batch of sample IDs as consumed by a specific training batch, preventing them from being re-exported for future training runs.",
)
async def mark_samples_used(
    request: MarkUsedRequest,
    session: AsyncSession = Depends(get_owner_session),
) -> MarkUsedResponse:
    """Mark a batch of samples as used in a specific training batch."""
    updated = await sample_service.mark_used(session, request)
    return MarkUsedResponse(
        updated_count=updated,
        training_batch_id=request.training_batch_id,
    )


@router.delete(
    "/purge",
    response_model=PurgeResponse,
    summary="Purge samples",
    description=(
        "Permanently delete samples matching the provided filters. Scoped to the "
        "caller's tenant (gateway-injected X-Tenant-Id). At least one filter "
        "(before_date, schema_id, provider, or validation_passed) must be specified "
        "to prevent accidental full-table deletes. Deleting across every tenant is a "
        "separate, explicitly named operation: pass all_tenants=true, which a caller "
        "carrying an X-Tenant-Id may not do."
    ),
)
async def purge_samples(
    before_date: datetime | None = Query(default=None),
    schema_id: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    validation_passed: bool | None = Query(default=None),
    all_tenants: bool = Query(
        default=False,
        description=(
            "Explicitly purge across EVERY tenant, including the fleet-global "
            "(NULL-tenant) rows. Reserved for platform maintenance; requires a "
            "caller with no X-Tenant-Id. Never a default."
        ),
    ),
    session: AsyncSession = Depends(get_owner_session),
    tenant: UUID | None = Depends(resolve_tenant),
) -> PurgeResponse:
    """Purge samples matching filters, within the caller's tenant.

    This handler runs on ``get_owner_session`` — the RLS-BYPASS session — which
    is correct for the flywheel's cross-tenant *reads* but was silently fatal
    here: a ``DELETE`` with no tenant predicate on a bypassing session spans
    every tenant. ``TENANT_RLS_MODE=enforce`` is live in production, so the
    ``_GLOBAL_DEL`` policy exists, but RLS is ENABLE-not-FORCE and this role
    owns the table, so the policy never applied to this statement. The predicate
    added in ``sample_service.purge_samples`` is the only control on this path.

    The pre-existing empty-filter guard below is retained but is NOT a tenancy
    control — it only blocks the fully-unfiltered full-table wipe, and a single
    ``schema_id`` (or a far-future ``before_date``) satisfied it while still
    deleting every tenant's rows.
    """
    from fastapi import HTTPException

    if all(v is None for v in (before_date, schema_id, provider, validation_passed)):
        raise HTTPException(status_code=400, detail="At least one filter is required")

    # Force ONE unambiguous scope. A caller that presents a tenant AND asks for
    # all_tenants is contradicting itself; fail loud rather than silently
    # picking the wider (destructive) reading.
    if all_tenants and tenant is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                "all_tenants=true is a platform-maintenance path and cannot be combined "
                "with a tenant-scoped caller (X-Tenant-Id present). Drop one."
            ),
        )
    if not all_tenants and tenant is None:
        # Fail CLOSED. Before this guard an untenanted caller purged every
        # tenant's rows by default; the widest possible blast radius must be
        # asked for by name, never inherited from a missing header.
        logger.warning("purge_refused_no_tenant_scope", all_tenants=all_tenants)
        raise HTTPException(
            status_code=400,
            detail=(
                "No tenant scope: this request carries no X-Tenant-Id. Purge is "
                "scoped to the caller's tenant. To purge across every tenant, ask "
                "for it by name with all_tenants=true."
            ),
        )

    return await sample_service.purge_samples(
        session,
        tenant_id=tenant,
        all_tenants=all_tenants,
        before_date=before_date,
        schema_id=schema_id,
        provider=provider,
        validation_passed=validation_passed,
    )
