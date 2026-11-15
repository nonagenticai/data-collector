"""Schema discovery endpoint for listing available schema identifiers."""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_owner_session, get_session
from models.domain import SchemaListResponse
from read_scope import ReadScope, UntenantedReads, resolve_read_scope
from routers.samples import resolve_tenant
from services import sample_service

logger = structlog.stdlib.get_logger(__name__)

router = APIRouter(prefix="/api/schemas", tags=["schemas"])


@router.get(
    "",
    response_model=SchemaListResponse,
    summary="List available schemas",
    description=(
        "Return a deduplicated list of schema identifiers that have at least one "
        "sample visible to the caller — the caller's own rows plus the fleet-global "
        "(NULL-tenant) bucket. Pass all_tenants=true for the cross-tenant platform "
        "view; that cannot be combined with an X-Tenant-Id caller."
    ),
)
async def list_schemas(
    all_tenants: bool = Query(
        default=False,
        description=(
            "List schema ids across EVERY tenant. Platform/flywheel path; never a "
            "default and not combinable with an X-Tenant-Id caller."
        ),
    ),
    session: AsyncSession = Depends(get_session),
    owner_session: AsyncSession = Depends(get_owner_session),
    tenant: UUID | None = Depends(resolve_tenant),
) -> SchemaListResponse:
    """Return the schema identifiers visible to the caller.

    A schema id is itself cross-tenant information — measured on prod 2026-09-07, 47
    of 236 schema ids carry rows from more than one tenant — so this list leaked which
    schemas every other tenant runs.

    ⚠️ Untenanted callers still get the cross-tenant list. This endpoint backs
    ``training-data-generator``'s trigger loop (``trigger_service.py:58``) **and its
    readiness probe** (``main.py:190``), and tdg sends no tenant header, so failing it
    closed would take tdg NotReady rather than close a hole. The branch is counted and
    warned instead — see ``read_scope``.
    """
    scope = resolve_read_scope(
        tenant=tenant,
        all_tenants=all_tenants,
        endpoint="GET /api/schemas",
        untenanted=UntenantedReads.CROSS_TENANT,
    )
    schema_ids = await sample_service.list_schema_ids(
        owner_session if scope is ReadScope.ALL_TENANTS else session,
        tenant_id=tenant,
        all_tenants=scope is ReadScope.ALL_TENANTS,
    )
    return SchemaListResponse(schema_ids=schema_ids)
