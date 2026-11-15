"""FastAPI application with async lifespan for the data-collector service."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware import Middleware

import tenant_binding
from db import dispose_engine, get_engine_for_check, init_db, tenant_ctx
from logging_config import configure_logging
from models.domain import HealthResponse, LivenessResponse, ReadinessResponse
from routers.samples import router as samples_router
from routers.schemas import router as schemas_router
from settings import get_settings
from telemetry import setup_telemetry

logger = structlog.stdlib.get_logger(__name__)

_cors_allow_origins = get_settings().cors_allow_origins_list


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Manage startup and shutdown resources."""
    settings = get_settings()

    # Logging
    configure_logging(settings)
    logger.info("starting", service=settings.service_name, port=settings.port)

    # Telemetry (graceful degradation)
    setup_telemetry(settings)

    # Instrument FastAPI with OpenTelemetry (best-effort)
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        logger.warning("fastapi_otel_instrumentation_unavailable", exc_info=True)
    else:
        try:
            FastAPIInstrumentor.instrument_app(application)
        except ImportError, RuntimeError:
            logger.warning("fastapi_otel_instrumentation_failed", exc_info=True)

    # Database
    await init_db(settings)
    logger.info("database_initialised", url=settings.database_url.split("@")[-1])

    yield

    # Shutdown
    logger.info("shutting_down", service=settings.service_name)
    await dispose_engine()


app = FastAPI(
    title="Data Collector",
    description="Receives and persists every inference interaction for training data generation",
    version="0.1.0",
    lifespan=lifespan,
    middleware=[
        Middleware(
            cast(Any, CORSMiddleware),
            allow_origins=_cors_allow_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ],
)

app.include_router(samples_router)
app.include_router(schemas_router)

# Paths excluded from the tenant-binding tally: probes and service metadata. They are
# unauthenticated by design (CLAUDE.md lists them as the license-authz exemptions), carry
# no tenant, and — for the two kubelet probes — fire every few seconds. Counting them
# would drown the population a fail-closed decision has to be sized against.
_UNCOUNTED_PATHS = frozenset({"/", "/health", "/health/live", "/health/ready"})


@app.middleware("http")
async def bind_tenant_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Set the request-scoped tenant from the X-Tenant-Id header.

    Read at transaction-begin by the RLS binding (db.register_tenant_guc).

    ⚠️ **This binding is NOT authenticated, and TENANT_RLS_MODE=enforce is LIVE.**
    The docstring here used to claim two defences and both are false:

    * *"Inert until TENANT_RLS_MODE=enforce"* — enforce has been live on the
      production Deployment since the Stage C cutover; verified on the running pod
      2026-09-07.
    * *"License-authz resolves the license token at the mesh edge"* — that ext_authz
      filter is attached with ``context: GATEWAY``, so it only runs on traffic that
      enters through the ingress gateway. Pod-to-pod traffic never traverses it, and
      that is >99.99% of this service's connections.

    So any workload that can open a connection to :8151 chooses its own tenant. This
    namespace has no waypoint (L4-only ambient), so no peer identity reaches the
    process and it cannot be rejected here — see ``tenant_binding`` for the full
    measurement and for why the fix is mesh-side. Every branch is counted so the
    exposure is observable and a future fail-closed has a denominator.

    ⚠️ Probe paths are excluded from the tally on purpose. kubelet's liveness and
    readiness probes carry no ``X-Tenant-Id``, run every few seconds, and are not API
    callers — counting them would swamp the ``absent`` branch with traffic that no
    fail-closed decision would ever apply to, and the denominator is the whole point
    of the tally. The binding itself still happens for those paths (they simply never
    touch the DB with a tenant).
    """
    raw = request.headers.get("X-Tenant-Id")
    authorization = request.headers.get("Authorization")
    outcome, tenant = tenant_binding.classify(raw, authorization)
    if request.url.path not in _UNCOUNTED_PATHS:
        tenant_binding.record(outcome, tenant, authorization)
    token = tenant_ctx.set(tenant)
    try:
        return await call_next(request)
    finally:
        tenant_ctx.reset(token)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a consistent JSON error for unhandled exceptions."""
    if isinstance(exc, HTTPException):
        raise exc
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {"message": "Internal server error", "type": "server_error", "code": 500}
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Liveness probe -- always returns OK if the process is up."""
    return HealthResponse(status="ok", service="data-collector", version="0.1.0")


@app.get("/health/live", response_model=LivenessResponse, tags=["health"], summary="Liveness probe")
async def liveness() -> LivenessResponse:
    """Lightweight liveness probe -- returns alive if the process is up."""
    return LivenessResponse()


@app.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
async def readiness() -> ReadinessResponse:
    """Readiness probe -- checks that the database connection is alive."""
    db_ok = await get_engine_for_check()
    return ReadinessResponse(
        status="ready" if db_ok else "not_ready",
        database="connected" if db_ok else "disconnected",
    )


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Root endpoint with service info."""
    return {
        "service": "data-collector",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
