"""Pydantic v2 domain models for the data-collector service."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid7

from pydantic import BaseModel, Field


class InferenceSample(BaseModel):
    """Full representation of a persisted inference sample."""

    id: UUID = Field(default_factory=uuid7)
    tenant_id: UUID | None = Field(
        default=None,
        description=(
            "Owning tenant. None = fleet-global sample (shared cross-tenant, the "
            "default while the gate is dormant). Resolved from the gateway-injected "
            "X-Tenant-Id header on write."
        ),
    )
    schema_id: str = Field(..., description="Schema identifier, e.g. ks_EntitySpotterAgent")
    job_id: str = Field(..., description="Job identifier from X-Job-Id header")
    inference_id: str = Field(..., description="Unique inference request identifier")
    messages: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Input messages array. Values may be None (e.g. reasoning_content "
            "on non-reasoning replies) or list-of-dicts (multimodal content "
            "parts), so this is dict[str, Any] not dict[str, str]."
        ),
    )
    response_content: str = Field(
        ...,
        description=(
            "Raw response content. May legitimately be '' -- see ``outcome`` / "
            "``tool_calls`` for what actually happened."
        ),
    )
    outcome: str | None = Field(
        default=None,
        description=(
            "inference-router's exhaustive outcome partition: ok | empty | "
            "empty_result | error | invalid | aborted. A plain string, NOT an "
            "enum -- the router owns the vocabulary and a new label must never "
            "need a migration. None = written by a pre-R22 inference-router."
        ),
    )
    finish_reason: str | None = Field(
        default=None,
        description=(
            "Upstream's own finish_reason (stop | length | tool_calls | ...). "
            "Turns an 'empty' outcome into a diagnosis: finish_reason='length' "
            "is a burned budget, 'tool_calls' is a healthy blank-content reply."
        ),
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Upstream tool-call payload (OpenAI shape: a list of dicts with id, "
            "type and function.name/function.arguments, arguments being a JSON "
            "string). For tool-calling traffic this is the ANSWER -- the row's "
            "response_content is legitimately ''."
        ),
    )
    provider: str = Field(
        ...,
        description="Provider that generated response: Qwen3-4B-AWQ|mlx_ring|deepseek|claude",
    )
    validation_passed: bool | None = Field(
        ...,
        description=(
            "Tri-state JSON-schema validation outcome: True = response passed schema "
            "validation; False = response failed validation; None = no schema was "
            "applicable to this routing rule (validation skipped). Distinguishing "
            "None from True keeps downstream pass-rate metrics honest."
        ),
    )
    fallback_used: bool = Field(default=False, description="Whether this was a fallback response")
    fallback_reason: str | None = Field(default=None, description="Why fallback was needed")
    latency_ms: int = Field(..., ge=0, description="Response latency in milliseconds")
    failover_chain_id: str = Field(..., description="UUID linking all attempts for one request")
    used_in_training: bool = Field(default=False, description="Whether used in training data")
    training_batch_id: str | None = Field(
        default=None, description="Which training batch used this"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SampleCreate(BaseModel):
    """Request body for ``POST /api/samples``."""

    tenant_id: UUID | None = Field(
        default=None,
        description=(
            "Owning tenant, as a body fallback. The gateway-injected X-Tenant-Id "
            "header takes precedence; this is used only when no header is present. "
            "None = fleet-global sample."
        ),
    )
    schema_id: str
    job_id: str
    inference_id: str
    messages: list[dict[str, Any]]
    response_content: str
    outcome: str | None = Field(
        default=None,
        description=(
            "inference-router's exhaustive outcome partition: ok | empty | "
            "empty_result | error | invalid | aborted. A plain string, NOT an "
            "enum -- the router owns the vocabulary and a new label must never "
            "need a migration. None = written by a pre-R22 inference-router."
        ),
    )
    finish_reason: str | None = Field(
        default=None,
        description=(
            "Upstream's own finish_reason (stop | length | tool_calls | ...). "
            "Turns an 'empty' outcome into a diagnosis: finish_reason='length' "
            "is a burned budget, 'tool_calls' is a healthy blank-content reply."
        ),
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Upstream tool-call payload (OpenAI shape: a list of dicts with id, "
            "type and function.name/function.arguments, arguments being a JSON "
            "string). For tool-calling traffic this is the ANSWER -- the row's "
            "response_content is legitimately ''."
        ),
    )
    provider: str
    validation_passed: bool | None
    fallback_used: bool = False
    fallback_reason: str | None = None
    latency_ms: int = Field(..., ge=0)
    failover_chain_id: str


class SampleFilter(BaseModel):
    """Query parameters for ``GET /api/samples``."""

    schema_id: str | None = None
    job_id: str | None = None
    inference_id: str | None = None
    failover_chain_id: str | None = None
    provider: str | None = None
    validation_passed: bool | None = None
    used_in_training: bool | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class SampleStats(BaseModel):
    """Aggregation statistics for a single schema_id."""

    schema_id: str
    total_samples: int
    validated_samples: int
    failed_samples: int
    used_in_training: int
    available_for_training: int  # validated but not yet used
    provider_breakdown: dict[str, int]
    avg_latency_ms: float
    fallback_rate: float


class MarkUsedRequest(BaseModel):
    """Request body for ``PATCH /api/samples/mark-used``."""

    sample_ids: list[UUID]
    training_batch_id: str


class MarkUsedResponse(BaseModel):
    """Response for the mark-used endpoint."""

    updated_count: int
    training_batch_id: str


class ExportParams(BaseModel):
    """Query parameters for ``GET /api/samples/export``."""

    schema_id: str
    only_validated: bool = True
    only_unused: bool = True
    provider: str | None = None
    fallback_used: bool | None = None
    limit: int | None = None


class PurgeResponse(BaseModel):
    """Response body for ``DELETE /api/samples/purge``."""

    purged_count: int
    filters_applied: dict[str, str]


class SampleCountResponse(BaseModel):
    """Response body for ``GET /api/samples/count``."""

    count: int
    schema_id: str | None = None


class SchemaListResponse(BaseModel):
    """Response body for ``GET /api/schemas``."""

    schema_ids: list[str]


class LivenessResponse(BaseModel):
    """Liveness probe response body."""

    status: str = "alive"


class HealthResponse(BaseModel):
    """Health-check response body."""

    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness-check response body."""

    status: str
    database: str
