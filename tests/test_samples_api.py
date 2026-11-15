"""Tests for the samples API endpoints.

Every test makes real assertions about persisted data, field values,
relationships between operations, and error conditions -- not just status codes.
"""

import json
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.db import InferenceSampleDB

# Reads are tenant-scoped (see src/read_scope.py). These tests predate that and
# exercise filtering/pagination/stats mechanics, not tenancy: they seed fleet-global
# (NULL-tenant) rows, which a tenanted caller still sees under the global-or-own
# predicate. Presenting a tenant keeps their assertions intact AND makes them a
# regression net -- if scoping were ever narrowed to own-rows-only, they go red.
_AS_TENANT = {"X-Tenant-Id": "019de842-c92a-7e37-8fdd-eb0d8c5396f9"}

# ---------------------------------------------------------------------------
# POST /api/samples
# ---------------------------------------------------------------------------


class TestCreateSample:
    async def test_create_sample_returns_201_with_all_fields(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201
        body = resp.json()

        # Verify all input fields echoed back correctly
        assert body["schema_id"] == sample_payload["schema_id"]
        assert body["job_id"] == sample_payload["job_id"]
        assert body["provider"] == sample_payload["provider"]
        assert body["validation_passed"] == sample_payload["validation_passed"]
        assert body["fallback_used"] == sample_payload["fallback_used"]
        assert body["fallback_reason"] == sample_payload["fallback_reason"]
        assert body["latency_ms"] == sample_payload["latency_ms"]
        assert body["failover_chain_id"] == sample_payload["failover_chain_id"]
        assert body["messages"] == sample_payload["messages"]
        assert body["response_content"] == sample_payload["response_content"]

        # Verify server-generated fields
        assert body["used_in_training"] is False
        assert body["training_batch_id"] is None
        # id should be a valid UUID
        UUID(body["id"])
        # created_at should be present and non-empty
        assert len(body["created_at"]) > 0

    async def test_create_sample_persisted_in_db(
        self, client: AsyncClient, session: AsyncSession, sample_payload: dict[str, Any]
    ) -> None:
        """The created sample should actually exist in the database."""
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201
        sample_id = resp.json()["id"]

        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == sample_id)
        db_result = await session.execute(stmt)
        row = db_result.scalar_one()
        assert row.schema_id == sample_payload["schema_id"]
        assert row.provider == sample_payload["provider"]
        assert row.latency_ms == sample_payload["latency_ms"]
        assert row.content_hash is not None

    async def test_create_sample_with_fallback(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        sample_payload["fallback_used"] = True
        sample_payload["fallback_reason"] = "primary provider timeout"
        sample_payload["provider"] = "claude"
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201
        body = resp.json()
        assert body["fallback_used"] is True
        assert body["fallback_reason"] == "primary provider timeout"
        assert body["provider"] == "claude"

    async def test_create_sample_deduplication(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """Posting the same content twice should return the same sample (dedup)."""
        payload = make_sample()
        r1 = await client.post("/api/samples", json=payload)
        r2 = await client.post("/api/samples", json=payload)
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["id"] == r2.json()["id"]

    async def test_create_sample_negative_latency_rejected(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        sample_payload["latency_ms"] = -1
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any("latency_ms" in str(err) for err in detail)

    async def test_create_sample_missing_required_field(self, client: AsyncClient) -> None:
        resp = await client.post("/api/samples", json={"schema_id": "test"})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        # Should flag multiple missing required fields
        missing_fields = {err["loc"][-1] for err in detail if err["type"] == "missing"}
        assert "job_id" in missing_fields
        assert "inference_id" in missing_fields
        assert "messages" in missing_fields
        assert "response_content" in missing_fields
        assert "provider" in missing_fields

    async def test_create_sample_zero_latency_accepted(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """latency_ms=0 should be accepted (ge=0 allows zero)."""
        sample_payload["latency_ms"] = 0
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201
        assert resp.json()["latency_ms"] == 0

    async def test_create_sample_accepts_null_reasoning_content(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """Regression: assistant turns with reasoning_content=null must not 422.

        inference-router posts samples whose ``messages`` array includes
        assistant turns from non-reasoning providers (or echo-back patterns)
        with ``reasoning_content: null``. Studio v3 e2e validation surfaced
        this — the schema is dict[str, Any] so any null-valued key inside a
        message dict is accepted.
        """
        sample_payload["messages"] = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "4",
                "reasoning_content": None,
            },
        ]
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["messages"][1]["reasoning_content"] is None

    async def test_create_sample_accepts_null_content_with_reasoning(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """Regression: reasoning models may emit content=null with reasoning_content set."""
        sample_payload["messages"] = [
            {"role": "user", "content": "Solve."},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "step 1...",
            },
        ]
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201, resp.text


# ---------------------------------------------------------------------------
# POST /api/samples -- outcome / finish_reason / tool_calls (R22 sample truth)
# ---------------------------------------------------------------------------

# A realistic upstream OpenAI-shaped tool-call payload: a LIST of dicts whose
# ``function.arguments`` is itself a JSON *string*, not a nested object. The
# ~97% of blank ``response_content`` rows that are healthy tool-call replies
# carry their real answer here and nowhere else.
OPENAI_TOOL_CALLS: list[dict[str, Any]] = [
    {
        "id": "call_9xKQ2mVb1p",
        "type": "function",
        "function": {
            "name": "search_entities",
            "arguments": '{"query": "Apple Inc", "limit": 5}',
        },
    },
    {
        "id": "call_7bNw4tRz8q",
        "type": "function",
        "function": {
            "name": "emit_result",
            "arguments": '{"entities": [{"text": "Apple", "type": "ORG"}]}',
        },
    },
]

# inference-router's exhaustive outcome partition, each paired with a DISTINCT
# finish_reason / tool_calls triple. Distinct-per-case on purpose: a mutant that
# always writes NULL, or always writes "ok", dies on at least one case.
OUTCOME_CASES: list[tuple[str, str | None, list[dict[str, Any]] | None]] = [
    ("ok", "stop", None),
    ("ok", "tool_calls", OPENAI_TOOL_CALLS),
    ("empty", "length", None),
    ("empty_result", "stop", []),
    ("error", None, None),
    ("invalid", "content_filter", None),
    ("aborted", "abort", None),
]


class TestCreateSampleOutcomeFields:
    """``outcome`` / ``finish_reason`` / ``tool_calls`` separate the three
    completely different realities that all land as ``response_content = ''``:
    a healthy tool-call reply, a breaker rejection, and a genuine blank
    completion. Without them this training corpus cannot tell them apart."""

    @pytest.mark.parametrize(
        ("outcome", "finish_reason", "tool_calls"),
        OUTCOME_CASES,
        ids=[c[0] + "-" + str(c[1]) for c in OUTCOME_CASES],
    )
    async def test_create_sample_round_trips_outcome_fields(
        self,
        client: AsyncClient,
        session: AsyncSession,
        make_sample: Callable[..., dict[str, Any]],
        outcome: str,
        finish_reason: str | None,
        tool_calls: list[dict[str, Any]] | None,
    ) -> None:
        """Each triple must survive POST -> DB -> response body intact."""
        payload = make_sample(
            outcome=outcome,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )
        resp = await client.post("/api/samples", json=payload)
        assert resp.status_code == 201, resp.text
        body = resp.json()

        # Response body carries the exact values posted.
        assert body.get("outcome") == outcome
        assert body.get("finish_reason") == finish_reason
        assert body.get("tool_calls") == tool_calls

        # ...and so does the row actually re-read from Postgres. The
        # ``getattr`` sentinel keeps a missing column an AssertionError
        # (a real RED) rather than an AttributeError.
        db_result = await session.execute(
            select(InferenceSampleDB).where(InferenceSampleDB.id == body["id"])
        )
        row = db_result.scalar_one()
        assert getattr(row, "outcome", "<no-such-column>") == outcome
        assert getattr(row, "finish_reason", "<no-such-column>") == finish_reason
        assert getattr(row, "tool_calls", "<no-such-column>") == tool_calls

    async def test_create_sample_without_outcome_fields_is_null(
        self, client: AsyncClient, session: AsyncSession, sample_payload: dict[str, Any]
    ) -> None:
        """Old-inference-router compat: a body with none of the three fields
        must still 201, and store NULL (not '' and not a default label)."""
        assert "outcome" not in sample_payload
        assert "finish_reason" not in sample_payload
        assert "tool_calls" not in sample_payload

        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201, resp.text
        body = resp.json()

        # The keys must EXIST on the response model and be null -- not absent.
        assert "outcome" in body, "outcome missing from the response model"
        assert "finish_reason" in body, "finish_reason missing from the response model"
        assert "tool_calls" in body, "tool_calls missing from the response model"
        assert body["outcome"] is None
        assert body["finish_reason"] is None
        assert body["tool_calls"] is None

        db_result = await session.execute(
            select(InferenceSampleDB).where(InferenceSampleDB.id == body["id"])
        )
        row = db_result.scalar_one()
        assert getattr(row, "outcome", "<no-such-column>") is None
        assert getattr(row, "finish_reason", "<no-such-column>") is None
        assert getattr(row, "tool_calls", "<no-such-column>") is None

    async def test_absent_tool_calls_is_sql_null_not_the_json_scalar_null(
        self, client: AsyncClient, session: AsyncSession, sample_payload: dict[str, Any]
    ) -> None:
        """⛔ THE ORM CANNOT SEE THIS ONE, WHICH IS WHY IT HAS ITS OWN TEST.

        SQLAlchemy's JSON/JSONB default is ``none_as_null=False``: Python
        ``None`` is written as the JSON scalar ``'null'``, and then read back
        as ``None``. Every ORM-level assertion above passes either way, so
        ``assert row.tool_calls is None`` is blind to the difference.

        In the DATABASE it is not blind. ``tool_calls IS NOT NULL`` -- the one
        query this column exists to serve, "find the tool-call replies among
        the blank ``response_content`` rows" -- would match EVERY row this
        service writes, a 100% false-positive rate. And pre-cutover rows are
        genuine SQL NULL, so the corpus would hold two permanent, silently
        different spellings of "absent".

        Asserted in SQL (``jsonb_typeof`` / ``IS NULL``), never through the
        ORM.
        """
        assert "tool_calls" not in sample_payload
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201, resp.text
        row_id = resp.json()["id"]

        probe = await session.execute(
            text(
                "SELECT tool_calls IS NULL AS is_sql_null, "
                "jsonb_typeof(tool_calls) AS json_type "
                "FROM inference_samples WHERE id = :id"
            ),
            {"id": row_id},
        )
        is_sql_null, json_type = probe.one()
        assert is_sql_null is True, (
            f"tool_calls was stored as the JSON scalar {json_type!r}, not SQL NULL -- "
            "`WHERE tool_calls IS NOT NULL` would match every row this service writes"
        )
        assert json_type is None, f"expected no jsonb type for SQL NULL, got {json_type!r}"

        # Positive control: a row that DOES carry tool calls must be IS NOT NULL,
        # so the assertion above cannot pass by the column being unreachable.
        with_calls = dict(sample_payload)
        with_calls["inference_id"] = sample_payload["inference_id"] + "-withcalls"
        with_calls["job_id"] = sample_payload["job_id"] + "-withcalls"
        with_calls["tool_calls"] = [
            {
                "id": "call_ctl",
                "type": "function",
                "function": {"name": "record_answer", "arguments": '{"answer": "Paris"}'},
            }
        ]
        ctl = await client.post("/api/samples", json=with_calls)
        assert ctl.status_code == 201, ctl.text
        ctl_probe = await session.execute(
            text(
                "SELECT tool_calls IS NULL AS is_sql_null, "
                "jsonb_typeof(tool_calls) AS json_type "
                "FROM inference_samples WHERE id = :id"
            ),
            {"id": ctl.json()["id"]},
        )
        ctl_is_null, ctl_type = ctl_probe.one()
        assert ctl_is_null is False, "positive control: a real tool-call payload read as SQL NULL"
        assert ctl_type == "array", f"positive control: expected jsonb array, got {ctl_type!r}"

    async def test_tool_calls_round_trips_nested_openai_payload(
        self, client: AsyncClient, session: AsyncSession, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """The realistic nested shape must survive JSONB byte-for-byte --
        including ``function.arguments`` staying a JSON *string*, not being
        re-parsed into an object."""
        payload = make_sample(
            # The canonical blank-content tool-call reply this work exists for.
            response_content="",
            outcome="ok",
            finish_reason="tool_calls",
            tool_calls=OPENAI_TOOL_CALLS,
        )
        resp = await client.post("/api/samples", json=payload)
        assert resp.status_code == 201, resp.text
        body = resp.json()

        db_result = await session.execute(
            select(InferenceSampleDB).where(InferenceSampleDB.id == body["id"])
        )
        row = db_result.scalar_one()
        stored = getattr(row, "tool_calls", "<no-such-column>")

        assert stored == OPENAI_TOOL_CALLS
        assert isinstance(stored, list) and len(stored) == 2
        assert stored[0]["id"] == "call_9xKQ2mVb1p"
        assert stored[0]["type"] == "function"
        assert stored[0]["function"]["name"] == "search_entities"
        # arguments stays a STRING (upstream's own encoding), not a dict.
        assert isinstance(stored[0]["function"]["arguments"], str)
        assert json.loads(stored[0]["function"]["arguments"]) == {"query": "Apple Inc", "limit": 5}
        assert stored[1]["function"]["name"] == "emit_result"
        # The row it belongs to really is one of the blank-content rows.
        assert row.response_content == ""


# ---------------------------------------------------------------------------
# GET /api/samples
# ---------------------------------------------------------------------------


class TestListSamples:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/samples", headers=_AS_TENANT)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_returns_created_samples(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        create_resp = await client.post("/api/samples", json=sample_payload)
        created_id = create_resp.json()["id"]

        resp = await client.get("/api/samples", headers=_AS_TENANT)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == created_id
        assert data[0]["schema_id"] == sample_payload["schema_id"]
        assert data[0]["provider"] == sample_payload["provider"]

    async def test_list_filter_by_schema_id(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample())
        r2 = await client.post("/api/samples", json=make_sample(schema_id="ks_OtherAgent"))

        resp = await client.get(
            "/api/samples", params={"schema_id": "ks_OtherAgent"}, headers=_AS_TENANT
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["schema_id"] == "ks_OtherAgent"
        assert data[0]["id"] == r2.json()["id"]

    async def test_list_filter_by_provider(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample(provider="Qwen3-4B-AWQ"))
        r2 = await client.post("/api/samples", json=make_sample(provider="claude"))

        resp = await client.get("/api/samples", params={"provider": "claude"}, headers=_AS_TENANT)
        data = resp.json()
        assert len(data) == 1
        assert data[0]["provider"] == "claude"
        assert data[0]["id"] == r2.json()["id"]

        # Non-existent provider returns empty
        resp2 = await client.get(
            "/api/samples", params={"provider": "nonexistent"}, headers=_AS_TENANT
        )
        assert resp2.json() == []

    async def test_list_filter_by_validation_passed(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample(validation_passed=True))
        await client.post("/api/samples", json=make_sample(validation_passed=False))

        resp = await client.get(
            "/api/samples", params={"validation_passed": True}, headers=_AS_TENANT
        )
        data = resp.json()
        assert len(data) == 1
        assert all(s["validation_passed"] is True for s in data)

        resp2 = await client.get(
            "/api/samples", params={"validation_passed": False}, headers=_AS_TENANT
        )
        data2 = resp2.json()
        assert len(data2) == 1
        assert all(s["validation_passed"] is False for s in data2)

    async def test_list_pagination(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        created_ids = []
        for _ in range(3):
            r = await client.post("/api/samples", json=make_sample())
            created_ids.append(r.json()["id"])

        resp = await client.get(
            "/api/samples", params={"limit": 2, "offset": 0}, headers=_AS_TENANT
        )
        page1 = resp.json()
        assert len(page1) == 2

        resp2 = await client.get(
            "/api/samples", params={"limit": 2, "offset": 2}, headers=_AS_TENANT
        )
        page2 = resp2.json()
        assert len(page2) == 1

        # All IDs across pages should be from our created set
        all_returned_ids = {s["id"] for s in page1 + page2}
        assert all_returned_ids == set(created_ids)

    async def test_list_naive_date_bounds_coerced_to_utc(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """Naive start/end date bounds are treated as UTC (no tz comparison error)."""
        r = await client.post("/api/samples", json=make_sample())
        created_id = r.json()["id"]

        # Naive datetimes (no offset). A wide window must include the fresh row.
        resp = await client.get(
            "/api/samples",
            params={
                "start_date": "2000-01-01T00:00:00",
                "end_date": "2999-12-31T23:59:59",
            },
            headers=_AS_TENANT,
        )
        assert resp.status_code == 200
        assert created_id in {s["id"] for s in resp.json()}

        # A naive upper bound in the far past excludes the fresh row (bound applied).
        resp_past = await client.get(
            "/api/samples",
            params={"end_date": "2000-01-01T00:00:00"},
            headers=_AS_TENANT,
        )
        assert resp_past.status_code == 200
        assert created_id not in {s["id"] for s in resp_past.json()}

    async def test_list_ordering_newest_first(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """List endpoint should return newest samples first."""
        r1 = await client.post("/api/samples", json=make_sample())
        r2 = await client.post("/api/samples", json=make_sample())
        r3 = await client.post("/api/samples", json=make_sample())

        resp = await client.get("/api/samples", headers=_AS_TENANT)
        data = resp.json()
        assert len(data) == 3
        assert data[0]["id"] == r3.json()["id"]
        assert data[1]["id"] == r2.json()["id"]
        assert data[2]["id"] == r1.json()["id"]

    async def test_list_filter_by_job_id(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """GET /api/samples?job_id=X should filter by job_id."""
        await client.post("/api/samples", json=make_sample(job_id="job-filter-A"))
        r2 = await client.post("/api/samples", json=make_sample(job_id="job-filter-B"))

        resp = await client.get(
            "/api/samples", params={"job_id": "job-filter-B"}, headers=_AS_TENANT
        )
        data = resp.json()
        assert len(data) == 1
        assert data[0]["job_id"] == "job-filter-B"
        assert data[0]["id"] == r2.json()["id"]

    async def test_list_filter_by_inference_id(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """GET /api/samples?inference_id=X should return exact match."""
        r1 = await client.post("/api/samples", json=make_sample(inference_id="inf-api-test-001"))
        await client.post("/api/samples", json=make_sample(inference_id="inf-api-test-002"))

        resp = await client.get(
            "/api/samples", params={"inference_id": "inf-api-test-001"}, headers=_AS_TENANT
        )
        data = resp.json()
        assert len(data) == 1
        assert data[0]["inference_id"] == "inf-api-test-001"
        assert data[0]["id"] == r1.json()["id"]

    async def test_list_filter_by_failover_chain_id(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """GET /api/samples?failover_chain_id=X should filter correctly."""
        r1 = await client.post("/api/samples", json=make_sample(failover_chain_id="chain-api-001"))
        await client.post("/api/samples", json=make_sample(failover_chain_id="chain-api-002"))

        resp = await client.get(
            "/api/samples", params={"failover_chain_id": "chain-api-001"}, headers=_AS_TENANT
        )
        data = resp.json()
        assert len(data) == 1
        assert data[0]["failover_chain_id"] == "chain-api-001"
        assert data[0]["id"] == r1.json()["id"]

    async def test_cross_job_dedup_via_api(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """Same content with different job_ids should create separate samples via API."""
        shared_content = '{"entities": [{"text": "Apple", "type": "ORG", "cross_job": true}]}'
        shared_messages = [
            {"role": "system", "content": "You are an entity spotter."},
            {"role": "user", "content": "Find entities."},
        ]
        r1 = await client.post(
            "/api/samples",
            json=make_sample(
                job_id="job-cross-A",
                messages=shared_messages,
                response_content=shared_content,
            ),
        )
        r2 = await client.post(
            "/api/samples",
            json=make_sample(
                job_id="job-cross-B",
                messages=shared_messages,
                response_content=shared_content,
            ),
        )
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["id"] != r2.json()["id"]


# ---------------------------------------------------------------------------
# GET /api/samples/count
# ---------------------------------------------------------------------------


class TestSampleCount:
    async def test_count_empty_db(self, client: AsyncClient) -> None:
        resp = await client.get("/api/samples/count", params={"schema_id": "ks_EntitySpotterAgent"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["schema_id"] == "ks_EntitySpotterAgent"

    async def test_count_after_inserts(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample())
        await client.post("/api/samples", json=make_sample())
        resp = await client.get("/api/samples/count", params={"schema_id": "ks_EntitySpotterAgent"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    async def test_count_only_validated(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample(validation_passed=True))
        await client.post("/api/samples", json=make_sample(validation_passed=False))

        resp = await client.get(
            "/api/samples/count",
            params={"schema_id": "ks_EntitySpotterAgent", "only_validated": True},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    async def test_count_only_unused(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        create_resp = await client.post("/api/samples", json=make_sample())
        sample_id = create_resp.json()["id"]
        await client.patch(
            "/api/samples/mark-used",
            json={"sample_ids": [sample_id], "training_batch_id": "batch-001"},
        )
        await client.post("/api/samples", json=make_sample())

        resp = await client.get(
            "/api/samples/count",
            params={"schema_id": "ks_EntitySpotterAgent", "only_unused": True},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    async def test_count_validated_and_unused(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        # validated + unused
        await client.post("/api/samples", json=make_sample())
        # validated + used
        create_resp = await client.post("/api/samples", json=make_sample())
        sample_id = create_resp.json()["id"]
        await client.patch(
            "/api/samples/mark-used",
            json={"sample_ids": [sample_id], "training_batch_id": "batch-002"},
        )
        # failed + unused
        await client.post("/api/samples", json=make_sample(validation_passed=False))

        resp = await client.get(
            "/api/samples/count",
            params={
                "schema_id": "ks_EntitySpotterAgent",
                "only_validated": True,
                "only_unused": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    async def test_count_aggregates_across_schemas_when_schema_id_omitted(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """Omitting schema_id returns a count aggregated across all schemas — the flywheel global-retrain trigger."""
        await client.post("/api/samples", json=make_sample(schema_id="ks_A"))
        await client.post("/api/samples", json=make_sample(schema_id="ks_A"))
        await client.post("/api/samples", json=make_sample(schema_id="ks_B"))

        resp = await client.get("/api/samples/count")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert body["schema_id"] is None

    async def test_count_isolated_between_schemas(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """Count for one schema should not include samples from another."""
        await client.post("/api/samples", json=make_sample(schema_id="ks_A"))
        await client.post("/api/samples", json=make_sample(schema_id="ks_A"))
        await client.post("/api/samples", json=make_sample(schema_id="ks_B"))

        resp_a = await client.get("/api/samples/count", params={"schema_id": "ks_A"})
        assert resp_a.json()["count"] == 2

        resp_b = await client.get("/api/samples/count", params={"schema_id": "ks_B"})
        assert resp_b.json()["count"] == 1


# ---------------------------------------------------------------------------
# GET /api/schemas
# ---------------------------------------------------------------------------


class TestSchemaList:
    async def test_schema_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/schemas")
        assert resp.status_code == 200
        body = resp.json()
        assert body["schema_ids"] == []

    async def test_schema_list_returns_unique_sorted_ids(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        # Insert samples with different schema_ids (including duplicates via dedup)
        p = make_sample()
        await client.post("/api/samples", json=p)
        await client.post("/api/samples", json=p)  # duplicate (dedup returns existing)
        await client.post("/api/samples", json=make_sample(schema_id="ks_OtherAgent"))
        await client.post("/api/samples", json=make_sample(schema_id="ks_AAgent"))

        resp = await client.get("/api/schemas")
        assert resp.status_code == 200
        schema_ids = resp.json()["schema_ids"]
        assert len(schema_ids) == 3
        assert "ks_EntitySpotterAgent" in schema_ids
        assert "ks_OtherAgent" in schema_ids
        assert "ks_AAgent" in schema_ids
        # Should be sorted alphabetically
        assert schema_ids == sorted(schema_ids)


# ---------------------------------------------------------------------------
# GET /api/samples/stats
# ---------------------------------------------------------------------------


class TestStats:
    async def test_stats_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/samples/stats", headers=_AS_TENANT)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_stats_comprehensive(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """Verify all stats fields with a known set of samples."""
        await client.post(
            "/api/samples",
            json=make_sample(validation_passed=True, latency_ms=100, fallback_used=False),
        )
        await client.post(
            "/api/samples",
            json=make_sample(
                validation_passed=False,
                latency_ms=200,
                fallback_used=True,
                fallback_reason="timeout",
                provider="claude",
            ),
        )

        resp = await client.get("/api/samples/stats", headers=_AS_TENANT)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        stats = data[0]
        assert stats["schema_id"] == "ks_EntitySpotterAgent"
        assert stats["total_samples"] == 2
        assert stats["validated_samples"] == 1
        assert stats["failed_samples"] == 1
        assert stats["used_in_training"] == 0
        assert stats["available_for_training"] == 1
        assert stats["avg_latency_ms"] == 150.0
        assert stats["fallback_rate"] == 0.5
        # Provider breakdown
        assert "Qwen3-4B-AWQ" in stats["provider_breakdown"]
        assert "claude" in stats["provider_breakdown"]
        assert stats["provider_breakdown"]["Qwen3-4B-AWQ"] == 1
        assert stats["provider_breakdown"]["claude"] == 1

    async def test_stats_filter_by_schema_id(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample())
        await client.post("/api/samples", json=make_sample(schema_id="ks_Other"))

        resp = await client.get(
            "/api/samples/stats", params={"schema_id": "ks_Other"}, headers=_AS_TENANT
        )
        data = resp.json()
        assert len(data) == 1
        assert data[0]["schema_id"] == "ks_Other"
        assert data[0]["total_samples"] == 1

    async def test_stats_reflects_mark_used(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """After marking a sample as used, stats should reflect the change."""
        r = await client.post("/api/samples", json=make_sample())
        sample_id = r.json()["id"]

        # Before marking
        resp1 = await client.get("/api/samples/stats", headers=_AS_TENANT)
        stats1 = resp1.json()[0]
        assert stats1["used_in_training"] == 0
        assert stats1["available_for_training"] == 1

        await client.patch(
            "/api/samples/mark-used",
            json={"sample_ids": [sample_id], "training_batch_id": "batch-stats"},
        )

        # After marking
        resp2 = await client.get("/api/samples/stats", headers=_AS_TENANT)
        stats2 = resp2.json()[0]
        assert stats2["used_in_training"] == 1
        assert stats2["available_for_training"] == 0


# ---------------------------------------------------------------------------
# PATCH /api/samples/mark-used
# ---------------------------------------------------------------------------


class TestMarkUsed:
    async def test_mark_used(self, client: AsyncClient, sample_payload: dict[str, Any]) -> None:
        create_resp = await client.post("/api/samples", json=sample_payload)
        sample_id = create_resp.json()["id"]

        mark_resp = await client.patch(
            "/api/samples/mark-used",
            json={
                "sample_ids": [sample_id],
                "training_batch_id": "batch-001",
            },
        )
        assert mark_resp.status_code == 200
        body = mark_resp.json()
        assert body["updated_count"] == 1
        assert body["training_batch_id"] == "batch-001"

        # Verify the sample is now marked via list endpoint
        list_resp = await client.get(
            "/api/samples", params={"used_in_training": True}, headers=_AS_TENANT
        )
        data = list_resp.json()
        assert len(data) == 1
        assert data[0]["id"] == sample_id
        assert data[0]["used_in_training"] is True
        assert data[0]["training_batch_id"] == "batch-001"

    async def test_mark_used_nonexistent_ids(self, client: AsyncClient) -> None:
        resp = await client.patch(
            "/api/samples/mark-used",
            json={
                "sample_ids": ["00000000-0000-0000-0000-000000000000"],
                "training_batch_id": "batch-nope",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["updated_count"] == 0
        assert resp.json()["training_batch_id"] == "batch-nope"

    async def test_mark_used_multiple_samples(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """Marking multiple samples used in one batch."""
        r1 = await client.post("/api/samples", json=make_sample())
        r2 = await client.post("/api/samples", json=make_sample())
        r3 = await client.post("/api/samples", json=make_sample())

        resp = await client.patch(
            "/api/samples/mark-used",
            json={
                "sample_ids": [r1.json()["id"], r2.json()["id"]],
                "training_batch_id": "batch-multi",
            },
        )
        assert resp.json()["updated_count"] == 2

        # r3 should still be unused
        list_resp = await client.get(
            "/api/samples", params={"used_in_training": False}, headers=_AS_TENANT
        )
        data = list_resp.json()
        assert len(data) == 1
        assert data[0]["id"] == r3.json()["id"]

    async def test_mark_used_persisted_in_db(
        self,
        client: AsyncClient,
        session: AsyncSession,
        sample_payload: dict[str, Any],
    ) -> None:
        """Verify mark-used writes to DB, not just returns a response."""
        create_resp = await client.post("/api/samples", json=sample_payload)
        sample_id = create_resp.json()["id"]

        await client.patch(
            "/api/samples/mark-used",
            json={"sample_ids": [sample_id], "training_batch_id": "batch-db"},
        )

        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == sample_id)
        db_result = await session.execute(stmt)
        row = db_result.scalar_one()
        assert row.used_in_training is True
        assert row.training_batch_id == "batch-db"


# ---------------------------------------------------------------------------
# DELETE /api/samples/purge
# ---------------------------------------------------------------------------


class TestPurge:
    async def test_purge_requires_at_least_one_filter(self, client: AsyncClient) -> None:
        resp = await client.delete("/api/samples/purge")
        assert resp.status_code == 400
        assert "At least one filter" in resp.json()["detail"]

    async def test_purge_by_schema_id(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample())
        await client.post("/api/samples", json=make_sample(schema_id="ks_Other"))

        resp = await client.delete(
            "/api/samples/purge",
            params={"schema_id": "ks_EntitySpotterAgent", "all_tenants": "true"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["purged_count"] == 1
        assert body["filters_applied"]["schema_id"] == "ks_EntitySpotterAgent"

        # Verify remaining
        list_resp = await client.get("/api/samples", headers=_AS_TENANT)
        remaining = list_resp.json()
        assert len(remaining) == 1
        assert remaining[0]["schema_id"] == "ks_Other"

    async def test_purge_by_provider(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample(provider="Qwen3-4B-AWQ"))
        await client.post("/api/samples", json=make_sample(provider="claude"))

        resp = await client.delete(
            "/api/samples/purge",
            params={"provider": "claude", "all_tenants": "true"},
        )
        assert resp.status_code == 200
        assert resp.json()["purged_count"] == 1
        assert resp.json()["filters_applied"]["provider"] == "claude"

        list_resp = await client.get("/api/samples", headers=_AS_TENANT)
        remaining = list_resp.json()
        assert len(remaining) == 1
        assert remaining[0]["provider"] == "Qwen3-4B-AWQ"

    async def test_purge_by_validation_passed(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample(validation_passed=True))
        await client.post("/api/samples", json=make_sample(validation_passed=False))

        resp = await client.delete(
            "/api/samples/purge",
            params={"validation_passed": False, "all_tenants": "true"},
        )
        assert resp.status_code == 200
        assert resp.json()["purged_count"] == 1

        list_resp = await client.get("/api/samples", headers=_AS_TENANT)
        remaining = list_resp.json()
        assert len(remaining) == 1
        assert remaining[0]["validation_passed"] is True

    async def test_purge_by_before_date(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample())

        resp = await client.delete(
            "/api/samples/purge",
            params={"before_date": "2099-12-31T23:59:59Z", "all_tenants": "true"},
        )
        assert resp.status_code == 200
        assert resp.json()["purged_count"] == 1

        list_resp = await client.get("/api/samples", headers=_AS_TENANT)
        assert list_resp.json() == []

    async def test_purge_no_matches(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample())

        resp = await client.delete(
            "/api/samples/purge",
            params={"schema_id": "ks_NonExistent", "all_tenants": "true"},
        )
        assert resp.status_code == 200
        assert resp.json()["purged_count"] == 0

        # Original sample still exists
        list_resp = await client.get("/api/samples", headers=_AS_TENANT)
        assert len(list_resp.json()) == 1

    async def test_purge_verified_in_db(
        self,
        client: AsyncClient,
        session: AsyncSession,
        make_sample: Callable[..., dict[str, Any]],
    ) -> None:
        """Verify purge actually deletes rows from the database."""
        r = await client.post("/api/samples", json=make_sample())
        sample_id = r.json()["id"]

        await client.delete(
            "/api/samples/purge",
            params={"schema_id": "ks_EntitySpotterAgent", "all_tenants": "true"},
        )

        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == sample_id)
        db_result = await session.execute(stmt)
        assert db_result.scalar_one_or_none() is None


class TestPurgeTenantIsolation:
    """``DELETE /api/samples/purge`` must not delete another tenant's samples.

    The handler runs on ``get_owner_session`` — the RLS-BYPASS session — and its
    DELETE carried no tenant predicate, so any caller who reached the route
    deleted every tenant's matching rows. ``TENANT_RLS_MODE=enforce`` is live in
    production, but RLS on this table is ENABLE-not-FORCE and the owner role owns
    the table, so migration 007's ``_GLOBAL_DEL`` policy never applied here. The
    pre-existing empty-filter guard is not a tenancy control: one ``schema_id``
    satisfies it and still spans every tenant.

    Against ``origin/main`` the first two tests FAIL — the purge returns 200,
    reports 2 rows, and tenant B's sample is gone.
    """

    async def test_purge_spares_another_tenants_samples(
        self,
        client: AsyncClient,
        session: AsyncSession,
        make_sample: Callable[..., dict[str, Any]],
    ) -> None:
        """Tenant A purging its own schema must not touch tenant B's rows."""
        tenant_a = "11111111-1111-7111-8111-111111111111"
        tenant_b = "22222222-2222-7222-8222-222222222222"

        a = await client.post(
            "/api/samples",
            json=make_sample(schema_id="ks_Shared"),
            headers={"X-Tenant-Id": tenant_a},
        )
        b = await client.post(
            "/api/samples",
            json=make_sample(schema_id="ks_Shared"),
            headers={"X-Tenant-Id": tenant_b},
        )
        assert a.status_code == 201
        assert b.status_code == 201
        b_id = b.json()["id"]

        resp = await client.delete(
            "/api/samples/purge",
            params={"schema_id": "ks_Shared"},
            headers={"X-Tenant-Id": tenant_a},
        )

        assert resp.status_code == 200
        body = resp.json()
        # Exactly ONE row — tenant A's. Not two.
        assert body["purged_count"] == 1
        assert body["filters_applied"]["tenant_scope"] == tenant_a

        # Tenant B's sample survives, in the database, not just in a response.
        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == b_id)
        assert (await session.execute(stmt)).scalar_one_or_none() is not None

    async def test_purge_spares_the_fleet_global_rows(
        self,
        client: AsyncClient,
        session: AsyncSession,
        make_sample: Callable[..., dict[str, Any]],
    ) -> None:
        """A tenant-scoped purge must not sweep the NULL-tenant (global) bucket."""
        tenant_a = "11111111-1111-7111-8111-111111111111"

        glob = await client.post("/api/samples", json=make_sample(schema_id="ks_Shared"))
        await client.post(
            "/api/samples",
            json=make_sample(schema_id="ks_Shared"),
            headers={"X-Tenant-Id": tenant_a},
        )
        global_id = glob.json()["id"]

        resp = await client.delete(
            "/api/samples/purge",
            params={"schema_id": "ks_Shared"},
            headers={"X-Tenant-Id": tenant_a},
        )

        assert resp.status_code == 200
        assert resp.json()["purged_count"] == 1

        stmt = select(InferenceSampleDB).where(InferenceSampleDB.id == global_id)
        assert (await session.execute(stmt)).scalar_one_or_none() is not None

    async def test_purge_without_a_tenant_refuses(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """An untenanted caller must ask for the cross-tenant path by name."""
        await client.post("/api/samples", json=make_sample(schema_id="ks_Shared"))

        resp = await client.delete("/api/samples/purge", params={"schema_id": "ks_Shared"})

        assert resp.status_code == 400
        assert "all_tenants=true" in resp.json()["detail"]

        # Nothing was destroyed by the refusal.
        assert len((await client.get("/api/samples", headers=_AS_TENANT)).json()) == 1

    async def test_all_tenants_is_the_explicit_admin_path(
        self,
        client: AsyncClient,
        make_sample: Callable[..., dict[str, Any]],
    ) -> None:
        """The named platform path still purges across tenants — loudly, on request."""
        await client.post(
            "/api/samples",
            json=make_sample(schema_id="ks_Shared"),
            headers={"X-Tenant-Id": "11111111-1111-7111-8111-111111111111"},
        )
        await client.post("/api/samples", json=make_sample(schema_id="ks_Shared"))

        resp = await client.delete(
            "/api/samples/purge",
            params={"schema_id": "ks_Shared", "all_tenants": "true"},
        )

        assert resp.status_code == 200
        assert resp.json()["purged_count"] == 2
        assert resp.json()["filters_applied"]["tenant_scope"] == "all_tenants"

    async def test_all_tenants_with_a_tenant_caller_is_contradictory(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        """Presenting a tenant AND asking for every tenant fails loud, not wide."""
        await client.post("/api/samples", json=make_sample(schema_id="ks_Shared"))

        resp = await client.delete(
            "/api/samples/purge",
            params={"schema_id": "ks_Shared", "all_tenants": "true"},
            headers={"X-Tenant-Id": "11111111-1111-7111-8111-111111111111"},
        )

        assert resp.status_code == 400
        assert len((await client.get("/api/samples", headers=_AS_TENANT)).json()) == 1


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_health(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "data-collector"
        assert body["version"] == "0.1.0"

    async def test_liveness(self, client: AsyncClient) -> None:
        resp = await client.get("/health/live")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "alive"

    async def test_readiness(self, client: AsyncClient) -> None:
        """Readiness endpoint should respond with status field indicating DB state."""
        resp = await client.get("/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ready", "not_ready")
        assert body["database"] in ("connected", "disconnected")

    async def test_root(self, client: AsyncClient) -> None:
        resp = await client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "data-collector"
        assert body["version"] == "0.1.0"
        assert body["docs"] == "/docs"
        assert body["health"] == "/health"
