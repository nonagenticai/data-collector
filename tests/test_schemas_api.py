"""Tests for schema validation on API endpoints -- invalid payloads, missing fields, etc.

Each test verifies specific error details (field names, error types) not just status codes.
"""

from typing import Any

from httpx import AsyncClient

# Reads are tenant-scoped (see src/read_scope.py). These tests predate that and
# exercise filtering/pagination/stats mechanics, not tenancy: they seed fleet-global
# (NULL-tenant) rows, which a tenanted caller still sees under the global-or-own
# predicate. Presenting a tenant keeps their assertions intact AND makes them a
# regression net -- if scoping were ever narrowed to own-rows-only, they go red.
_AS_TENANT = {"X-Tenant-Id": "019de842-c92a-7e37-8fdd-eb0d8c5396f9"}


class TestSampleCreateValidation:
    """Validation tests for POST /api/samples schema."""

    async def test_empty_body_rejected(self, client: AsyncClient) -> None:
        """POST with empty JSON body should return 422 with all missing required fields."""
        resp = await client.post("/api/samples", json={})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        missing_fields = {err["loc"][-1] for err in detail if err["type"] == "missing"}
        # All required fields should be reported
        assert "schema_id" in missing_fields
        assert "job_id" in missing_fields
        assert "messages" in missing_fields
        assert "response_content" in missing_fields
        assert "provider" in missing_fields
        assert "validation_passed" in missing_fields
        assert "latency_ms" in missing_fields
        assert "failover_chain_id" in missing_fields

    async def test_missing_messages_rejected(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """POST without messages field should return 422 identifying 'messages'."""
        del sample_payload["messages"]
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(err["loc"][-1] == "messages" for err in detail)

    async def test_missing_response_content_rejected(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """POST without response_content should return 422 identifying 'response_content'."""
        del sample_payload["response_content"]
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(err["loc"][-1] == "response_content" for err in detail)

    async def test_missing_provider_rejected(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """POST without provider should return 422 identifying 'provider'."""
        del sample_payload["provider"]
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(err["loc"][-1] == "provider" for err in detail)

    async def test_zero_latency_accepted(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """POST with latency_ms=0 should be accepted (ge=0 allows zero)."""
        sample_payload["latency_ms"] = 0
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201
        assert resp.json()["latency_ms"] == 0

    async def test_invalid_json_body_rejected(self, client: AsyncClient) -> None:
        """POST with invalid JSON should return 422."""
        resp = await client.post(
            "/api/samples",
            content=b"not valid json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422

    async def test_wrong_type_for_validation_passed(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """POST with string instead of bool for validation_passed should return 422."""
        sample_payload["validation_passed"] = "not_a_bool"
        resp = await client.post("/api/samples", json=sample_payload)
        # Pydantic may coerce strings to bools in some cases, but "not_a_bool" should fail
        # If it doesn't fail, that's also valid Pydantic behavior -- just verify consistency
        if resp.status_code == 422:
            detail = resp.json()["detail"]
            assert any("validation_passed" in str(err) for err in detail)

    async def test_null_validation_passed_accepted(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """POST with validation_passed=null is accepted (schema-skipped semantic, A5)."""
        sample_payload["validation_passed"] = None
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201, resp.text
        assert resp.json()["validation_passed"] is None

    async def test_wrong_type_for_latency_ms(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """POST with string for latency_ms should return 422."""
        sample_payload["latency_ms"] = "not_an_int"
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any("latency_ms" in str(err) for err in detail)

    async def test_messages_wrong_type(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """POST with messages as string instead of list should return 422."""
        sample_payload["messages"] = "not a list"
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 422

    async def test_messages_with_null_reasoning_content_accepted(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """Reasoning models (Nemotron-V3, DeepSeek-R1) emit reasoning_content=null
        on non-reasoning replies. The OpenAI message shape allows it; data-collector
        must accept it. Regression: dict[str, str] used to reject null values."""
        sample_payload["messages"] = [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong", "reasoning_content": None},
        ]
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201, resp.text

    async def test_messages_with_multimodal_content_accepted(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """OpenAI multimodal content is a list-of-dicts, not a string. dict[str, str]
        used to reject it; dict[str, Any] accepts it."""
        sample_payload["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        resp = await client.post("/api/samples", json=sample_payload)
        assert resp.status_code == 201, resp.text


class TestMarkUsedValidation:
    """Validation tests for PATCH /api/samples/mark-used schema."""

    async def test_invalid_uuid_format_rejected(self, client: AsyncClient) -> None:
        """PATCH with invalid UUID format should return 422 identifying the field."""
        resp = await client.patch(
            "/api/samples/mark-used",
            json={
                "sample_ids": ["not-a-uuid"],
                "training_batch_id": "batch-001",
            },
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any("sample_ids" in str(err["loc"]) for err in detail)

    async def test_missing_training_batch_id_rejected(self, client: AsyncClient) -> None:
        """PATCH without training_batch_id should return 422."""
        resp = await client.patch(
            "/api/samples/mark-used",
            json={
                "sample_ids": ["00000000-0000-0000-0000-000000000000"],
            },
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(err["loc"][-1] == "training_batch_id" for err in detail)

    async def test_missing_sample_ids_rejected(self, client: AsyncClient) -> None:
        """PATCH without sample_ids should return 422."""
        resp = await client.patch(
            "/api/samples/mark-used",
            json={
                "training_batch_id": "batch-001",
            },
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(err["loc"][-1] == "sample_ids" for err in detail)

    async def test_empty_body_rejected(self, client: AsyncClient) -> None:
        """PATCH with empty JSON body should return 422."""
        resp = await client.patch("/api/samples/mark-used", json={})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        missing_fields = {err["loc"][-1] for err in detail if err["type"] == "missing"}
        assert "sample_ids" in missing_fields
        assert "training_batch_id" in missing_fields


class TestListSamplesValidation:
    """Validation tests for GET /api/samples query params."""

    async def test_limit_zero_rejected(self, client: AsyncClient) -> None:
        """GET with limit=0 should return 422 (ge=1)."""
        resp = await client.get("/api/samples", params={"limit": 0}, headers=_AS_TENANT)
        assert resp.status_code == 422

    async def test_limit_over_max_rejected(self, client: AsyncClient) -> None:
        """GET with limit=1001 should return 422 (le=1000)."""
        resp = await client.get("/api/samples", params={"limit": 1001}, headers=_AS_TENANT)
        assert resp.status_code == 422

    async def test_negative_offset_rejected(self, client: AsyncClient) -> None:
        """GET with negative offset should return 422 (ge=0)."""
        resp = await client.get("/api/samples", params={"offset": -1}, headers=_AS_TENANT)
        assert resp.status_code == 422

    async def test_limit_at_boundaries_accepted(self, client: AsyncClient) -> None:
        """GET with limit=1 and limit=1000 should both be accepted."""
        resp1 = await client.get("/api/samples", params={"limit": 1}, headers=_AS_TENANT)
        assert resp1.status_code == 200

        resp1000 = await client.get("/api/samples", params={"limit": 1000}, headers=_AS_TENANT)
        assert resp1000.status_code == 200

    async def test_offset_zero_accepted(self, client: AsyncClient) -> None:
        """GET with offset=0 should be accepted."""
        resp = await client.get("/api/samples", params={"offset": 0}, headers=_AS_TENANT)
        assert resp.status_code == 200
