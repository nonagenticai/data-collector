"""Tests for the JSONL export endpoint."""

import json
from collections.abc import Callable
from typing import Any

from httpx import AsyncClient


class TestExport:
    async def test_export_empty(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/samples/export",
            params={"schema_id": "ks_EntitySpotterAgent"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        assert resp.text == ""

    async def test_export_returns_jsonl(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        # Create several samples (each with unique content to avoid dedup)
        for _ in range(3):
            await client.post("/api/samples", json=make_sample())

        resp = await client.get(
            "/api/samples/export",
            params={"schema_id": "ks_EntitySpotterAgent"},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert len(lines) == 3

        # Each line must be valid JSON
        for line in lines:
            obj = json.loads(line)
            assert obj["schema_id"] == "ks_EntitySpotterAgent"
            assert "messages" in obj
            assert "response_content" in obj
            assert "id" in obj

    async def test_export_only_validated(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample())
        await client.post("/api/samples", json=make_sample(validation_passed=False))

        resp = await client.get(
            "/api/samples/export",
            params={
                "schema_id": "ks_EntitySpotterAgent",
                "only_validated": True,
            },
        )
        lines = [line for line in resp.text.strip().split("\n") if line]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["validation_passed"] is True

    async def test_export_only_unused(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        create_resp = await client.post("/api/samples", json=sample_payload)
        sample_id = create_resp.json()["id"]

        # Mark as used
        await client.patch(
            "/api/samples/mark-used",
            json={"sample_ids": [sample_id], "training_batch_id": "batch-x"},
        )

        resp = await client.get(
            "/api/samples/export",
            params={
                "schema_id": "ks_EntitySpotterAgent",
                "only_unused": True,
            },
        )
        lines = [line for line in resp.text.strip().split("\n") if line]
        assert len(lines) == 0

    async def test_export_with_limit(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        for _ in range(5):
            await client.post("/api/samples", json=make_sample())

        resp = await client.get(
            "/api/samples/export",
            params={
                "schema_id": "ks_EntitySpotterAgent",
                "limit": 2,
            },
        )
        lines = [line for line in resp.text.strip().split("\n") if line]
        assert len(lines) == 2

    async def test_export_filter_by_provider(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample())
        await client.post("/api/samples", json=make_sample(provider="claude"))

        resp = await client.get(
            "/api/samples/export",
            params={
                "schema_id": "ks_EntitySpotterAgent",
                "provider": "claude",
                "only_validated": False,
                "only_unused": False,
            },
        )
        lines = [line for line in resp.text.strip().split("\n") if line]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["provider"] == "claude"

    async def test_export_filter_by_fallback_used(
        self, client: AsyncClient, make_sample: Callable[..., dict[str, Any]]
    ) -> None:
        await client.post("/api/samples", json=make_sample(fallback_used=False))
        await client.post("/api/samples", json=make_sample(fallback_used=True))

        # Only teacher samples (fallback_used=True)
        resp = await client.get(
            "/api/samples/export",
            params={
                "schema_id": "ks_EntitySpotterAgent",
                "fallback_used": True,
                "only_validated": False,
                "only_unused": False,
            },
        )
        lines = [line for line in resp.text.strip().split("\n") if line]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["fallback_used"] is True

        # Only non-teacher samples (fallback_used=False)
        resp2 = await client.get(
            "/api/samples/export",
            params={
                "schema_id": "ks_EntitySpotterAgent",
                "fallback_used": False,
                "only_validated": False,
                "only_unused": False,
            },
        )
        lines2 = [line for line in resp2.text.strip().split("\n") if line]
        assert len(lines2) == 1
        obj2 = json.loads(lines2[0])
        assert obj2["fallback_used"] is False

    async def test_export_content_disposition_header(self, client: AsyncClient) -> None:
        resp = await client.get(
            "/api/samples/export",
            params={"schema_id": "ks_EntitySpotterAgent"},
        )
        assert "content-disposition" in resp.headers
        assert "ks_EntitySpotterAgent_samples.jsonl" in resp.headers["content-disposition"]
