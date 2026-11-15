"""Edge-case and coverage-gap tests for the data-collector service.

Targets uncovered branches and paths discovered via coverage analysis:
- telemetry.py (setup_telemetry, _test_connection)
- main.py (exception handler HTTPException re-raise, OTEL instrumentation)
- settings.py (get_settings caching)
- model validation edge cases
- sample_service edge cases
"""

import json
import os
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from models.domain import (
    ExportParams,
    InferenceSample,
    LivenessResponse,
    MarkUsedRequest,
    MarkUsedResponse,
    PurgeResponse,
    SampleCountResponse,
    SampleCreate,
    SampleFilter,
    SampleStats,
    SchemaListResponse,
)
from services import sample_service

# ---------------------------------------------------------------------------
# telemetry.py coverage
# ---------------------------------------------------------------------------


class TestTelemetrySetup:
    """Tests for telemetry.py -- setup_telemetry and _test_connection."""

    def test_test_connection_unreachable(self) -> None:
        """_test_connection should return False when the connect times out.

        Original test called a real RFC-5737 TEST-NET-1 host (192.0.2.1) at
        port 9999 with a 100ms timeout. On Tailnet-routed ARC runners the
        SYN never reaches the upstream firewall in time and the kernel
        returns False for unrelated reasons (ENETUNREACH ≠ TimeoutError).
        Mocking the socket keeps the behaviour-under-test ("TimeoutError
        ⇒ False") and removes the environment dependency.
        """
        import telemetry

        with patch("telemetry.socket.create_connection", side_effect=TimeoutError("timed out")):
            result = telemetry._test_connection("192.0.2.1", 9999, timeout=0.1)
            assert result is False

    def test_test_connection_connection_refused(self) -> None:
        """_test_connection should return False when connection refused."""
        import telemetry

        # Port 1 is typically not listening
        result = telemetry._test_connection("127.0.0.1", 1, timeout=0.5)
        assert result is False

    def test_test_connection_oserror(self) -> None:
        """_test_connection should return False on OSError."""
        import telemetry

        with patch("telemetry.socket.create_connection", side_effect=OSError("mock error")):
            result = telemetry._test_connection("localhost", 4318, timeout=0.1)
            assert result is False

    def test_test_connection_success(self) -> None:
        """_test_connection should return True when connection succeeds."""
        import telemetry

        mock_sock = MagicMock()
        with patch("telemetry.socket.create_connection", return_value=mock_sock):
            result = telemetry._test_connection("localhost", 4318)
            assert result is True
            mock_sock.close.assert_called_once()

    def test_setup_telemetry_disabled_via_env(self) -> None:
        """setup_telemetry should return False when OTEL_SDK_DISABLED=true."""
        import telemetry

        original = telemetry._initialized
        telemetry._initialized = False
        try:
            settings = MagicMock()
            settings.service_name = "test-service"
            settings.otlp_traces_endpoint = "http://localhost:4318/v1/traces"
            with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
                result = telemetry.setup_telemetry(settings)
                assert result is False
        finally:
            telemetry._initialized = original

    def test_setup_telemetry_already_initialized(self) -> None:
        """setup_telemetry should return True immediately when already initialized."""
        import telemetry

        original = telemetry._initialized
        telemetry._initialized = True
        try:
            settings = MagicMock()
            result = telemetry.setup_telemetry(settings)
            assert result is True
        finally:
            telemetry._initialized = original

    def test_setup_telemetry_import_error(self) -> None:
        """setup_telemetry should return False when OTEL packages not available."""
        import telemetry

        original = telemetry._initialized
        telemetry._initialized = False
        try:
            settings = MagicMock()
            settings.service_name = "test-service"
            settings.otlp_traces_endpoint = "http://localhost:4318/v1/traces"

            # Temporarily remove otel modules to cause ImportError
            with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
                with patch("builtins.__import__", side_effect=ImportError("no otel")):
                    result = telemetry.setup_telemetry(settings)
                    assert result is False
        finally:
            telemetry._initialized = original

    def test_setup_telemetry_endpoint_not_reachable(self) -> None:
        """setup_telemetry should still set provider even when endpoint is unreachable."""
        import telemetry

        original = telemetry._initialized
        telemetry._initialized = False
        try:
            settings = MagicMock()
            settings.service_name = "test-service"
            settings.otlp_traces_endpoint = "http://192.0.2.1:4318/v1/traces"

            with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}):
                result = telemetry.setup_telemetry(settings)
                # It should succeed (provider is set) even if endpoint is unreachable
                assert result is True
        finally:
            telemetry._initialized = original

    def test_setup_telemetry_endpoint_reachable(self) -> None:
        """setup_telemetry should configure exporter when endpoint is reachable."""
        import telemetry

        original = telemetry._initialized
        telemetry._initialized = False
        try:
            settings = MagicMock()
            settings.service_name = "test-service"
            settings.otlp_traces_endpoint = "http://localhost:4318/v1/traces"

            with (
                patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}),
                patch.object(telemetry, "_test_connection", return_value=True),
            ):
                result = telemetry.setup_telemetry(settings)
                assert result is True
        finally:
            telemetry._initialized = original

    def test_setup_telemetry_exporter_config_exception(self) -> None:
        """setup_telemetry should handle exceptions during exporter configuration."""
        import telemetry

        original = telemetry._initialized
        telemetry._initialized = False
        try:
            settings = MagicMock()
            settings.service_name = "test-service"
            settings.otlp_traces_endpoint = "http://localhost:4318/v1/traces"

            with (
                patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}),
                patch.object(telemetry, "_test_connection", return_value=True),
                patch(
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
                    side_effect=OSError("exporter config failed"),
                ),
            ):
                result = telemetry.setup_telemetry(settings)
                assert result is True
        finally:
            telemetry._initialized = original

    def test_setup_telemetry_httpx_instrumentation_failure(self) -> None:
        """setup_telemetry should handle HTTPX instrumentation failure gracefully."""
        import telemetry

        original = telemetry._initialized
        telemetry._initialized = False
        try:
            settings = MagicMock()
            settings.service_name = "test-service"
            settings.otlp_traces_endpoint = "http://192.0.2.1:4318/v1/traces"

            with (
                patch.dict(os.environ, {"OTEL_SDK_DISABLED": "false"}),
                patch(
                    "opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor",
                    side_effect=ImportError("instrumentation failed"),
                ),
            ):
                result = telemetry.setup_telemetry(settings)
                assert result is True
        finally:
            telemetry._initialized = original


# ---------------------------------------------------------------------------
# main.py exception handler coverage
# ---------------------------------------------------------------------------


class TestExceptionHandlerEdgeCases:
    """Edge cases for the global exception handler."""

    async def test_exception_handler_reraises_http_exception(self) -> None:
        """global_exception_handler should re-raise HTTPException."""
        from fastapi import HTTPException, Request

        from main import global_exception_handler

        mock_request = MagicMock(spec=Request)
        http_exc = HTTPException(status_code=404, detail="Not found")
        with pytest.raises(HTTPException) as exc_info:
            await global_exception_handler(mock_request, http_exc)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not found"

    async def test_exception_handler_returns_500_for_generic_exception(self) -> None:
        """global_exception_handler should return 500 JSON for non-HTTP exceptions."""
        from fastapi import Request

        from main import global_exception_handler

        mock_request = MagicMock(spec=Request)
        response = await global_exception_handler(mock_request, ValueError("bad value"))
        assert response.status_code == 500
        body = json.loads(bytes(response.body))
        assert body["error"]["message"] == "Internal server error"
        assert body["error"]["type"] == "server_error"
        assert body["error"]["code"] == 500


# ---------------------------------------------------------------------------
# settings.py coverage
# ---------------------------------------------------------------------------


class TestSettingsCaching:
    """Tests for settings caching behavior."""

    def test_get_settings_returns_settings_with_correct_defaults(self) -> None:
        """get_settings should return a Settings instance with expected default values."""
        from settings import Settings, get_settings

        get_settings.cache_clear()
        with patch.dict(
            os.environ,
            {
                "DATA_COLLECTOR_DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5451/data_collector",
                "DATA_COLLECTOR_TRAINING_TRIGGER_URL": "http://tdg.test:8152/api/triggers/fire",
                "DATA_COLLECTOR_OTLP_TRACES_ENDPOINT": "http://alloy.test:4318/v1/traces",
                "DATA_COLLECTOR_ENVIRONMENT": "production",
            },
        ):
            settings = get_settings()
            assert isinstance(settings, Settings)
            assert settings.service_name == "data-collector"
            assert settings.port == 8151
            assert settings.environment == "production"
            assert settings.log_level == "INFO"
            assert settings.log_format == "json"
        get_settings.cache_clear()

    def test_get_settings_caches_result(self) -> None:
        """get_settings should return the same instance on repeated calls."""
        from settings import get_settings

        get_settings.cache_clear()
        with patch.dict(
            os.environ,
            {
                "DATA_COLLECTOR_DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5451/data_collector"
            },
        ):
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Model domain edge cases
# ---------------------------------------------------------------------------


class TestDomainModelEdgeCases:
    """Edge cases for Pydantic domain models."""

    def test_inference_sample_defaults(self) -> None:
        """InferenceSample should have sensible defaults for optional fields."""
        sample = InferenceSample(
            schema_id="ks_Test",
            job_id="job-1",
            inference_id="inf-1",
            messages=[{"role": "user", "content": "hello"}],
            response_content="{}",
            provider="Qwen3-4B-AWQ",
            validation_passed=True,
            latency_ms=50,
            failover_chain_id="chain-1",
        )
        assert sample.fallback_used is False
        assert sample.fallback_reason is None
        assert sample.used_in_training is False
        assert sample.training_batch_id is None
        assert isinstance(sample.id, UUID)
        assert isinstance(sample.created_at, datetime)

    def test_liveness_response_default(self) -> None:
        """LivenessResponse should default to status=alive."""
        resp = LivenessResponse()
        assert resp.status == "alive"

    def test_liveness_response_custom(self) -> None:
        """LivenessResponse should accept custom status."""
        resp = LivenessResponse(status="custom")
        assert resp.status == "custom"

    def test_sample_filter_defaults(self) -> None:
        """SampleFilter should have correct default values."""
        f = SampleFilter()
        assert f.schema_id is None
        assert f.provider is None
        assert f.validation_passed is None
        assert f.used_in_training is None
        assert f.start_date is None
        assert f.end_date is None
        assert f.limit == 100
        assert f.offset == 0

    def test_sample_filter_limit_boundaries(self) -> None:
        """SampleFilter should accept limit=1 and limit=1000."""
        f1 = SampleFilter(limit=1)
        assert f1.limit == 1
        f2 = SampleFilter(limit=1000)
        assert f2.limit == 1000

    def test_export_params_defaults(self) -> None:
        """ExportParams should have correct defaults."""
        p = ExportParams(schema_id="test")
        assert p.only_validated is True
        assert p.only_unused is True
        assert p.provider is None
        assert p.fallback_used is None
        assert p.limit is None

    def test_mark_used_request_validation(self) -> None:
        """MarkUsedRequest should validate UUID list and store training_batch_id."""
        ids = [uuid4(), uuid4()]
        req = MarkUsedRequest(
            sample_ids=ids,
            training_batch_id="batch-123",
        )
        assert len(req.sample_ids) == 2
        assert req.sample_ids == ids
        assert req.training_batch_id == "batch-123"

    def test_mark_used_response_model(self) -> None:
        """MarkUsedResponse should store fields correctly."""
        resp = MarkUsedResponse(updated_count=5, training_batch_id="batch-x")
        assert resp.updated_count == 5
        assert resp.training_batch_id == "batch-x"

    def test_sample_count_response_model(self) -> None:
        """SampleCountResponse should store fields correctly."""
        resp = SampleCountResponse(count=42, schema_id="ks_Test")
        assert resp.count == 42
        assert resp.schema_id == "ks_Test"

    def test_purge_response_model(self) -> None:
        """PurgeResponse should store purged_count and filters_applied."""
        resp = PurgeResponse(
            purged_count=3,
            filters_applied={"schema_id": "ks_Test", "provider": "claude"},
        )
        assert resp.purged_count == 3
        assert resp.filters_applied["schema_id"] == "ks_Test"
        assert resp.filters_applied["provider"] == "claude"

    def test_schema_list_response_model(self) -> None:
        """SchemaListResponse should store fields correctly."""
        resp = SchemaListResponse(schema_ids=["ks_A", "ks_B"])
        assert resp.schema_ids == ["ks_A", "ks_B"]

    def test_sample_stats_model_all_fields(self) -> None:
        """SampleStats should store all fields with correct types."""
        stats = SampleStats(
            schema_id="ks_Test",
            total_samples=100,
            validated_samples=80,
            failed_samples=20,
            used_in_training=50,
            available_for_training=30,
            provider_breakdown={"Qwen3-4B-AWQ": 60, "claude": 40},
            avg_latency_ms=150.5,
            fallback_rate=0.1,
        )
        assert stats.total_samples == 100
        assert stats.validated_samples == 80
        assert stats.failed_samples == 20
        assert stats.used_in_training == 50
        assert stats.available_for_training == 30
        assert stats.provider_breakdown["claude"] == 40
        assert stats.avg_latency_ms == 150.5
        assert stats.fallback_rate == 0.1
        # Invariant check
        assert stats.total_samples == stats.validated_samples + stats.failed_samples

    def test_inference_sample_model_dump_json_serializable(self) -> None:
        """InferenceSample model_dump should produce a JSON-serializable dict."""
        sample = InferenceSample(
            schema_id="ks_Test",
            job_id="job-1",
            inference_id="inf-1",
            messages=[{"role": "user", "content": "hello"}],
            response_content="{}",
            provider="Qwen3-4B-AWQ",
            validation_passed=True,
            latency_ms=50,
            failover_chain_id="chain-1",
        )
        dumped = sample.model_dump(mode="json")
        assert isinstance(dumped["id"], str)
        assert isinstance(dumped["created_at"], str)
        # Should be valid JSON round-trip
        json_str = json.dumps(dumped)
        reparsed = json.loads(json_str)
        assert reparsed["schema_id"] == "ks_Test"

    def test_sample_create_latency_ge_zero(self) -> None:
        """SampleCreate should reject negative latency_ms."""
        with pytest.raises(Exception):
            SampleCreate(
                schema_id="ks_Test",
                job_id="job-1",
                inference_id="inf-1",
                messages=[{"role": "user", "content": "hi"}],
                response_content="{}",
                provider="Qwen3-4B-AWQ",
                validation_passed=True,
                latency_ms=-1,
                failover_chain_id="chain-1",
            )


# ---------------------------------------------------------------------------
# sample_service edge cases -- using real DB
# ---------------------------------------------------------------------------


def _make_payload(**overrides: Any) -> SampleCreate:
    """Helper to build a SampleCreate with unique content per call."""
    uid = uuid4().hex[:12]
    defaults: dict[str, Any] = {
        "schema_id": "ks_EdgeCase",
        "job_id": "job-edge-001",
        "inference_id": f"inf-{uuid4().hex[:12]}",
        "messages": [{"role": "user", "content": "test"}],
        "response_content": f'{{"result": "ok", "uid": "{uid}"}}',
        "provider": "Qwen3-4B-AWQ",
        "validation_passed": True,
        "fallback_used": False,
        "fallback_reason": None,
        "latency_ms": 50,
        "failover_chain_id": "chain-edge-001",
    }
    defaults.update(overrides)
    return SampleCreate(**defaults)


class TestSampleServiceEdgeCases:
    """Edge cases in sample_service using real DB sessions."""

    async def test_list_samples_empty_db(self, session: AsyncSession) -> None:
        """list_samples on empty DB should return empty list."""
        filters = SampleFilter()
        results = await sample_service.list_samples(session, filters)
        assert results == []

    async def test_list_samples_with_all_filters(self, session: AsyncSession) -> None:
        """list_samples with all filters set should return matching samples."""
        created = await sample_service.create_sample(
            session,
            _make_payload(
                schema_id="ks_Full",
                provider="claude",
                validation_passed=True,
            ),
        )
        # Non-matching sample
        await sample_service.create_sample(
            session,
            _make_payload(schema_id="ks_Other", provider="Qwen3-4B-AWQ"),
        )

        filters = SampleFilter(
            schema_id="ks_Full",
            provider="claude",
            validation_passed=True,
            used_in_training=False,
            limit=10,
            offset=0,
        )
        results = await sample_service.list_samples(session, filters)
        assert len(results) == 1
        assert results[0].id == created.id
        assert results[0].schema_id == "ks_Full"

    async def test_list_samples_pagination_offset_beyond(self, session: AsyncSession) -> None:
        """list_samples with offset beyond available rows returns empty."""
        await sample_service.create_sample(session, _make_payload())
        filters = SampleFilter(offset=100)
        results = await sample_service.list_samples(session, filters)
        assert results == []

    async def test_get_count_nonexistent_schema(self, session: AsyncSession) -> None:
        """get_count for a non-existent schema should return 0."""
        result = await sample_service.get_count(session, "ks_DoesNotExist")
        assert result.count == 0
        assert result.schema_id == "ks_DoesNotExist"

    async def test_export_no_matching_samples(self, session: AsyncSession) -> None:
        """export_samples_jsonl with no matches yields nothing."""
        params = ExportParams(schema_id="ks_Missing")
        lines = []
        async for line in sample_service.export_samples_jsonl(session, params):
            lines.append(line)
        assert lines == []

    async def test_export_with_limit_zero(self, session: AsyncSession) -> None:
        """export_samples_jsonl with limit=0 should return no lines (SQLAlchemy LIMIT 0)."""
        await sample_service.create_sample(session, _make_payload())
        params = ExportParams(schema_id="ks_EdgeCase", limit=0)
        lines = []
        async for line in sample_service.export_samples_jsonl(session, params):
            lines.append(line)
        assert lines == []

    async def test_export_only_validated_false_includes_failed(self, session: AsyncSession) -> None:
        """export with only_validated=False should include failed samples."""
        created = await sample_service.create_sample(
            session, _make_payload(validation_passed=False)
        )
        params = ExportParams(
            schema_id="ks_EdgeCase",
            only_validated=False,
            only_unused=False,
        )
        lines = []
        async for line in sample_service.export_samples_jsonl(session, params):
            lines.append(line)
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["validation_passed"] is False
        assert parsed["id"] == str(created.id)

    async def test_export_only_unused_false_includes_used(self, session: AsyncSession) -> None:
        """export with only_unused=False should include used samples."""
        sample = await sample_service.create_sample(session, _make_payload())
        req = MarkUsedRequest(sample_ids=[sample.id], training_batch_id="batch-edge")
        await sample_service.mark_used(session, req)

        params = ExportParams(
            schema_id="ks_EdgeCase",
            only_validated=False,
            only_unused=False,
        )
        lines = []
        async for line in sample_service.export_samples_jsonl(session, params):
            lines.append(line)
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["used_in_training"] is True
        assert parsed["training_batch_id"] == "batch-edge"

    async def test_get_stats_avg_latency_none_handling(self, session: AsyncSession) -> None:
        """get_stats should handle no samples gracefully (empty result)."""
        stats = await sample_service.get_stats(session, schema_id="ks_NoSamples")
        assert stats == []

    async def test_mark_used_empty_list(self, session: AsyncSession) -> None:
        """mark_used with empty sample_ids list should return 0."""
        req = MarkUsedRequest(sample_ids=[], training_batch_id="batch-empty")
        updated = await sample_service.mark_used(session, req)
        assert updated == 0

    async def test_create_sample_with_fallback_fields(self, session: AsyncSession) -> None:
        """create_sample with fallback_used=True should persist fallback details."""
        sample = await sample_service.create_sample(
            session,
            _make_payload(
                fallback_used=True,
                fallback_reason="primary provider timeout",
                provider="claude",
            ),
        )
        assert sample.fallback_used is True
        assert sample.fallback_reason == "primary provider timeout"
        assert sample.provider == "claude"

    async def test_get_count_both_validated_and_unused(self, session: AsyncSession) -> None:
        """get_count with both only_validated and only_unused should AND the filters."""
        await sample_service.create_sample(session, _make_payload(validation_passed=True))
        await sample_service.create_sample(session, _make_payload(validation_passed=False))
        result = await sample_service.get_count(
            session, "ks_EdgeCase", only_validated=True, only_unused=True
        )
        assert result.count == 1

    async def test_list_samples_filter_provider_no_match(self, session: AsyncSession) -> None:
        """list_samples with provider filter that matches nothing returns empty."""
        await sample_service.create_sample(session, _make_payload(provider="Qwen3-4B-AWQ"))
        filters = SampleFilter(provider="nonexistent_provider")
        results = await sample_service.list_samples(session, filters)
        assert results == []


# ---------------------------------------------------------------------------
# API endpoint edge cases via client
# ---------------------------------------------------------------------------


class TestApiEdgeCases:
    """API-level edge cases testing uncovered router branches."""

    async def test_export_requires_schema_id(self, client: AsyncClient) -> None:
        """GET /api/samples/export without schema_id should return 422."""
        resp = await client.get("/api/samples/export")
        assert resp.status_code == 422

    async def test_export_with_all_params(
        self, client: AsyncClient, sample_payload: dict[str, Any]
    ) -> None:
        """GET /api/samples/export with all params should return matching JSONL."""
        await client.post("/api/samples", json=sample_payload)
        resp = await client.get(
            "/api/samples/export",
            params={
                "schema_id": "ks_EntitySpotterAgent",
                "only_validated": False,
                "only_unused": False,
                "provider": "Qwen3-4B-AWQ",
                "limit": 10,
            },
        )
        assert resp.status_code == 200
        lines = [ln for ln in resp.text.strip().split("\n") if ln]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["schema_id"] == "ks_EntitySpotterAgent"
        assert parsed["provider"] == "Qwen3-4B-AWQ"

    async def test_mark_used_empty_ids_via_api(self, client: AsyncClient) -> None:
        """PATCH /api/samples/mark-used with empty list should return 0 updated."""
        resp = await client.patch(
            "/api/samples/mark-used",
            json={"sample_ids": [], "training_batch_id": "batch-empty"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["updated_count"] == 0
        assert body["training_batch_id"] == "batch-empty"

    async def test_stats_with_schema_filter_no_match(self, client: AsyncClient) -> None:
        """GET /api/samples/stats with non-existent schema_id returns empty list."""
        resp = await client.get(
            "/api/samples/stats",
            params={"schema_id": "ks_NonExistent"},
            # /stats is tenant-scoped and fails closed when untenanted (src/read_scope.py).
            headers={"X-Tenant-Id": "019de842-c92a-7e37-8fdd-eb0d8c5396f9"},
        )
        assert resp.status_code == 200
        assert resp.json() == []
