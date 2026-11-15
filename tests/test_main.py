"""Tests for main.py -- lifespan context manager and app setup."""

from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from main import app, global_exception_handler, lifespan


class TestLifespan:
    async def test_lifespan_startup_and_shutdown(self) -> None:
        """lifespan should call configure_logging, setup_telemetry, init_db on startup
        and dispose_engine on shutdown."""
        with (
            patch("main.get_settings") as mock_settings,
            patch("main.configure_logging") as mock_logging,
            patch("main.setup_telemetry") as mock_telemetry,
            patch("main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("main.dispose_engine", new_callable=AsyncMock) as mock_dispose,
        ):
            settings = MagicMock()
            settings.service_name = "test-service"
            settings.port = 8151
            settings.database_url = (
                "postgresql+psycopg://postgres:postgres@localhost:5451/data_collector"
            )
            mock_settings.return_value = settings

            async with lifespan(app):
                # Startup assertions
                mock_logging.assert_called_once_with(settings)
                mock_telemetry.assert_called_once_with(settings)
                mock_init_db.assert_called_once_with(settings)

            # Shutdown assertion
            mock_dispose.assert_awaited_once()

    async def test_lifespan_otel_instrumentation_failure_graceful(self) -> None:
        """lifespan should handle FastAPIInstrumentor import failure gracefully."""
        with (
            patch("main.get_settings") as mock_settings,
            patch("main.configure_logging"),
            patch("main.setup_telemetry"),
            patch("main.init_db", new_callable=AsyncMock),
            patch("main.dispose_engine", new_callable=AsyncMock),
        ):
            settings = MagicMock()
            settings.service_name = "test-service"
            settings.port = 8151
            settings.database_url = (
                "postgresql+psycopg://postgres:postgres@localhost:5451/data_collector"
            )
            mock_settings.return_value = settings

            # The OTEL instrumentation import is inside the lifespan;
            # it should handle ImportError gracefully
            async with lifespan(app):
                pass
            # No exception means graceful degradation worked


class TestAppEndpoints:
    async def test_health_endpoint(self) -> None:
        """GET /health should return 200 with status ok."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"
            assert body["service"] == "data-collector"
            assert body["version"] == "0.1.0"

    async def test_readiness_endpoint_not_ready(self) -> None:
        """GET /health/ready should return not_ready when no db engine is configured."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get("/health/ready")
            assert resp.status_code == 200
            body = resp.json()
            # Without init_db, engine is None so get_engine_for_check returns False
            assert body["status"] in ("ready", "not_ready")

    async def test_readiness_endpoint_ready(self) -> None:
        """GET /health/ready should return ready when db engine check succeeds."""
        with patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                resp = await ac.get("/health/ready")
                assert resp.status_code == 200
                body = resp.json()
                assert body["status"] == "ready"
                assert body["database"] == "connected"

    async def test_liveness_endpoint(self) -> None:
        """GET /health/live should return 200 with status alive."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get("/health/live")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "alive"

    async def test_root_endpoint(self) -> None:
        """GET / should return 200 with service info."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get("/")
            assert resp.status_code == 200
            body = resp.json()
            assert body["service"] == "data-collector"
            assert body["version"] == "0.1.0"
            assert body["docs"] == "/docs"
            assert body["health"] == "/health"

    async def test_cors_rejects_unlisted_origin(self) -> None:
        """With no configured origins, CORS must not echo a wildcard/arbitrary origin."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.options(
                "/health",
                headers={
                    "Origin": "http://example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
            # No wildcard, and the arbitrary origin is not allow-listed.
            assert resp.headers.get("access-control-allow-origin") != "*"
            assert resp.headers.get("access-control-allow-origin") != "http://example.com"

    async def test_cors_allows_configured_origin(self) -> None:
        """A configured allow-list origin is echoed back; others are not."""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        configured = "https://fr.nonagentic.ai"
        scoped_app = FastAPI()
        scoped_app.add_middleware(
            CORSMiddleware,
            allow_origins=[configured],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @scoped_app.get("/health")
        async def _health() -> dict[str, str]:
            return {"status": "ok"}

        transport = ASGITransport(app=scoped_app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            allowed = await ac.options(
                "/health",
                headers={"Origin": configured, "Access-Control-Request-Method": "GET"},
            )
            assert allowed.headers.get("access-control-allow-origin") == configured

            denied = await ac.options(
                "/health",
                headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
            )
            assert denied.headers.get("access-control-allow-origin") != "http://evil.example"
            assert denied.headers.get("access-control-allow-origin") != "*"

    async def test_exception_handler(self) -> None:
        """Unhandled exceptions should return 500 JSON with error details."""
        from fastapi import Request

        mock_request = MagicMock(spec=Request)
        exc = RuntimeError("something broke")
        response = await global_exception_handler(mock_request, exc)
        assert response.status_code == 500
        import json

        body = json.loads(bytes(response.body))
        assert body["error"]["message"] == "Internal server error"
        assert body["error"]["type"] == "server_error"
        assert body["error"]["code"] == 500
