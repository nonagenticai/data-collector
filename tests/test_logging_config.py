"""Tests for logging_config.py -- configure_logging, get_logger, processors."""

import os
from unittest.mock import MagicMock, patch

import pytest

import logging_config as lc_module
from settings import Settings

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5451/data_collector",
)


@pytest.fixture(autouse=True)
def _reset_configured_flag():
    """Reset the _configured flag so each test can trigger configure_logging."""
    original = lc_module._configured
    lc_module._configured = False
    yield
    lc_module._configured = original


class TestAddTraceContext:
    def test_returns_event_dict_unchanged_when_otel_unavailable(self) -> None:
        """When OTEL is not available, event_dict is returned as-is."""
        with patch.object(lc_module, "OTEL_AVAILABLE", False):
            event_dict: dict = {"event": "test"}
            result = lc_module.add_trace_context(None, "info", event_dict)
            assert result == {"event": "test"}
            assert "trace_id" not in result

    def test_returns_event_dict_when_otel_available_but_no_recording_span(self) -> None:
        """When OTEL is available but no span is recording, no trace context is added."""
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        mock_trace.get_current_span.return_value = mock_span

        with (
            patch.object(lc_module, "OTEL_AVAILABLE", True),
            patch.object(lc_module, "trace", mock_trace),
        ):
            event_dict: dict = {"event": "test"}
            result = lc_module.add_trace_context(None, "info", event_dict)
            assert "trace_id" not in result

    def test_adds_trace_context_when_span_is_recording(self) -> None:
        """When a valid recording span exists, trace_id and span_id are injected."""
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_ctx = MagicMock()
        mock_ctx.is_valid = True
        mock_ctx.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
        mock_ctx.span_id = 0x1234567890ABCDEF
        mock_span.get_span_context.return_value = mock_ctx
        mock_trace.get_current_span.return_value = mock_span

        with (
            patch.object(lc_module, "OTEL_AVAILABLE", True),
            patch.object(lc_module, "trace", mock_trace),
        ):
            event_dict: dict = {"event": "test"}
            result = lc_module.add_trace_context(None, "info", event_dict)
            assert "trace_id" in result
            assert "span_id" in result
            assert len(result["trace_id"]) == 32
            assert len(result["span_id"]) == 16

    def test_no_trace_context_when_context_invalid(self) -> None:
        """When span context is invalid, no trace info is added."""
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_ctx = MagicMock()
        mock_ctx.is_valid = False
        mock_span.get_span_context.return_value = mock_ctx
        mock_trace.get_current_span.return_value = mock_span

        with (
            patch.object(lc_module, "OTEL_AVAILABLE", True),
            patch.object(lc_module, "trace", mock_trace),
        ):
            event_dict: dict = {"event": "test"}
            result = lc_module.add_trace_context(None, "info", event_dict)
            assert "trace_id" not in result


class TestAddServiceContext:
    def test_adds_service_and_environment(self) -> None:
        """Should add 'service' and 'environment' keys."""
        event_dict: dict = {"event": "test"}
        result = lc_module.add_service_context(None, "info", event_dict)
        assert "service" in result
        assert "environment" in result

    def test_uses_env_var_values(self) -> None:
        """Should use env vars for service name and environment."""
        with patch.dict(
            os.environ,
            {
                "OTEL_SERVICE_NAME": "my-service",
                "DATA_COLLECTOR_ENVIRONMENT": "staging",
            },
        ):
            event_dict: dict = {"event": "test"}
            result = lc_module.add_service_context(None, "info", event_dict)
            assert result["service"] == "my-service"
            assert result["environment"] == "staging"


class TestConfigureLogging:
    def test_configure_logging_json_format(self) -> None:
        """configure_logging with json format should not raise."""
        with patch.dict(os.environ, {"DATA_COLLECTOR_LOG_FORMAT": "json"}):
            lc_module.configure_logging()
            assert lc_module._configured is True

    def test_configure_logging_console_format(self) -> None:
        """configure_logging with console format should not raise."""
        with patch.dict(os.environ, {"DATA_COLLECTOR_LOG_FORMAT": "console"}):
            lc_module.configure_logging()
            assert lc_module._configured is True

    def test_configure_logging_with_level_override(self) -> None:
        """configure_logging should accept a level override."""
        import logging

        lc_module.configure_logging(level=logging.DEBUG)
        assert lc_module._configured is True

    def test_configure_logging_idempotent(self) -> None:
        """Second call to configure_logging should be a no-op."""
        lc_module.configure_logging()
        assert lc_module._configured is True
        # Calling again should return immediately
        lc_module.configure_logging()
        assert lc_module._configured is True

    def test_configure_logging_reads_log_level_env(self) -> None:
        """Should respect DATA_COLLECTOR_LOG_LEVEL env var."""
        with patch.dict(os.environ, {"DATA_COLLECTOR_LOG_LEVEL": "WARNING"}):
            lc_module.configure_logging()
            assert lc_module._configured is True

    def test_configure_logging_with_settings_object(self) -> None:
        """configure_logging should accept a Settings instance and use its values."""
        settings = Settings(
            database_url=TEST_DATABASE_URL,
            log_level="DEBUG",
            log_format="console",
        )
        lc_module.configure_logging(settings)
        assert lc_module._configured is True

    def test_configure_logging_with_settings_json_format(self) -> None:
        """configure_logging with Settings using json format should not raise."""
        settings = Settings(
            database_url=TEST_DATABASE_URL,
            log_level="INFO",
            log_format="json",
        )
        lc_module.configure_logging(settings)
        assert lc_module._configured is True


class TestGetLogger:
    def test_get_logger_returns_bound_logger(self) -> None:
        """get_logger should return a structlog BoundLogger."""
        logger = lc_module.get_logger("test_logger")
        assert logger is not None

    def test_get_logger_default_name(self) -> None:
        """get_logger with no args should use 'data_collector' name."""
        logger = lc_module.get_logger()
        assert logger is not None

    def test_get_logger_calls_configure_logging(self) -> None:
        """get_logger should trigger configure_logging if not yet configured."""
        assert lc_module._configured is False
        lc_module.get_logger("test")
        assert lc_module._configured is True
