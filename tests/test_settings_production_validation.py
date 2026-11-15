"""Settings refuse to start in production with empty URL placeholders."""

import pytest

from settings import Settings


def _mk(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql+psycopg://u:p@h:6432/dc",  # pragma: allowlist secret
        "training_trigger_url": "http://tdg:8152/api/triggers/fire",
        "otlp_traces_endpoint": "http://alloy:4318/v1/traces",
        "environment": "production",
    }
    base.update(overrides)
    return Settings.model_validate(base)


class TestProductionValidation:
    def test_complete_config_allows_init(self) -> None:
        settings = _mk()
        assert settings.environment == "production"

    @pytest.mark.parametrize(
        "empty_field",
        [
            "database_url",
            "training_trigger_url",
            "otlp_traces_endpoint",
        ],
    )
    def test_empty_required_raises(self, empty_field: str) -> None:
        with pytest.raises(RuntimeError, match=empty_field):
            _mk(**{empty_field: ""})

    def test_test_environment_allows_empty(self) -> None:
        Settings(environment="test")  # must not raise
