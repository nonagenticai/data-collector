"""Application settings using Pydantic BaseSettings with env_prefix."""

from functools import lru_cache
from typing import Any
from uuid import UUID

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Data Collector service configuration.

    All settings can be overridden via environment variables prefixed with
    ``DATA_COLLECTOR_``.  For example, ``DATA_COLLECTOR_PORT=9000`` overrides
    the default port.

    Environment-specific values (database DSN, upstream service URLs, OTLP
    endpoint) carry empty-string placeholders here — real values come from
    the Deployment env block in gitops-production. In production, startup
    fails fast if any required field is empty.
    """

    model_config = SettingsConfigDict(
        env_prefix="DATA_COLLECTOR_",
        case_sensitive=False,
    )

    service_name: str = "data-collector"
    port: int = Field(default=8151, description="HTTP listen port")
    cors_allow_origins: str = Field(
        default="",
        description=(
            "Comma-separated list of browser origins allowed by CORS. Empty "
            "means no browser origins are allowed (non-browser callers ignore "
            "CORS). The only browser consumer is fr; set its origin here via "
            "the Deployment env block if browser access is needed."
        ),
    )
    database_url: str = Field(
        default="",
        description="Async SQLAlchemy database URL (psycopg3 driver)",
    )
    owner_database_url: str = Field(
        default="",
        description=(
            "Owner (table-owner / RLS-bypass) DSN. Empty until the enforce "
            "cutover: when set (and != database_url) the runtime connects as a "
            "restricted non-owner role via database_url while migrations and the "
            "cross-tenant flywheel reads/counts run on this owner connection. "
            "Unset → owner falls back to database_url (dormant / local / tests), "
            "so behavior is unchanged. See CLAUDE.md enforce-cutover."
        ),
    )
    otlp_traces_endpoint: str = Field(
        default="",
        description="OTLP HTTP endpoint for trace export",
    )
    environment: str = Field(default="production", description="Deployment environment tag")
    log_level: str = Field(default="INFO", description="Root log level")
    log_format: str = Field(
        default="json",
        description="Log format: 'json' for production, 'console' for local dev",
    )
    training_threshold: int = Field(
        default=25,
        ge=1,
        description="Validated unused samples needed to trigger training for a schema",
    )
    training_trigger_url: str = Field(
        default="",
        description="URL to POST training trigger when threshold is crossed",
    )
    training_trigger_enabled: bool = Field(
        default=True,
        description="Enable automatic training trigger notifications",
    )
    global_training_threshold: int = Field(
        default=1000,
        ge=1,
        description=(
            "Aggregated validated/unused samples (across ALL schemas) needed "
            "to trigger a system-wide single-LoRA retrain. Fires exactly once "
            "when the count crosses the threshold; resets when training "
            "consumes the samples (used_in_training=true)."
        ),
    )
    global_training_trigger_url: str = Field(
        default="",
        description=(
            "URL to POST the global-training trigger when the aggregated "
            "threshold is crossed. Empty disables the global trigger."
        ),
    )
    global_training_trigger_enabled: bool = Field(
        default=False,
        description=(
            "Enable system-wide single-LoRA retrain notifications. Disabled "
            "by default; enable in the gitops Deployment env block once the "
            "single-LoRA cutover is in flight."
        ),
    )

    delegating_tenants: str = Field(
        default="",
        description=(
            "Comma-separated tenant UUIDs whose service accounts may act on "
            "behalf of another tenant by sending ``X-Effective-Tenant-Id``. "
            "Mirrors w2v_common's ``AUTH_DELEGATING_SUBJECTS`` delegation, but "
            "keyed on the mesh-injected ``X-Tenant-Id`` rather than an "
            "introspected subject: this service does not introspect Bearer "
            "tokens (license-authz resolves the tenant at the mesh edge), so "
            "the caller's tenant is the only discriminator it can trust. "
            "Empty ⇒ delegation is a no-op and every caller is stamped with "
            "its own tenant, exactly as before."
        ),
    )

    _REQUIRED_IN_PRODUCTION: tuple[str, ...] = (
        "database_url",
        "training_trigger_url",
        "otlp_traces_endpoint",
    )

    @property
    def cors_allow_origins_list(self) -> list[str]:
        """Parse ``cors_allow_origins`` into a clean list of origins."""
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def delegating_tenants_set(self) -> frozenset[str]:
        """Parse ``delegating_tenants`` into a set of normalised tenant UUIDs.

        Values are normalised through ``UUID`` so a configured tenant matches
        regardless of case or hyphenation; anything that does not parse is
        dropped rather than silently widening the allowlist to a literal.
        """
        parsed: set[str] = set()
        for raw in self.delegating_tenants.split(","):
            candidate = raw.strip()
            if not candidate:
                continue
            try:
                parsed.add(str(UUID(candidate)))
            except ValueError:
                continue
        return frozenset(parsed)

    def model_post_init(self, __context: Any) -> None:
        if self.environment != "production":
            return
        missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"data-collector settings missing required fields in production: {missing}. "
                "Set them via env vars on the Deployment (see gitops-production)."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
