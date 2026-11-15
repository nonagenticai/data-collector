# Data Collector

Receives and persists every inference interaction for training data generation. Port 8151.

## What Data Collector Does

When an inference router processes a request, it forwards a copy of the prompt, response, and
metadata to this service. Data Collector persists each sample with its schema ID, provider name,
validation status, latency, and failover chain ID so a training data generator can later query
for fresh examples to build fine-tuning datasets. Every incoming sample is deduplicated by a
SHA-256 content hash computed from the schema ID, messages, and response content, so duplicate
inference replays never inflate the dataset.

Beyond raw collection, the service tracks which samples have been consumed by training. The
export endpoint streams results as newline-delimited JSON so arbitrarily large datasets can be
transferred without loading everything into memory. After a training batch completes, the
mark-used endpoint flags those samples with a batch ID so they are excluded from future runs. A
purge endpoint allows bulk deletion of old or low-quality samples by date, schema, provider, or
validation status.

The service also provides aggregation statistics grouped by schema ID: total samples, validated
vs. failed counts, training utilization, provider breakdown, average latency, and fallback rate.
A separate schema discovery endpoint lists all distinct schema identifiers so upstream tooling
can enumerate what function-calling schemas have collected data.

Data Collector is backed by PostgreSQL. Samples are append-only under normal operation.

## Features

- **Content-hash deduplication** — SHA-256 over schema ID, messages, and response content, so a
  duplicate inference replay collapses to the existing row
- **Streaming JSONL export** — SQLAlchemy server-side cursors stream arbitrarily large datasets
- **Schema-scoped queries** — filter by schema ID, provider, validation status, date range, and
  training usage
- **Training batch tracking** — mark consumed samples with a batch ID
- **Aggregation statistics** — per-schema totals, validation rates, provider breakdown, average
  latency, and fallback rate
- **Bulk purge** — hard-delete matching samples by date, schema, provider, or validation status
- **Schema discovery** — list every distinct schema identifier
- **Per-tenant isolation** — an optional nullable `tenant_id` with PostgreSQL row-level security
  (`ENABLE`, never `FORCE`); the runtime role binds the `app.current_tenant` GUC per transaction
- **OpenTelemetry tracing** — instruments FastAPI and HTTPX with graceful degradation
- **Health probes** — liveness, readiness (with database connectivity), and root info endpoints

## Quick Start

With Docker Compose (service + PostgreSQL 18):

```bash
docker compose up --build
curl -s http://localhost:8151/health/ready
```

With `uv` and a PostgreSQL you already have:

```bash
uv sync
DATA_COLLECTOR_ENVIRONMENT=development \
DATA_COLLECTOR_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5451/data_collector \
  uv run uvicorn main:app --host 0.0.0.0 --port 8151 --app-dir src
```

Migrations run automatically to `head` on startup.

## API Endpoints

### Samples `/api/samples`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/` | Ingest a request+response pair from inference |
| GET | `/` | List samples with schema/date/provider/training filters |
| GET | `/count` | Count samples matching filters |
| GET | `/export` | Export samples as streaming JSONL for training pipelines |
| GET | `/stats` | Aggregation statistics grouped by schema_id |
| PATCH | `/mark-used` | Mark a batch of samples as used in a training batch |
| DELETE | `/purge` | Purge samples matching filters (at least one required) |

### Schemas `/api/schemas`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | List all distinct schema identifiers |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Liveness probe |
| GET | `/health/live` | Liveness probe (lightweight) |
| GET | `/health/ready` | Readiness probe (checks database connectivity) |

## Configuration

All configuration via environment variables prefixed with `DATA_COLLECTOR_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_COLLECTOR_SERVICE_NAME` | `data-collector` | Service name for telemetry |
| `DATA_COLLECTOR_PORT` | `8151` | HTTP listen port |
| `DATA_COLLECTOR_DATABASE_URL` | `""` | Async SQLAlchemy database URL (`postgresql+psycopg://…`); required in production |
| `DATA_COLLECTOR_OWNER_DATABASE_URL` | `""` | Optional owner (RLS-bypass) DSN for migrations and cross-tenant reads; empty falls back to `DATA_COLLECTOR_DATABASE_URL` |
| `DATA_COLLECTOR_OTLP_TRACES_ENDPOINT` | `""` | OTLP HTTP endpoint for trace export; required in production |
| `DATA_COLLECTOR_TRAINING_TRIGGER_URL` | `""` | POST target when a schema crosses its training threshold; required in production |
| `DATA_COLLECTOR_ENVIRONMENT` | `production` | `production` enables the fail-fast required-field check |
| `DATA_COLLECTOR_LOG_LEVEL` | `INFO` | Root log level |
| `DATA_COLLECTOR_LOG_FORMAT` | `json` | Log format (`json` or `console`) |
| `TENANT_RLS_MODE` | `off` | `enforce` binds `app.current_tenant` and activates the RLS policies |

## Architecture

```
src/
├── main.py                  # FastAPI app with async lifespan, health endpoints, tenant middleware
├── settings.py              # Pydantic BaseSettings (DATA_COLLECTOR_ prefix)
├── db.py                    # async engine/session factories, migrations, health check
├── rls.py                   # generic PostgreSQL row-level-security helpers
├── read_scope.py            # tenant-scoped read resolution
├── tenant_binding.py        # X-Tenant-Id / X-Effective-Tenant-Id resolution
├── logging_config.py        # structlog JSON/console modes and OTEL trace injection
├── telemetry.py             # OpenTelemetry setup with graceful degradation
├── models/
│   ├── db.py                # SQLAlchemy ORM models
│   └── domain.py            # Pydantic request/response schemas
├── routers/
│   ├── samples.py           # /api/samples CRUD, export, stats, mark-used, purge
│   └── schemas.py           # /api/schemas discovery endpoint
└── services/
    └── sample_service.py    # dedup, filtering, export streaming, stats aggregation
alembic/
└── versions/                # database migrations
```

## Testing

The suite requires a real PostgreSQL (no SQLite). The DSN is resolved in one place,
`tests/db_url.py`: `TEST_DATABASE_URL` → `DATABASE_URL` → `localhost:5451/data_collector`.

```bash
# throwaway local PostgreSQL 18
docker run -d --name dc-testpg -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=data_collector -p 5451:5432 postgres:18-alpine

TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5451/data_collector" \
  DATA_COLLECTOR_ENVIRONMENT=test uv run pytest
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format --check .
```

There is no coverage floor.

## CI

`.github/workflows/ci-public.yml` runs on GitHub-hosted runners: betterleaks secret scan,
the private-reference guard, ruff, the full pytest suite against a PostgreSQL 18 service
container, and `pip-audit`. The image build/publish job is off unless the repository variable
`PUBLISH_IMAGE=true` is set.

## Docker

```bash
docker build -t data-collector .
docker run --rm -p 8151:8151 \
  -e DATA_COLLECTOR_ENVIRONMENT=development \
  -e DATA_COLLECTOR_DATABASE_URL=postgresql+psycopg://postgres:postgres@host.docker.internal:5451/data_collector \
  data-collector
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Licensing

There is no licence gate in this service: it needs no licence key, token, or network call to
build, test or run. Every feature is available by default.
