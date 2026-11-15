#!/usr/bin/env bash
# Production entrypoint for data-collector when run as a host process under
# systemd-user on pgx. Invoked by
# gitops-production/scripts/host-configs/pgx/systemd/data-collector.service.
# Sibling pattern: pytorch_backend / pytorch_ingestion_backend / model-trainer.
#
# Reads runtime config from $ENV_FILE (defaults to
# $XDG_CONFIG_HOME/data-collector/env, falls back to /etc/data-collector/env).
# Source-of-truth for that file is the gitops ConfigMap (when wired up);
# until then it's a hand-managed file on pgx (mode 600).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="${ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/data-collector/env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
elif [ -f /etc/data-collector/env ]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/data-collector/env
  set +a
fi

# uv on PATH (installer drops it at ~/.local/bin/uv).
export PATH="${HOME}/.local/bin:${PATH}"

exec uv run uvicorn main:app \
  --app-dir src \
  --host "${DATA_COLLECTOR_BIND_HOST:-0.0.0.0}" \
  --port "${DATA_COLLECTOR_PORT:-8151}"
