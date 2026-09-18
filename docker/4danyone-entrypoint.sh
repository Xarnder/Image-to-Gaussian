#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/input /workspace/output /workspace/models /workspace/cache/huggingface /workspace/cache/torch /workspace/logs

# Keep upstream defaults (model_dir="models") while storing weights on the bind mount.
ln -sfn /workspace/models /opt/4danyone/models

cd /opt/4danyone

if [[ -x /opt/venv/bin/python ]]; then
  /opt/venv/bin/python - <<'PY' || true
from pathlib import Path
from fdanyone.download import create_classic_gvhmr_links

models = Path("/workspace/models")
if models.exists():
    create_classic_gvhmr_links(
        models,
        "third_party/GVHMR",
        require_models=False,
        require_smplx=False,
    )
PY
fi

exec "$@"
