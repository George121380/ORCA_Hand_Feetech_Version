#!/usr/bin/env bash
# Linux + NVIDIA only. Separate from uv sync, whose existing lock uses CPU torch.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo "This installer is for the Linux 4090 host. No training starts here." >&2
  exit 1
fi
command -v uv >/dev/null || { echo "Install uv from https://docs.astral.sh/uv/ first."; exit 1; }
command -v nvidia-smi >/dev/null || { echo "Install the NVIDIA host driver first."; exit 1; }
nvidia-smi
python3 scripts/bootstrap.py
uv sync --frozen --extra train --extra vision --extra hardware
# The repository lock deliberately selects CPU torch on Linux for ordinary CI.
# Replace only torch after syncing. Do not run uv sync on this GPU environment
# afterwards: it would restore CPU torch. Re-run this script instead.
uv pip install --python .venv/bin/python --reinstall-package torch \
  'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python - <<'PY'
import json
import torch, mujoco, cv2
assert torch.cuda.is_available(), "CUDA torch is installed, but the driver/device is unavailable"
x = torch.ones(1024, device="cuda")
assert x.sum().item() == 1024
print(json.dumps(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                     cuda=torch.version.cuda, mujoco=mujoco.__version__, opencv=cv2.__version__), indent=2))
PY
mkdir -p local
uv pip freeze --python .venv/bin/python > local/training-environment.txt
echo "Ready. No model training was started. Read docs/software/sim2real.md for smoke and curriculum commands."
