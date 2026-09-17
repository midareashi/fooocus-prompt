#!/bin/zsh
set -eu

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "This launcher requires an Apple-silicon Mac."
  exit 1
fi

CONDA_ENV="${FOOOCUS_CONDA_ENV:-fooocus}"

if [[ -n "${FOOOCUS_CONDA_BIN:-}" ]]; then
  CONDA_BIN="$FOOOCUS_CONDA_BIN"
elif [[ -x /opt/homebrew/bin/conda ]]; then
  CONDA_BIN=/opt/homebrew/bin/conda
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  print -u2 "Conda was not found. Install Miniforge, or set FOOOCUS_CONDA_BIN."
  exit 1
fi

if ! "$CONDA_BIN" run -n "$CONDA_ENV" python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))' >/dev/null 2>&1; then
  print -u2 "Conda environment '$CONDA_ENV' is missing or is not using Python 3.10."
  print -u2 "Create it with: $CONDA_BIN env create -f environment.yaml"
  exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
  shift
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -c '
import platform
import sys
import torch
import torchvision
from modules.launch_util import requirements_met

print(f"macOS: {platform.mac_ver()[0]} ({platform.machine()})")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
if not requirements_met("requirements_versions.txt"):
    sys.exit("Fooocus Python requirements are incomplete.")
if not torch.backends.mps.is_built():
    sys.exit("This PyTorch build has no Metal (MPS) support.")
if not torch.backends.mps.is_available():
    sys.exit("Metal (MPS) is not available to PyTorch.")
probe = torch.ones(1, device="mps")
print(f"Metal test: {probe.device} is working")
print("Fooocus Mac preflight passed. Model files were not checked.")
'
  exit $?
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
# Avoid launch.py selecting CUDA wheels if Torch ever needs to be reinstalled.
export TORCH_COMMAND="pip install torch==2.1.0 torchvision==0.16.0"

exec "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python launch.py --disable-preset-download --disable-offload-from-vram "$@"
