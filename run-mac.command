#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

exec /opt/homebrew/bin/conda run --no-capture-output -n fooocus \
  python launch.py --preset mac --disable-preset-download --disable-offload-from-vram "$@"
