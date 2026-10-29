#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
DS_BUILD_OPS=0 python3 -m pip install --no-deps -r "$COMPASSPLAY_ROOT/experiments/lean/requirements.txt"
LEAN_BUILD_DIR="${LEAN_WORKSPACE:-$COMPASSPLAY_ROOT/upstream/lean}"
mkdir -p "$LEAN_BUILD_DIR"
cd "$LEAN_BUILD_DIR"
bash "$COMPASSPLAY_ROOT/experiments/lean/scripts/build_lean.sh"
echo "Lean setup complete. Run prepare_data.sh --preseed for cosine variants."
