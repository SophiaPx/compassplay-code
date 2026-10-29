#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
VERL_PATH="${VERL_PATH:-$COMPASSPLAY_ROOT/upstream/verl}"
mkdir -p "$(dirname "$VERL_PATH")"
checkout_upstream https://github.com/volcengine/verl.git "$VERL_PATH" v0.8.0
python3 -m pip install --no-deps -r "$COMPASSPLAY_ROOT/experiments/coding/requirements.txt"
python3 -m pip install --no-deps -e "$VERL_PATH"
echo "Coding setup complete. Run prepare_data.sh next."
