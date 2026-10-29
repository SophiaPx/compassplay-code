#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Normalize ROCm visibility before importing torch or vLLM.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
export HIP_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
if [ -f "$HOME/.elan/env" ]; then source "$HOME/.elan/env"; fi
exec python3 -m experiments.lean.run "$@"
