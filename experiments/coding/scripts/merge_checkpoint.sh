#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
if [ "$#" -ne 2 ]; then
    echo "Usage: bash merge_checkpoint.sh /path/to/global_step_N/actor /path/to/hf_model" >&2
    exit 2
fi
python3 -m verl.model_merger merge --backend fsdp --local_dir "$1" --target_dir "$2"
