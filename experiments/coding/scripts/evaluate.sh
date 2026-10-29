#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
# Evaluate all eight benchmarks on a Hugging Face model directory or model ID.
python3 -m experiments.coding.run --eval "$@"
python3 -m experiments.coding.evaluate_math "$@"
