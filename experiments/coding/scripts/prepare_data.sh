#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
python3 -m experiments.download_data coding
python3 -m experiments.coding.data.prepare_benchmarks
python3 -m experiments.coding.prepare_data
