#!/usr/bin/env bash
# Sourced by the public scripts. Resolve all defaults relative to this checkout.
COMPASSPLAY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export COMPASSPLAY_ROOT
export PYTHONPATH="$COMPASSPLAY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

checkout_upstream() {
    local url="$1" destination="$2" revision="$3"
    if [ ! -d "$destination" ]; then
        git clone "$url" "$destination"
        git -C "$destination" checkout "$revision"
    fi
    local expected actual
    expected=$(git -C "$destination" rev-parse "$revision")
    actual=$(git -C "$destination" rev-parse HEAD)
    if [ "$expected" != "$actual" ]; then
        echo "Expected $destination at $revision; use a separate clean checkout." >&2
        return 1
    fi
}

