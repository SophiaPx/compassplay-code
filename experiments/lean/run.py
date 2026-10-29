"""Start a main Lean experiment with an automatically generated run seed."""

import argparse
import os
from pathlib import Path
import sys

from experiments.runtime import (
    ROOT, data_directory, lean_workspace, environment_settings, input_identity,
    execute, new_run, require_files,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", choices=["cos_novelty", "cos", "difficulty", "sgs", "solve_only"], default="cos_novelty")
    parser.add_argument("--resume", help="Existing run directory, retaining its original seed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    preseed = args.variant in ("cos", "cos_novelty")
    dataset = data_directory("lean") / ("D_3k_preseeded_prover_dataset.json" if preseed else "D_3k_prover_dataset.json")
    config = {
        "model": os.environ.get("MODEL", "deepseek-ai/DeepSeek-Prover-V2-7B"),
        "guide_model": os.environ.get("GUIDE_MODEL", "LukeBailey181Pub/dspv2_guide"),
        "dataset": str(dataset), "iterations": int(os.environ.get("TOTAL_ITERATIONS", "150")),
        "inputs": input_identity([dataset, data_directory("lean") / "conjecturer_dataset.json"]),
        "environment": environment_settings({
            "SGS_GRAD_DEVICE": "cuda:0", "SGS_GRAD_MAX_PROOFS": 4, "SGS_GRAD_MAX_LEN": 2048,
            "SGS_GPU_MEM_UTIL": 0.7, "SGS_MAX_MODEL_LEN": "",
            "SGS_TRAIN_GPUS": "", "SGS_ENGINE_INIT_RETRIES": 8, "SGS_ENGINE_INIT_WAIT": 20,
            "SGS_REPL_PATH": lean_workspace() / "repl/.lake/build/bin/repl",
            "SGS_MATHLIB_PATH": lean_workspace() / "mathlib4",
            "SGS_VERIFIER_TIMEOUT": 500,
        }),
    }
    run, _ = new_run("lean", args.variant, config, args.resume)
    os.environ["COMPASSPLAY_RUN"] = str(run)
    os.environ["COMPASSPLAY_VARIANT"] = args.variant
    os.environ["COMPASSPLAY_COVERAGE_FILE"] = str(run / "coverage.json")
    os.environ["SGS_NOVELTY"] = "jaccard" if args.variant == "cos_novelty" else "none"
    os.environ.setdefault("SGS_GRAD_DEVICE", "cuda:0")
    os.environ.setdefault("SGS_GRAD_MAX_PROOFS", "4")
    os.environ.setdefault("SGS_GRAD_MAX_LEN", "2048")
    os.environ.setdefault("SGS_GPU_MEM_UTIL", "0.7")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("SGS_MATHLIB_PATH", str(lean_workspace() / "mathlib4"))
    os.environ.setdefault("SGS_REPL_PATH", str(lean_workspace() / "repl/.lake/build/bin/repl"))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])
    if not args.dry_run:
        require_files(dataset, data_directory("lean") / "conjecturer_dataset.json", os.environ["SGS_REPL_PATH"])
    execute([sys.executable, "-m", "experiments.lean.train"], ROOT, args.dry_run)


if __name__ == "__main__":
    main()
