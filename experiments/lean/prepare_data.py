"""Prepare the verified initial seed pool once, separately from training."""

import argparse
import json
import os
from pathlib import Path
import sys

from experiments.runtime import ROOT, data_directory, lean_workspace, execute, new_run
from experiments.download_data import download


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--preseed", action="store_true", help="Generate seed proofs required by cosine variants")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])
    os.environ.setdefault("SGS_MATHLIB_PATH", str(lean_workspace() / "mathlib4"))
    os.environ.setdefault("SGS_REPL_PATH", str(lean_workspace() / "repl/.lake/build/bin/repl"))
    download("lean", dry_run=args.dry_run)
    if not args.preseed:
        print("Base datasets downloaded from the pinned SGS revision.")
        return
    target = data_directory("lean") / "D_3k_preseeded_prover_dataset.json"
    if target.exists():
        print(f"Reusing existing seed pool: {target}")
        return
    config = {"model": os.environ.get("MODEL", "deepseek-ai/DeepSeek-Prover-V2-7B"),
              "k": 6, "max_new": 768, "temperature": 0.8, "top_p": 0.95}
    run, _ = new_run("lean", "prepare_seed_pool", config)
    os.environ["MODEL_PATH"] = config["model"]
    shard = run / "solved.jsonl"
    execute([sys.executable, "-m", "experiments.lean.data.collect_solvable",
             "--start", 0, "--end", 3323, "--k", 6, "--out", shard], ROOT, args.dry_run)
    execute([sys.executable, "-m", "experiments.lean.data.build_preseeded_dataset",
             "--base", data_directory("lean") / "D_3k_prover_dataset.json", "--shards", shard,
             "--out", target], ROOT, args.dry_run)
    if not args.dry_run:
        dataset = json.loads(target.read_text())
        solved = sum(bool(row["proofs"]) for row in dataset["target_statements"].values())
        if not solved:
            target.unlink()
            raise ValueError("Seed collection produced no verified proofs")
        target.with_suffix(".manifest.json").write_text((run / "run.json").read_text())
        print(f"Prepared {solved} solved targets. Reuse this pool for both cosine variants.")


if __name__ == "__main__":
    main()
