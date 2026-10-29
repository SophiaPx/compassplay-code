"""Run the five mathematics benchmarks with the AZR evaluation harness."""

import argparse
import os
from pathlib import Path
import sys

from experiments.runtime import ROOT, data_directory, execute, new_run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model = os.environ.get("MODEL", "Qwen/Qwen2.5-Coder-7B")
    run, seed = new_run("coding", "eval_math", {"model": model, "temperature": 0, "k": 1})
    if os.environ.get("ROCR_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.pop("ROCR_VISIBLE_DEVICES")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.setdefault("EVAL_GPU_MEM_UTIL", "0.9")
    # Default to TP=1 and keep it independent of the visible-device list:
    # Qwen2.5-Coder-7B's 28 attention heads do not divide across eight GPUs.
    os.environ.setdefault("EVAL_TP", "1")
    execute([
        sys.executable, "-m", "experiments.coding.evaluation.run", "--model_name_or_path", model,
        "--data_names", "math500,amc23,gsm8k,minerva_math,olympiadbench",
        "--data_dir", data_directory("coding") / "evaluation", "--output_dir", run,
        "--prompt_type", "azr", "--split", "test", "--num_test_sample", -1,
        "--temperature", 0, "--top_p", 1, "--n_sampling", 1,
        "--max_tokens_per_call", 4096, "--seed", seed, "--use_vllm", "--save_outputs",
    ], ROOT, args.dry_run)


if __name__ == "__main__":
    main()
