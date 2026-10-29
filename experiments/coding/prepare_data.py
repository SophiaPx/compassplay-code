"""Rebuild the fixed MATH-train reference pool; do not package dataset files."""

import hashlib
import json
import os
from pathlib import Path
import random
import re

from datasets import load_dataset
from experiments.coding.data.formatting import extract_solution
from experiments.runtime import data_directory


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    path = Path(os.environ.get(
        "REF_FILE", data_directory("coding") / "code_reason/gref_heldout_math.jsonl"
    )).resolve()
    if path.exists():
        # Reuse an existing pool unchanged instead of rebuilding it.
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        valid = [row for row in rows if row.get("problem") and row.get("answer") is not None]
        if len(valid) < 32:
            raise ValueError(f"Reference pool contains fewer than 32 usable problems: {path}")
        print(f"Reusing reference pool unchanged: {path} ({len(valid)} usable problems, ref_seed=0)")
        return
    evaluation = data_directory("coding") / "evaluation"
    excluded = set()
    for name in ("math500", "amc23", "gsm8k", "minerva_math", "olympiadbench"):
        with (evaluation / name / "test.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                problem = row.get("problem", row.get("question"))
                if problem:
                    excluded.add(normalize(problem))
    dataset_name = "DigitalLearningGmbH/MATH-lighteval"
    dataset = load_dataset(dataset_name, split="train")
    rows = []
    for index, row in enumerate(dataset):
        if normalize(row["problem"]) in excluded:
            continue
        answer = row.get("answer")
        if answer is None:
            answer = extract_solution(row["solution"])
        if answer is None:
            raise ValueError(f"No gold answer for MATH train row {index}")
        rows.append({"id": row.get("id", index), "problem": row["problem"],
                     "solution": row["solution"], "answer": answer})
    if len(rows) < 32:
        raise ValueError("Fewer than 32 disjoint reference problems")
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to silently change a pool already used by a run.
    if path.exists() and path.read_text() != content:
        raise FileExistsError(f"Reference pool changed: preserve {path} before rebuilding")
    path.write_text(content)
    selected = random.Random(0).sample(rows, 32)
    path.with_suffix(".manifest.json").write_text(json.dumps({
        "dataset": dataset_name, "split": "train", "fingerprint": dataset._fingerprint,
        "pool_size": len(rows), "ref_seed": 0, "n_ref": 32,
        "selected_ids": [row["id"] for row in selected],
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }, indent=2) + "\n")
    print(f"Fixed reference pool: {len(rows)} problems; 32 selected with ref_seed=0")


if __name__ == "__main__":
    main()
