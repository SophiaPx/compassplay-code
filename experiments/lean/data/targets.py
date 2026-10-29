# Source: https://github.com/LukeBailey181/sgs/blob/64d2de468a0ddf222d5836248bea1d239a253f49/sgs/data/load_dataset.py; modified for CompassPlay.
import os
import random
from typing import List

from datasets import load_dataset

from experiments.lean.data.types import DatasetType, Statement, StatementTag


def convert_raw_dict_to_statements(
    raw_dict: List[dict], dataset: DatasetType
) -> List[Statement]:
    if dataset != DatasetType.D_3K:
        raise ValueError(f"Invalid dataset type: {dataset}")

    ids = [row["id"] for row in raw_dict]
    assert len(ids) == len(set(ids)), "Ids are not unique"

    statements: List[Statement] = [
        Statement(
            id=row["id"],
            header=row["header"],
            theorem=row["theorem"],
            tag=StatementTag.TARGET.value,
            source=dataset.value,
            proofs=[],
        )
        for row in raw_dict
    ]

    for statement in statements:
        assert (
            "sorry" not in statement.theorem
        ), f"Statement {statement.id} has a sorry in the theorem"
        assert statement.theorem.endswith(
            ":= by"
        ), f"Statement {statement.id} does not end with ':= by'"

    return statements


def load_eval_dataset(eval_dataset: DatasetType) -> List[Statement]:
    if eval_dataset != DatasetType.D_3K:
        raise ValueError(f"Invalid dataset type: {eval_dataset}")

    raw = load_dataset("LukeBailey181Pub/D_3k")["train"]
    rows = [x for x in raw]
    # Optional: deterministic subsample for a cheap, consistent trend-curve eval.
    n = os.environ.get("SGS_EVAL_N")
    if n:
        idx = list(range(len(rows)))
        random.Random(0).shuffle(idx)
        rows = [rows[i] for i in sorted(idx[: int(n)])]
    return convert_raw_dict_to_statements(rows, eval_dataset)


if __name__ == "__main__":
    dataset = load_eval_dataset(DatasetType.D_3K)
    print(f"Loaded {len(dataset)} statements from D_3K")
