"""Record target coverage from SGS's accumulated verified proof pool."""

import json
import os
from pathlib import Path


def save_coverage(dataset, iteration):
    targets = list(dataset.target_statements.values())
    if any(not proof.is_correct for target in targets for proof in target.proofs):
        raise ValueError("Target pool contains an unverified proof")
    solved = sum(bool(target.proofs) for target in targets)
    row = {"iteration": iteration, "solved": solved, "total": len(targets),
           "coverage": solved / len(targets) if targets else 0.0}
    path = Path(os.environ["COMPASSPLAY_COVERAGE_FILE"])
    rows = json.loads(path.read_text()) if path.exists() else []
    rows = [entry for entry in rows if entry["iteration"] != iteration] + [row]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(sorted(rows, key=lambda x: x["iteration"]), indent=2) + "\n")
    temporary.replace(path)
    print(f"Target coverage after verification, iteration {iteration}: {solved}/{len(targets)}", flush=True)
