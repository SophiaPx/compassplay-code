# Written for CompassPlay.
import argparse
import glob
import json


def load_solved_by_theorem(pattern):
    by_thm, seen_idx = {}, set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["idx"] in seen_idx:
                continue
            seen_idx.add(r["idx"])
            correct = [p for p in r.get("correct", []) if p and p.strip()]
            if correct:
                by_thm[r["theorem"].strip()] = correct
    return by_thm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="upstream data/D_3k_prover_dataset.json")
    ap.add_argument("--shards", required=True, help="glob of collect_solvable.py outputs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = json.load(open(args.base))
    by_thm = load_solved_by_theorem(args.shards)
    print(f"solved theorems from shards: {len(by_thm)}")

    n_injected, n_proofs = 0, 0
    for stmt in ds["target_statements"].values():
        bodies = by_thm.get(stmt["theorem"].strip())
        if not bodies:
            continue
        for body in bodies:
            stmt["proofs"].append({
                "proof_str": body,
                "full_generation": body,
                "is_correct": True,
                "iteration_created": 0,
            })
        n_injected += 1
        n_proofs += len(bodies)

    ds["iterations"] = ds.get("iterations", [])

    json.dump(ds, open(args.out, "w"))
    print(f"injected proofs into {n_injected} target_statements ({n_proofs} proofs total)")
    print(f"wrote {args.out}")

    check = json.load(open(args.out))
    with_proof = sum(1 for s in check["target_statements"].values() if s["proofs"])
    assert with_proof == len(by_thm), f"expected {len(by_thm)} seeded, got {with_proof}"


if __name__ == "__main__":
    main()
