"""Download data separately from bundled source, pinned by revision and SHA256."""

import argparse
import hashlib
import os
from pathlib import Path
import urllib.request

from experiments.runtime import data_directory

SOURCES = {
    "coding": ("LeapLabTHU/Absolute-Zero-Reasoner", "484afa480c8f6fd77faa3d35451f24f287f58ee1"),
    "lean": ("LukeBailey181/sgs", "64d2de468a0ddf222d5836248bea1d239a253f49"),
}

FILES = {'coding': {'data/7b_coder_seed_io.jsonl': '33b9608e76737785fa159b94dabe6dfdbc6389a8c55e082ab54e52d9031c717f',
            'data/7b_coder_code_f_seed_io.jsonl': 'ce609e87fe330f90cfa9f167ac050b4df8bd1bf9d4ab135d90b9a0f5d48f4b84',
            'evaluation/math_eval/eval/data/math500/test.jsonl': '35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132',
            'evaluation/math_eval/eval/data/amc23/test.jsonl': '43707e1ca784602d8479de24ee8ce36609dbebc46dd152b0b12a46611aa898a4',
            'evaluation/math_eval/eval/data/gsm8k/test.jsonl': '9f098df7ff4522078d6ccec27a5c38a454790fdb51131bedfaa0e490070d5394',
            'evaluation/math_eval/eval/data/minerva_math/test.jsonl': '0e656b430d3a0a1dab6aa853f453ba3805e28e2cabe4fc72486a897cc09d8ba2',
            'evaluation/math_eval/eval/data/olympiadbench/test.jsonl': 'bb8c1e38a16ea9eacb77530315a674d883f267fbff79183effc53681b4fcc416'},
 'lean': {'data/D_3k_prover_dataset.json': '1c2a89d8347280f4cf11ae71fca5f0274f9b8efa4f9b1d18b6eaddf62d0c8f42',
          'data/conjecturer_dataset.json': 'dd83affcbe53c035b62b8ea9673a2f73b35d1849ab08dcb2ec9170746efa08a7'}}


def download(task, dry_run=False):
    repo, revision = SOURCES[task]
    destination = data_directory(task)
    for relative, expected in FILES[task].items():
        local = relative.removeprefix("data/")
        if relative.startswith("evaluation/math_eval/eval/data/"):
            local = "evaluation/" + relative.removeprefix("evaluation/math_eval/eval/data/")
        target = destination / local
        if target.exists():
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Existing data differ from the pinned source: {target}")
            continue
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{relative}"
        print(f"Download {url} -> {target}", flush=True)
        if dry_run:
            continue
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Downloaded data checksum mismatch: {url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("task", choices=SOURCES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    download(args.task, args.dry_run)


if __name__ == "__main__":
    main()
