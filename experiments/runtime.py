import json
import hashlib
import os
from pathlib import Path
import secrets
import shlex
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def data_directory(task):
    """Keep downloaded/generated inputs separate from Python source."""
    return Path(os.environ.get(f"{task.upper()}_DATA_DIR", ROOT / "datasets" / task)).resolve()


def lean_workspace():
    return Path(os.environ.get("LEAN_WORKSPACE", ROOT / "upstream/lean")).resolve()


def input_identity(paths):
    """Record checksums of the actual inputs, including externally supplied pools."""
    result = {}
    for name in paths:
        path = Path(name).resolve()
        digest = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            checksum = digest.hexdigest()
        else:
            checksum = None  # Dry-run can inspect a recipe before preparation.
        result[str(path)] = checksum
    return result


def environment_settings(defaults):
    return {name: os.environ.get(name, str(default)) for name, default in defaults.items()}


def new_run(task, variant, config, resume=None):
    if resume:
        run = Path(resume).resolve()
        saved = json.loads((run / "run.json").read_text())
        if (saved["task"], saved["variant"]) != (task, variant):
            raise ValueError("Resume task/variant does not match run.json")
        for key, value in config.items():
            if saved["config"].get(key) != value:
                raise ValueError(f"Resume configuration changed: {key}")
        seed = saved["seed"]
    else:
        seed = secrets.randbelow(2**31 - 1)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = Path(os.environ.get("OUT", ROOT / "runs")).resolve() / task / f"{variant}_{stamp}_s{seed}"
        run.mkdir(parents=True, exist_ok=False)
        (run / "run.json").write_text(json.dumps({
            "task": task, "variant": variant, "seed": seed, "config": config,
        }, indent=2) + "\n")
    print(f"Run: {run}\nAutomatic seed: {seed}", flush=True)
    os.environ["COMPASSPLAY_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return run, seed


def execute(command, cwd, dry_run=False):
    print(shlex.join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def require_files(*paths):
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {path}; run prepare_data.sh first")
