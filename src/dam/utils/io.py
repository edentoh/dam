import os
import json
from pathlib import Path

def atomic_write_json(path: Path, data: dict):
    """
    Writes JSON to a temporary file first, then renames it to the target path.
    Prevents file corruption if the process is interrupted during write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    os.replace(tmp_path, path)

def ensure_unique_run_dir(runs_root: Path, run_name: str) -> Path:
    """
    Creates a unique directory for an experiment run.
    If 'run_name' exists, appends '__{i}' (e.g., 'run_name__2').
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    base = runs_root / run_name
    
    if not base.exists():
        base.mkdir()
        return base

    i = 2
    while True:
        cand = runs_root / f"{run_name}__{i}"
        if not cand.exists():
            cand.mkdir()
            return cand
        i += 1