import os
import re
import json
import random
import torch
import numpy as np
from pathlib import Path

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_id(s: str) -> str | None:
    m = re.search(r"(\d{3})", str(s))
    return m.group(1) if m else None

def atomic_write_json(path: Path, data: dict):
    """Writes JSON atomically to prevent corruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)

def ensure_unique_run_dir(runs_root: Path, run_name: str) -> Path:
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