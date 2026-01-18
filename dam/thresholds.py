import json
from pathlib import Path

import numpy as np


def resolve_under_model_dir(model_path: Path, maybe_rel: Path) -> Path:
    """If maybe_rel is relative, resolve it next to model_path."""
    return maybe_rel if maybe_rel.is_absolute() else (model_path.parent / maybe_rel)


def load_threshold_vector(path: Path, num_classes: int, fallback_thr: float) -> tuple[np.ndarray, dict]:
    """Load per-item thresholds from JSON.

    Supports:
      - JSON dict containing {"thresholds": [...]} (preferred)
      - raw JSON list [...]

    Returns: (thr_vec, info_dict)
    """
    info = {
        "threshold_mode": "scalar_fallback",
        "threshold_vector_path": str(path),
    }

    if not path.exists():
        thr_vec = np.full((num_classes,), float(fallback_thr), dtype=np.float32)
        info["threshold_mode"] = "scalar_fallback_missing_json"
        return thr_vec, info

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        arr = obj
        info["threshold_mode"] = "vector_from_json_list"
    elif isinstance(obj, dict) and "thresholds" in obj:
        arr = obj["thresholds"]
        info["threshold_mode"] = "vector_from_json.thresholds"
    else:
        raise ValueError(f"Unsupported threshold JSON format in {path}. Expect list or dict with 'thresholds'.")

    if not isinstance(arr, list) or len(arr) != num_classes:
        raise ValueError(
            f"Threshold vector must be a list of length {num_classes}. Got {type(arr)} len={len(arr) if isinstance(arr, list) else 'NA'}."
        )

    thr_vec = np.array(arr, dtype=np.float32)
    thr_vec = np.clip(thr_vec, 0.0, 1.0)
    return thr_vec, info
