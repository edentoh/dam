import json
from pathlib import Path
import numpy as np

def resolve_under_model_dir(model_path: Path, maybe_rel: Path) -> Path:
    """If maybe_rel is relative, resolve it next to model_path."""
    return maybe_rel if maybe_rel.is_absolute() else (model_path.parent / maybe_rel)

def load_threshold_vector(path: Path, num_classes: int, fallback_thr: float) -> tuple[np.ndarray, dict]:
    """
    Load per-item thresholds from JSON.
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

def optimize_thresholds_accuracy(
    y_true: np.ndarray, 
    y_probs: np.ndarray, 
    num_steps: int = 101
) -> tuple[np.ndarray, list, list, list]:
    """
    Finds thresholds maximizing Accuracy per class (matching original logic).
    Tie-break: choose threshold closest to 0.5.
    
    Returns:
        best_thresholds: (num_classes,)
        best_accs: list of best accuracies per class
        pos_rates: list of positive rates per class
        all_thresholds: list of thresholds chosen
    """
    num_classes = y_true.shape[1]
    
    # Original logic used linspace(0.0, 1.0, 101) -> step 0.01
    grid = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    
    thresholds = []
    per_item_best_acc = []
    per_item_pos_rate = []

    print(f"[Thresholds] Optimizing {num_classes} classes for Accuracy (grid={num_steps})...")

    for j in range(num_classes):
        yt = y_true[:, j].astype(np.int32)
        yp = y_probs[:, j]
        
        best_acc = -1.0
        best_ts = []
        
        for t in grid:
            y_pred = (yp >= t).astype(np.int32)
            acc = float((y_pred == yt).mean())
            
            # Logic strict match: > best + epsilon
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_ts = [float(t)]
            elif abs(acc - best_acc) <= 1e-12:
                best_ts.append(float(t))
        
        # Tie-break: closest to 0.5
        best_t = min(best_ts, key=lambda x: abs(x - 0.5))
        
        thresholds.append(best_t)
        per_item_best_acc.append(best_acc)
        per_item_pos_rate.append(float(yt.mean()))
        
    return np.array(thresholds, dtype=np.float32), per_item_best_acc, per_item_pos_rate, thresholds