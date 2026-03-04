import json
from pathlib import Path
import numpy as np

THRESHOLD_CAP = 0.9

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
        fallback = float(np.clip(fallback_thr, 0.0, THRESHOLD_CAP))
        thr_vec = np.full((num_classes,), fallback, dtype=np.float32)
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
    thr_vec = np.clip(thr_vec, 0.0, THRESHOLD_CAP)
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
    
    # Keep threshold search capped project-wide.
    grid = np.linspace(0.0, THRESHOLD_CAP, num_steps, dtype=np.float32)
    
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


def optimize_thresholds_micro_f1(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_steps: int = 101,
    max_rounds: int = 5,
    min_threshold: float = 0.0,
    max_threshold: float = THRESHOLD_CAP,
    tie_break: str = "closest_to_0.5",
    score_tolerance: float = 0.0,
) -> tuple[np.ndarray, float, list, list]:
    """
    Finds a per-class threshold vector that maximizes global micro-F1.

    Optimization method:
    - Coordinate ascent over classes.
    - For each class, scan threshold grid [0, 1] and keep the threshold that
      gives the best global micro-F1 while all other class thresholds are fixed.
    - Optional conservative tie-break can prefer higher thresholds among
      candidates whose micro-F1 is effectively equivalent.

    Args:
        min_threshold: lower bound for threshold search (inclusive).
        max_threshold: upper bound for threshold search (inclusive).
        tie_break: one of {"closest_to_0.5", "highest", "lowest"}.
        score_tolerance: allow selecting thresholds with score >= (best - tol).
            Use 0.0 to keep exact-best behavior.

    Returns:
        best_thresholds: (num_classes,)
        best_micro_f1: scalar
        pos_rates: per-class positive rates in y_true
        threshold_list: python list of thresholds
    """
    y_true = (y_true > 0).astype(np.int32)
    y_probs = np.asarray(y_probs, dtype=np.float32)

    if y_true.ndim != 2 or y_probs.ndim != 2 or y_true.shape != y_probs.shape:
        raise ValueError(f"Shape mismatch: y_true{y_true.shape} vs y_probs{y_probs.shape}")

    _, num_classes = y_true.shape
    lo = float(np.clip(min_threshold, 0.0, THRESHOLD_CAP))
    hi = float(np.clip(max_threshold, 0.0, THRESHOLD_CAP))
    if lo > hi:
        raise ValueError(f"min_threshold ({lo}) cannot be greater than max_threshold ({hi}).")
    tol = max(0.0, float(score_tolerance))

    valid_tie_breaks = {"closest_to_0.5", "highest", "lowest"}
    tie_break = str(tie_break)
    if tie_break not in valid_tie_breaks:
        raise ValueError(f"Unsupported tie_break='{tie_break}'. Expected one of {sorted(valid_tie_breaks)}.")

    grid = np.linspace(lo, hi, num_steps, dtype=np.float32)

    init_thr = float(np.clip(0.5, lo, hi))
    thresholds = np.full((num_classes,), init_thr, dtype=np.float32)
    y_pred = (y_probs >= thresholds.reshape(1, -1)).astype(np.int32)

    tp = int((y_pred * y_true).sum())
    fp = int((y_pred * (1 - y_true)).sum())
    fn = int(((1 - y_pred) * y_true).sum())

    def _micro_f1(tp_v: int, fp_v: int, fn_v: int) -> float:
        denom = (2 * tp_v + fp_v + fn_v)
        return float((2 * tp_v) / denom) if denom > 0 else 0.0

    best_micro = _micro_f1(tp, fp, fn)
    pos_rates = [float(y_true[:, j].mean()) for j in range(num_classes)]

    print(
        f"[Thresholds] Optimizing {num_classes} classes for micro-F1 "
        f"(grid={num_steps}, max_rounds={max_rounds}, range=[{lo:.3f}, {hi:.3f}], "
        f"tie_break={tie_break}, tol={tol:.6f})..."
    )

    for _ in range(max(1, int(max_rounds))):
        any_change = False

        for j in range(num_classes):
            yj_true = y_true[:, j]
            col_cur = y_pred[:, j]

            tp_cur = int((col_cur * yj_true).sum())
            fp_cur = int((col_cur * (1 - yj_true)).sum())
            fn_cur = int(((1 - col_cur) * yj_true).sum())

            tp_other = tp - tp_cur
            fp_other = fp - fp_cur
            fn_other = fn - fn_cur

            scores_for_j = []
            for t in grid:
                col_new = (y_probs[:, j] >= t).astype(np.int32)
                tp_new_j = int((col_new * yj_true).sum())
                fp_new_j = int((col_new * (1 - yj_true)).sum())
                fn_new_j = int(((1 - col_new) * yj_true).sum())

                score = _micro_f1(tp_other + tp_new_j, fp_other + fp_new_j, fn_other + fn_new_j)
                scores_for_j.append((float(t), float(score)))

            best_score_j = max(s for _, s in scores_for_j)
            admissible = [t for t, s in scores_for_j if s >= (best_score_j - tol)]

            if tie_break == "highest":
                best_t_j = max(admissible)
            elif tie_break == "lowest":
                best_t_j = min(admissible)
            else:
                # Default behavior for backward compatibility.
                best_t_j = min(admissible, key=lambda x: abs(x - 0.5))

            if abs(best_t_j - float(thresholds[j])) > 1e-12:
                thresholds[j] = np.float32(best_t_j)
                new_col = (y_probs[:, j] >= thresholds[j]).astype(np.int32)
                y_pred[:, j] = new_col

                tp = int((y_pred * y_true).sum())
                fp = int((y_pred * (1 - y_true)).sum())
                fn = int(((1 - y_pred) * y_true).sum())
                best_micro = _micro_f1(tp, fp, fn)
                any_change = True

        if not any_change:
            break

    best_micro = _micro_f1(tp, fp, fn)
    threshold_list = [float(x) for x in thresholds.tolist()]
    return thresholds, best_micro, pos_rates, threshold_list
