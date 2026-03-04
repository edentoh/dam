import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# --- Modular Imports ---
from dam.core.config import load_config
from dam.inference.predictor import DAMPredictor
from dam.inference.utils import (
    list_images, 
    load_labels_from_excel_strict, 
    get_labels_path_for_predict
)
# FIXED: resolve_under_model_dir is in thresholds, not utils
from dam.inference.thresholds import optimize_thresholds_micro_f1, resolve_under_model_dir
from dam.utils.io import atomic_write_json
from dam.utils.seeding import seed_everything


def _compute_metrics_under_thresholds(y_true_bin: np.ndarray, y_probs: np.ndarray, thresholds: np.ndarray):
    y_pred = (y_probs >= thresholds.reshape(1, -1)).astype(np.int32)
    overall_acc = float((y_pred == y_true_bin).mean())

    tp = float((y_pred * y_true_bin).sum())
    fp = float((y_pred * (1 - y_true_bin)).sum())
    fn = float(((1 - y_pred) * y_true_bin).sum())
    micro_f1 = float((2 * tp) / (2 * tp + fp + fn + 1e-9))

    tp_c = (y_pred * y_true_bin).sum(axis=0).astype(np.float64)
    fp_c = (y_pred * (1 - y_true_bin)).sum(axis=0).astype(np.float64)
    fn_c = ((1 - y_pred) * y_true_bin).sum(axis=0).astype(np.float64)
    macro_f1 = float(np.mean((2.0 * tp_c) / (2.0 * tp_c + fp_c + fn_c + 1e-9)))
    return y_pred, overall_acc, micro_f1, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_score.toml", help="Path to config file")
    args = parser.parse_args()

    # 1. Setup
    cfg = load_config(args.config)
    predict_cfg = cfg.get("predict", {})
    
    # Using defaults from original script (seed 42, device check)
    seed_everything(cfg["system"].get("seed", 42))
    
    device_pref = cfg.get("system", {}).get("device", "cuda")
    device = torch.device(device_pref if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # 2. Config Paths
    model_path = Path(predict_cfg["model_path"])
    input_dir = Path(predict_cfg["input_image_dir"])
    fit_cfg = predict_cfg.get("threshold_fit", {})
    num_steps = int(fit_cfg.get("num_steps", 1001))
    max_rounds = int(fit_cfg.get("max_rounds", 20))
    min_threshold = float(fit_cfg.get("min_threshold", 0.0))
    max_threshold = float(fit_cfg.get("max_threshold", 0.9))
    tie_break = str(fit_cfg.get("tie_break", "closest_to_0.5"))
    score_tolerance = float(fit_cfg.get("score_tolerance", 0.0))
    toward_highest_target = float(fit_cfg.get("toward_highest", 0.0))
    toward_highest_target = float(np.clip(toward_highest_target, 0.0, 1.0))
    enforce_no_perf_drop = bool(fit_cfg.get("enforce_no_perf_drop", True))
    perf_drop_tolerance = max(0.0, float(fit_cfg.get("perf_drop_tolerance", 0.0)))
    safe_search_steps = max(2, int(fit_cfg.get("safe_search_steps", 201)))
    
    out_json_cfg = Path(predict_cfg.get("threshold_vector_path", "threshold_vector.json"))
    out_json = resolve_under_model_dir(model_path, out_json_cfg)

    # 3. Load Labels (Strict)
    labels_path = get_labels_path_for_predict(cfg)
    if not labels_path:
        raise KeyError("Missing labels_path (expected [predict.labels].labels_path or similar)")
    
    label_map = load_labels_from_excel_strict(Path(labels_path))

    # 4. List Images & Filter Common
    images_map = list_images(input_dir)
    common_ids = sorted(set(images_map.keys()) & set(label_map.keys()))
    
    if not common_ids:
        raise RuntimeError("No matching IDs between images in predict.input_image_dir and label spreadsheet columns.")
        
    items = [(img_id, images_map[img_id]) for img_id in common_ids]
    print(f"Labeled images found for threshold fitting: {len(items)}")

    # 5. Run Inference
    # DAMPredictor handles model loading, transforms (incl. CropToInk config), and batching.
    predictor = DAMPredictor(model_path, cfg, device)
    probs_by_id = predictor.predict_batch(items)

    # 6. Align Data for Optimization
    # Create (N, 48) arrays
    y_probs = np.stack([probs_by_id[i] for i in common_ids], axis=0)
    y_true = np.stack([label_map[i] for i in common_ids], axis=0)
    y_true_bin = (y_true > 0).astype(np.int32)

    # 7. Optimize Thresholds (global micro-F1 objective)
    best_thresholds, best_micro_f1, pos_rates, best_thresholds_list = optimize_thresholds_micro_f1(
        y_true_bin,
        y_probs, 
        num_steps=num_steps,
        max_rounds=max_rounds,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        tie_break=tie_break,
        score_tolerance=score_tolerance,
    )
    base_thresholds = best_thresholds.copy()
    _, _, base_micro_f1, _ = _compute_metrics_under_thresholds(y_true_bin, y_probs, base_thresholds)
    min_allowed_micro = float(base_micro_f1 - perf_drop_tolerance)
    toward_highest_applied = float(toward_highest_target)
    max_no_drop_thresholds = np.full_like(base_thresholds, np.float32(max_threshold))

    if toward_highest_target > 1e-12:
        # 1) Build per-class safe ceilings (with other classes fixed at base thresholds).
        #    This avoids one sensitive class blocking all others.
        if enforce_no_perf_drop:
            max_no_drop_thresholds = base_thresholds.copy()
            for j in range(base_thresholds.shape[0]):
                lo = float(base_thresholds[j])
                hi = float(max_threshold)
                best_j = lo

                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    thr_try = base_thresholds.copy()
                    thr_try[j] = np.float32(mid)
                    _, _, micro_try, _ = _compute_metrics_under_thresholds(y_true_bin, y_probs, thr_try)
                    if micro_try + 1e-12 >= min_allowed_micro:
                        best_j = mid
                        lo = mid
                    else:
                        hi = mid

                max_no_drop_thresholds[j] = np.float32(best_j)

        def _blend(alpha: float) -> np.ndarray:
            out = base_thresholds + np.float32(alpha) * (max_no_drop_thresholds - base_thresholds)
            return np.clip(out, np.float32(min_threshold), np.float32(max_threshold)).astype(np.float32)

        # 2) Apply user factor toward the safe ceilings, with an optional global guard.
        if enforce_no_perf_drop:
            best_feasible = 0.0
            for alpha in np.linspace(0.0, toward_highest_target, safe_search_steps, dtype=np.float32):
                thr_try = _blend(float(alpha))
                _, _, micro_try, _ = _compute_metrics_under_thresholds(y_true_bin, y_probs, thr_try)
                if micro_try + 1e-12 >= min_allowed_micro and float(alpha) >= best_feasible:
                    best_feasible = float(alpha)
            toward_highest_applied = best_feasible
        else:
            toward_highest_applied = float(toward_highest_target)

        best_thresholds = _blend(toward_highest_applied)
        best_thresholds_list = [float(x) for x in best_thresholds.tolist()]
        print(
            f"[Thresholds] toward_highest target={toward_highest_target:.3f}, "
            f"applied={toward_highest_applied:.3f} "
            f"(guard={'on' if enforce_no_perf_drop else 'off'}, tol={perf_drop_tolerance:.6f})"
        )

    # 8. Calculate Overall Metrics
    y_pred, overall_acc, micro_f1_under_vec, macro_f1_under_vec = _compute_metrics_under_thresholds(
        y_true_bin, y_probs, best_thresholds
    )
    
    # Per-item accuracy under vector
    per_item_acc_under_vec = (y_pred == y_true_bin).mean(axis=0)

    # 9. Save Output
    out_data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_model_path": str(model_path),
        "input_image_dir": str(input_dir),
        "label_file": str(labels_path),
        "backbone": cfg["model"]["backbone"],
        "img_size": int(predictor.img_size),
        "num_images_used": int(len(items)),
        "threshold_grid_step": float((max_threshold - min_threshold) / (num_steps - 1)) if num_steps > 1 else None,
        "threshold_objective": "micro_f1",
        "threshold_optimizer": "coordinate_ascent",
        "optimizer_max_rounds": int(max_rounds),
        "threshold_search_min": float(min_threshold),
        "threshold_search_max": float(max_threshold),
        "threshold_tie_break": tie_break,
        "score_tolerance": float(score_tolerance),
        "toward_highest_target": float(toward_highest_target),
        "toward_highest_applied": float(toward_highest_applied),
        "enforce_no_perf_drop": bool(enforce_no_perf_drop),
        "perf_drop_tolerance": float(perf_drop_tolerance),
        "safe_search_steps": int(safe_search_steps),
        "base_micro_f1_before_toward": float(base_micro_f1),
        "base_thresholds": [float(x) for x in base_thresholds.tolist()],
        "max_no_drop_thresholds": [float(x) for x in max_no_drop_thresholds.tolist()],
        "thresholds": best_thresholds_list,
        "per_item_accuracy_under_threshold_vector": [float(x) for x in per_item_acc_under_vec.tolist()],
        "per_item_positive_rate": [float(x) for x in pos_rates],
        "overall_elementwise_accuracy_using_threshold_vector": overall_acc,
        "micro_f1_using_threshold_vector": micro_f1_under_vec,
        "macro_f1_using_threshold_vector": macro_f1_under_vec,
        "best_micro_f1_during_optimization": float(best_micro_f1),
        "notes": (
            "Per-criterion thresholds chosen to maximize global micro-F1 with coordinate ascent. "
            f"Tie-break='{tie_break}', score_tolerance={score_tolerance}, "
            f"toward_highest_target={toward_highest_target:.3f}, "
            f"toward_highest_applied={toward_highest_applied:.3f}, "
            f"enforce_no_perf_drop={enforce_no_perf_drop}, "
            f"perf_drop_tolerance={perf_drop_tolerance:.6f}."
        ),
    }
    
    atomic_write_json(out_json, out_data)
    
    print(f"Saved threshold vector JSON -> {out_json.resolve()}")
    print(f"Micro F1 with threshold vector: {micro_f1_under_vec:.4f}")
    print(f"Macro F1 with threshold vector: {macro_f1_under_vec:.4f}")
    print(f"Overall elementwise accuracy with threshold vector: {overall_acc:.4f}")

if __name__ == "__main__":
    main()
