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
from dam.inference.thresholds import optimize_thresholds_accuracy, resolve_under_model_dir
from dam.utils.io import atomic_write_json
from dam.utils.seeding import seed_everything

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

    # 7. Optimize Thresholds (Maximizing Accuracy, Tie-break 0.5)
    best_thresholds, best_accs, pos_rates, best_thresholds_list = optimize_thresholds_accuracy(
        y_true, 
        y_probs, 
        num_steps=101 # Matches original np.linspace(0, 1, 101)
    )

    # 8. Calculate Overall Metrics
    y_pred = (y_probs >= best_thresholds.reshape(1, -1)).astype(np.int32)
    overall_acc = float((y_pred == y_true.astype(np.int32)).mean())
    
    # Per-item accuracy under vector (should match best_accs)
    per_item_acc_under_vec = (y_pred == y_true.astype(np.int32)).mean(axis=0)

    # 9. Save Output
    out_data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_model_path": str(model_path),
        "input_image_dir": str(input_dir),
        "label_file": str(labels_path),
        "backbone": cfg["model"]["backbone"],
        "img_size": int(predictor.img_size),
        "num_images_used": int(len(items)),
        "threshold_grid_step": 0.01,
        "thresholds": best_thresholds_list,
        "per_item_best_accuracy": [float(x) for x in best_accs],
        "per_item_accuracy_under_threshold_vector": [float(x) for x in per_item_acc_under_vec.tolist()],
        "per_item_positive_rate": [float(x) for x in pos_rates],
        "overall_elementwise_accuracy_using_threshold_vector": overall_acc,
        "notes": "Per-criterion thresholds chosen to maximize per-criterion accuracy. Tie-break picks threshold closest to 0.5.",
    }
    
    atomic_write_json(out_json, out_data)
    
    print(f"Saved threshold vector JSON -> {out_json.resolve()}")
    print(f"Overall elementwise accuracy with per-item thresholds: {overall_acc:.4f}")

if __name__ == "__main__":
    main()