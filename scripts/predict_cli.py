import argparse
import numpy as np
import torch
from pathlib import Path

# --- Modular Imports ---
from dam.core.config import load_config
from dam.inference.predictor import DAMPredictor
from dam.inference.export import save_predictions_to_excel
from dam.inference.utils import list_images, load_labels_lenient, get_labels_path_for_predict
from dam.inference.thresholds import resolve_under_model_dir, load_threshold_vector

def calculate_and_print_metrics(probs_by_id, label_map, thr_vec):
    """Computes Elementwise Accuracy and Micro/Macro F1 if labels exist."""
    common_ids = sorted(set(probs_by_id.keys()) & set(label_map.keys()))
    
    if not common_ids:
        print("\n[Evaluation] No matching labels found. Skipping metrics.")
        return

    print(f"\n[Evaluation] Found labels for {len(common_ids)} images. Calculating metrics...")

    # Stack: (N, 48)
    y_true = np.stack([label_map[i] for i in common_ids], axis=0).astype(int)
    y_prob = np.stack([probs_by_id[i] for i in common_ids], axis=0)
    
    # Broadcast comparison
    y_pred = (y_prob >= thr_vec.reshape(1, -1)).astype(int)

    # 1. Element-wise Accuracy
    acc = (y_pred == y_true).mean()

    # 2. Micro F1
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    micro_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)

    # 3. Macro F1
    tp_c = np.sum((y_pred == 1) & (y_true == 1), axis=0)
    fp_c = np.sum((y_pred == 1) & (y_true == 0), axis=0)
    fn_c = np.sum((y_pred == 0) & (y_true == 1), axis=0)
    denom = (2 * tp_c + fp_c + fn_c + 1e-9)
    macro_f1 = np.mean((2 * tp_c) / denom)

    print(f"-> Accuracy (Elem-wise): {acc:.4f}")
    print(f"-> Micro F1:             {micro_f1:.4f}")
    print(f"-> Macro F1:             {macro_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="DAM Batch Predictor (CLI)")
    parser.add_argument("--config", default="configs/config_score.toml", help="Path to config.toml")
    args = parser.parse_args()

    # 1. Load Config
    cfg = load_config(args.config)
    
    device_pref = cfg.get("system", {}).get("device", "cuda")
    device = torch.device(device_pref if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    predict_cfg = cfg.get("predict", {})
    if not predict_cfg:
        raise ValueError("Config missing [predict] section.")

    model_path = Path(predict_cfg["model_path"])
    input_dir = Path(predict_cfg["input_image_dir"])
    out_excel = Path(predict_cfg.get("output_excel", "DAM_Predictions.xlsx"))

    # 2. Setup Thresholds
    thr_scalar = float(predict_cfg.get("threshold_scalar_fallback", predict_cfg.get("threshold", 0.5)))
    thr_vec_path_cfg = Path(predict_cfg.get("threshold_vector_path", "threshold_vector.json"))
    thr_vec_path = resolve_under_model_dir(model_path, thr_vec_path_cfg)

    num_classes = int(cfg["model"].get("num_classes", 48))
    thr_vec, thr_info = load_threshold_vector(thr_vec_path, num_classes=num_classes, fallback_thr=thr_scalar)

    # Strict threshold check
    if bool(predict_cfg.get("require_threshold_vector", False)) and thr_info["threshold_mode"].startswith("scalar_fallback"):
        raise RuntimeError(f"require_threshold_vector=true but vector thresholds not available at {thr_vec_path}")

    # 3. List Images
    images_map = list_images(input_dir)
    items = [(img_id, path) for img_id, path in sorted(images_map.items())]
    print(f"Predicting {len(items)} images from: {input_dir}")
    print(f"Threshold mode: {thr_info['threshold_mode']} | path={thr_info['threshold_vector_path']}")

    # 4. Initialize Predictor & Run Batch
    predictor = DAMPredictor(model_path, cfg, device)
    probs_by_id = predictor.predict_batch(items)

    # 5. Load Labels (Optional)
    labels_path = get_labels_path_for_predict(cfg)
    label_map = load_labels_lenient(Path(labels_path)) if labels_path else {}

    # 6. Calculate & Print Metrics
    calculate_and_print_metrics(probs_by_id, label_map, thr_vec)

    # 7. Export Results
    save_predictions_to_excel(
        probs_by_id=probs_by_id,
        label_map=label_map,
        thr_vec=thr_vec,
        thr_info=thr_info,
        out_path=out_excel
    )

if __name__ == "__main__":
    main()
