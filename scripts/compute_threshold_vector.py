import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

from dam.config import load_config
from dam.inference import (
    InferDataset,
    build_default_timm_transforms,
    build_model,
    infer_in_channels,
    list_images,
    load_labels_from_excel_strict,
    safe_num_workers,
)
from dam.predicting import get_labels_path_for_predict, get_predict_data_cfg
from dam.thresholds import resolve_under_model_dir
from dam.transforms import CropToInk


# -------------------------
# Threshold fitting
# -------------------------

def best_threshold_for_accuracy(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    """Returns (best_threshold, best_accuracy) for a single criterion.

    Tie-break: choose threshold closest to 0.5 among best.
    """
    best_acc = -1.0
    best_ts = []
    for t in grid:
        y_pred = (y_prob >= t).astype(np.int32)
        acc = float((y_pred == y_true).mean())
        if acc > best_acc + 1e-12:
            best_acc = acc
            best_ts = [float(t)]
        elif abs(acc - best_acc) <= 1e-12:
            best_ts.append(float(t))

    # tie-break: closest to 0.5
    best_t = min(best_ts, key=lambda x: abs(x - 0.5))
    return best_t, best_acc


def main():
    cfg = load_config("config.toml")

    # Predict/App settings
    predict_cfg = cfg.get("predict", {})
    data_cfg = get_predict_data_cfg(cfg)

    model_path = Path(predict_cfg["model_path"])
    input_dir = Path(predict_cfg["input_image_dir"])

    out_json_cfg = Path(predict_cfg.get("threshold_vector_path", "threshold_vector.json"))
    out_json = resolve_under_model_dir(model_path, out_json_cfg)

    # Data/Model settings
    labels_path = get_labels_path_for_predict(cfg)
    if not labels_path:
        raise KeyError("Missing labels_path (expected [predict.labels].labels_path)")
    label_file = Path(labels_path)

    backbone = cfg["model"]["backbone"]
    num_classes = int(cfg["model"].get("num_classes", 48))
    timm_pretrained = bool(cfg["model"].get("pretrained", True))

    device_pref = cfg.get("system", {}).get("device", "cuda")
    device = torch.device(device_pref if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # Load labels + images
    label_map = load_labels_from_excel_strict(label_file)
    images = list_images(input_dir)

    common_ids = sorted(set(images.keys()) & set(label_map.keys()))
    if not common_ids:
        raise RuntimeError("No matching IDs between images in predict.input_image_dir and label spreadsheet columns.")

    items = [(img_id, images[img_id]) for img_id in common_ids]
    print(f"Labeled images found for threshold fitting: {len(items)}")

    # Load checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path.resolve()}")
    ckpt = torch.load(model_path, map_location="cpu")
    img_size = int(ckpt.get("img_size", data_cfg.get("img_size", 420)))

    # Build model and load weights
    model = build_model(backbone, num_classes=num_classes, pretrained=timm_pretrained)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Build transforms (match model input channels)
    in_ch = infer_in_channels(model)

    use_crop = bool(data_cfg.get("use_crop_to_ink", False))
    crop_threshold = int(data_cfg.get("crop_threshold", 245))
    crop_pad = int(data_cfg.get("crop_pad", 12))
    crop_min_size = int(data_cfg.get("crop_min_size", 50))

    crop_op = CropToInk(threshold=crop_threshold, pad=crop_pad, min_size=crop_min_size)
    tfm = build_default_timm_transforms(img_size, use_crop=use_crop, crop_to_ink=crop_op, in_ch=in_ch)

    # DataLoader
    num_workers = safe_num_workers(int(data_cfg.get("num_workers", 0)))

    loader = DataLoader(
        InferDataset(items, tfm),
        batch_size=int(predict_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Inference: probs per image
    probs = np.zeros((len(items), num_classes), dtype=np.float32)
    y_true = np.zeros((len(items), num_classes), dtype=np.float32)

    with torch.no_grad():
        offset = 0
        for img_ids, x in loader:
            bsz = x.shape[0]
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            probs[offset : offset + bsz] = p

            for i, img_id in enumerate(img_ids):
                y_true[offset + i] = label_map[str(img_id)]

            offset += bsz

    # Fit thresholds per item to maximize accuracy
    grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)  # step=0.01

    thresholds = []
    per_item_best_acc = []
    per_item_pos_rate = []
    for j in range(num_classes):
        yt = y_true[:, j].astype(np.int32)
        yp = probs[:, j]
        t, acc = best_threshold_for_accuracy(yt, yp, grid)
        thresholds.append(float(t))
        per_item_best_acc.append(float(acc))
        per_item_pos_rate.append(float(yt.mean()))

    thr_vec = np.array(thresholds, dtype=np.float32)

    # Overall metrics using per-item thresholds
    y_pred = (probs >= thr_vec.reshape(1, -1)).astype(np.int32)
    overall_elementwise_acc = float((y_pred == y_true.astype(np.int32)).mean())

    # Also compute per-item accuracy under the chosen thresholds (should match best_acc for each item)
    per_item_acc_under_vec = (y_pred == y_true.astype(np.int32)).mean(axis=0).astype(np.float32)

    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_model_path": str(model_path),
        "input_image_dir": str(input_dir),
        "label_file": str(label_file),
        "backbone": str(backbone),
        "img_size": int(img_size),
        "num_images_used": int(len(items)),
        "threshold_grid_step": 0.01,
        "thresholds": [float(x) for x in thresholds],  # length 48
        "per_item_best_accuracy": [float(x) for x in per_item_best_acc],
        "per_item_accuracy_under_threshold_vector": [float(x) for x in per_item_acc_under_vec.tolist()],
        "per_item_positive_rate": [float(x) for x in per_item_pos_rate],
        "overall_elementwise_accuracy_using_threshold_vector": overall_elementwise_acc,
        "notes": "Per-criterion thresholds chosen to maximize per-criterion accuracy on the labeled images in predict.input_image_dir. Tie-break picks threshold closest to 0.5.",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved threshold vector JSON -> {out_json.resolve()}")
    print(f"Overall elementwise accuracy with per-item thresholds: {overall_elementwise_acc:.4f}")


if __name__ == "__main__":
    main()
