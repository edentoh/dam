import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

from dam.config import load_config
from dam.inference import InferDataset, list_images, load_labels_lenient, safe_num_workers
from dam.predicting import get_labels_path_for_predict, get_predict_data_cfg
from dam.thresholds import load_threshold_vector, resolve_under_model_dir
from dam.transforms import CropToInk


@torch.no_grad()
def elementwise_accuracy_vector(y_true: np.ndarray, y_prob: np.ndarray, thr_vec: np.ndarray) -> float:
    """y_true: (N,48) {0,1}
    y_prob: (N,48) in [0,1]
    thr_vec: (48,)
    """
    y_pred = (y_prob >= thr_vec.reshape(1, -1)).astype(np.int32)
    return float((y_pred == y_true.astype(np.int32)).mean())


@torch.no_grad()
def micro_f1_vector(y_true: np.ndarray, y_prob: np.ndarray, thr_vec: np.ndarray) -> float:
    y_pred = (y_prob >= thr_vec.reshape(1, -1)).astype(np.int32)
    y_true_i = y_true.astype(np.int32)

    tp = int((y_pred & y_true_i).sum())
    fp = int((y_pred & (1 - y_true_i)).sum())
    fn = int(((1 - y_pred) & y_true_i).sum())
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def main():
    cfg = load_config("config.toml")

    device_pref = cfg.get("system", {}).get("device", "cuda")
    device = torch.device(device_pref if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    backbone = cfg["model"]["backbone"]
    num_classes = int(cfg["model"].get("num_classes", 48))

    predict_cfg = cfg.get("predict", {})
    data_cfg = get_predict_data_cfg(cfg)

    model_path = Path(predict_cfg["model_path"])
    input_dir = Path(predict_cfg["input_image_dir"])
    out_excel = Path(predict_cfg.get("output_excel", "DAM_Predictions.xlsx"))

    # Scalar fallback only (used if vector JSON missing/invalid)
    thr_scalar = float(predict_cfg.get("threshold_scalar_fallback", predict_cfg.get("threshold", 0.5)))

    # Vector threshold path (if relative, resolve next to model)
    thr_vec_path_cfg = Path(predict_cfg.get("threshold_vector_path", "threshold_vector.json"))
    thr_vec_path = resolve_under_model_dir(model_path, thr_vec_path_cfg)

    thr_vec, thr_info = load_threshold_vector(thr_vec_path, num_classes=num_classes, fallback_thr=thr_scalar)

    if bool(predict_cfg.get("require_threshold_vector", False)) and thr_info["threshold_mode"].startswith(
        "scalar_fallback"
    ):
        raise RuntimeError(
            f"require_threshold_vector=true but vector thresholds not available. Tried: {thr_vec_path.resolve()}"
        )

    labels_path = get_labels_path_for_predict(cfg)
    label_map = load_labels_lenient(Path(labels_path)) if labels_path else {}

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    img_size = int(ckpt.get("img_size", data_cfg.get("img_size", 384)))

    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)
    model.to(device)
    model.eval()

    # Transforms
    tfm_ops = []
    if bool(data_cfg.get("use_crop_to_ink", False)):
        tfm_ops.append(
            CropToInk(
                threshold=int(data_cfg.get("crop_threshold", 245)),
                pad=int(data_cfg.get("crop_pad", 12)),
                min_size=int(data_cfg.get("crop_min_size", 50)),
            )
        )

    tfm_ops.extend(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tfm = transforms.Compose(tfm_ops)

    images = list_images(input_dir)
    items = [(img_id, images[img_id]) for img_id in sorted(images.keys())]

    print(f"Predicting {len(items)} images from: {input_dir}")
    print(f"Threshold mode: {thr_info['threshold_mode']} | path={thr_info['threshold_vector_path']}")

    num_workers = safe_num_workers(int(data_cfg.get("num_workers", 0)))

    loader = DataLoader(
        InferDataset(items, tfm),
        batch_size=int(predict_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    probs_by_id: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for img_ids, x in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)  # (B,48)
            for i, img_id in enumerate(img_ids):
                probs_by_id[str(img_id)] = prob[i]

    # Build output tables (rows=48 criteria, cols="Image 001"...)
    ordered_ids = sorted(probs_by_id.keys())
    cols = [f"Image {img_id}" for img_id in ordered_ids]

    # pred_prob: (48, N)
    pred_prob = np.stack([probs_by_id[img_id] for img_id in ordered_ids], axis=1)

    # Apply per-item thresholds:
    pred_bin = (pred_prob >= thr_vec.reshape(-1, 1)).astype(np.float32)

    df_pred = pd.DataFrame(pred_bin, columns=cols)
    df_prob = pd.DataFrame(pred_prob, columns=cols)

    df_pred.insert(0, "Item", [f"Item {i}" for i in range(1, 49)])
    df_prob.insert(0, "Item", [f"Item {i}" for i in range(1, 49)])

    # Total score per image (sum across 48 items)
    totals_01 = pred_bin.sum(axis=0)  # (N,)
    total_row_pred = pd.DataFrame([["Total", *totals_01.tolist()]], columns=["Item", *cols])
    df_pred = pd.concat([df_pred, total_row_pred], ignore_index=True)

    # Optional: sum of probabilities per image
    totals_prob = pred_prob.sum(axis=0)  # (N,)
    total_row_prob = pd.DataFrame([["Total_prob", *totals_prob.tolist()]], columns=["Item", *cols])
    df_prob = pd.concat([df_prob, total_row_prob], ignore_index=True)

    # ---- Metrics (overall + per-item accuracy) if labels exist
    metrics_rows = []
    per_item_rows = []
    common_ids = sorted(set(probs_by_id.keys()) & set(label_map.keys()))

    metrics_rows.append(["threshold_mode", thr_info["threshold_mode"]])
    metrics_rows.append(["threshold_vector_path", str(thr_vec_path)])
    metrics_rows.append(["threshold_scalar_fallback", thr_scalar])
    metrics_rows.append(["num_pred_images", len(probs_by_id)])
    metrics_rows.append(["num_labeled_images", len(common_ids)])

    if common_ids:
        # y_true/y_prob aligned: (N,48)
        y_true = np.stack([label_map[i] for i in common_ids], axis=0).astype(np.float32)
        y_prob = np.stack([probs_by_id[i] for i in common_ids], axis=0).astype(np.float32)

        acc_overall = elementwise_accuracy_vector(y_true, y_prob, thr_vec)
        f1_overall = micro_f1_vector(y_true, y_prob, thr_vec)

        print(f"Folder metrics on {len(common_ids)} labeled images (per-item thresholds):")
        print(f"  Elementwise accuracy: {acc_overall:.4f}")
        print(f"  Micro F1:             {f1_overall:.4f}")

        metrics_rows.append(["elementwise_accuracy_overall", acc_overall])
        metrics_rows.append(["micro_f1_overall", f1_overall])

        # Per-item accuracy under per-item thresholds
        y_pred = (y_prob >= thr_vec.reshape(1, -1)).astype(np.int32)  # (N,48)
        y_true_i = y_true.astype(np.int32)

        for k in range(num_classes):
            item_acc = float((y_pred[:, k] == y_true_i[:, k]).mean())
            per_item_rows.append([f"Item {k+1}", float(thr_vec[k]), item_acc])

    else:
        print("No matching labels found for this input folder; exporting predictions only.")

    df_metrics_summary = pd.DataFrame(metrics_rows, columns=["metric", "value"])
    df_metrics_per_item = pd.DataFrame(per_item_rows, columns=["item", "threshold", "accuracy"])

    out_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_pred.to_excel(writer, index=False, sheet_name="Predictions_0_1")
        df_prob.to_excel(writer, index=False, sheet_name="Probabilities_0_1")
        df_metrics_summary.to_excel(writer, index=False, sheet_name="Metrics_Summary")
        df_metrics_per_item.to_excel(writer, index=False, sheet_name="Metrics_PerItem")

    print(f"Saved: {out_excel}")


if __name__ == "__main__":
    main()
