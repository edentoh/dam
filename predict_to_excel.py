import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import tomllib as toml
except ImportError:
    import tomli as toml  # pip install tomli

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


def extract_id(s: str) -> str | None:
    m = re.search(r"(\d{3})", str(s))
    return m.group(1) if m else None


def load_config(path: str = "config.toml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config: {path}")
    with open(path, "rb") as f:
        return toml.load(f)


class CropToInk:
    def __init__(self, threshold: int = 245, pad: int = 10, min_size: int = 50):
        self.threshold = int(threshold)
        self.pad = int(pad)
        self.min_size = int(min_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        g = img.convert("L")
        arr = np.array(g)
        mask = arr < self.threshold
        if int(mask.sum()) < self.min_size:
            return img
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        y0 = max(0, y0 - self.pad)
        x0 = max(0, x0 - self.pad)
        y1 = min(arr.shape[0] - 1, y1 + self.pad)
        x1 = min(arr.shape[1] - 1, x1 + self.pad)
        return img.crop((x0, y0, x1 + 1, y1 + 1))


def load_labels(label_path: Path) -> dict[str, np.ndarray]:
    if not label_path.exists():
        return {}

    if label_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(label_path, engine="openpyxl")
    else:
        df = pd.read_csv(label_path)

    image_cols = [c for c in df.columns if isinstance(c, str) and "image" in c.lower()]
    if not image_cols:
        return {}

    df_criteria = df.iloc[:48].copy()

    label_map = {}
    for col in image_cols:
        img_id = extract_id(col)
        if not img_id:
            continue
        y = pd.to_numeric(df_criteria[col], errors="coerce").to_numpy(dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0)
        if y.shape[0] == 48:
            label_map[img_id] = y
    return label_map


def list_images(folder: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = {}
    if not folder.exists():
        raise FileNotFoundError(f"Input image dir not found: {folder}")
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            img_id = extract_id(p.name)
            if img_id:
                out[img_id] = p
    return out


class InferDataset(Dataset):
    def __init__(self, items: list[tuple[str, Path]], tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, path = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.tfm(img)
        return img_id, img


@torch.no_grad()
def elementwise_accuracy_vector(y_true: np.ndarray, y_prob: np.ndarray, thr_vec: np.ndarray) -> float:
    """
    y_true: (N,48) {0,1}
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


def load_threshold_vector(path: Path, num_classes: int, fallback_thr: float) -> tuple[np.ndarray, dict]:
    """
    Supports:
      - JSON object containing {"thresholds": [...]} (preferred)
      - raw JSON list [...]
    Returns (thr_vec, info_dict)
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
    elif isinstance(obj, dict) and "thresholds" in obj:
        arr = obj["thresholds"]
        info["threshold_mode"] = "vector_from_json.thresholds"
    else:
        raise ValueError(f"Unsupported threshold JSON format in {path}. Expect list or dict with 'thresholds'.")

    if not isinstance(arr, list) or len(arr) != num_classes:
        raise ValueError(f"Threshold vector must be a list of length {num_classes}. Got {type(arr)} len={len(arr) if isinstance(arr, list) else 'NA'}.")

    thr_vec = np.array(arr, dtype=np.float32)
    thr_vec = np.clip(thr_vec, 0.0, 1.0)

    # If it was a raw list, reflect mode:
    if info["threshold_mode"] == "scalar_fallback":
        info["threshold_mode"] = "vector_from_json_list"

    return thr_vec, info



def _resolve_under_model_dir(model_path: Path, maybe_rel: Path) -> Path:
    """If maybe_rel is relative, resolve it next to model_path."""
    return maybe_rel if maybe_rel.is_absolute() else (model_path.parent / maybe_rel)


def _get_predict_data_cfg(cfg: dict) -> dict:
    # New layout: [predict.data]
    pdcfg = cfg.get("predict", {}).get("data", {})
    if pdcfg:
        return pdcfg
    # Back-compat: older layout used [data]
    return cfg.get("data", {})


def _get_labels_path_for_predict(cfg: dict) -> str | None:
    # New layout: [predict.labels].labels_path
    pl = cfg.get("predict", {}).get("labels", {})
    if isinstance(pl, dict) and pl.get("labels_path"):
        return pl.get("labels_path")

    # Reasonable fallback: training labels
    tl = cfg.get("train", {}).get("data", {})
    if isinstance(tl, dict) and tl.get("labels_path"):
        return tl.get("labels_path")

    # Back-compat: older layout used [data].csv_path
    d = cfg.get("data", {})
    if isinstance(d, dict) and d.get("csv_path"):
        return d.get("csv_path")

    return None


def main():
    cfg = load_config("config.toml")

    device_pref = cfg.get("system", {}).get("device", "cuda")
    device = torch.device(device_pref if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    backbone = cfg["model"]["backbone"]
    num_classes = int(cfg["model"].get("num_classes", 48))

    predict_cfg = cfg.get("predict", {})
    data_cfg = _get_predict_data_cfg(cfg)

    model_path = Path(predict_cfg["model_path"])
    input_dir = Path(predict_cfg["input_image_dir"])
    out_excel = Path(predict_cfg.get("output_excel", "DAM_Predictions.xlsx"))

    # Scalar fallback only (used if vector JSON missing/invalid)
    thr_scalar = float(predict_cfg.get("threshold_scalar_fallback", predict_cfg.get("threshold", 0.5)))

    # Vector threshold path (if relative, resolve next to model)
    thr_vec_path_cfg = Path(predict_cfg.get("threshold_vector_path", "threshold_vector.json"))
    thr_vec_path = _resolve_under_model_dir(model_path, thr_vec_path_cfg)

    thr_vec, thr_info = load_threshold_vector(thr_vec_path, num_classes=num_classes, fallback_thr=thr_scalar)

    if bool(predict_cfg.get("require_threshold_vector", False)) and thr_info["threshold_mode"].startswith("scalar_fallback"):
        raise RuntimeError(
            f"require_threshold_vector=true but vector thresholds not available. Tried: {thr_vec_path.resolve()}"
        )

    labels_path = _get_labels_path_for_predict(cfg)
    label_map = load_labels(Path(labels_path)) if labels_path else {}

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

    # Windows safety
    num_workers = int(data_cfg.get("num_workers", 0))
    if os.name == "nt" and num_workers > 0:
        num_workers = 0

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
