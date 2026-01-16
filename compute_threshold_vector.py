import os
import re
import json
from pathlib import Path
from datetime import datetime

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


# -------------------------
# Utilities
# -------------------------
def extract_id(s: str) -> str | None:
    m = re.search(r"(\d{3})", str(s))
    return m.group(1) if m else None


def load_config(path: str = "config.toml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p.resolve()}")
    with open(p, "rb") as f:
        return toml.load(f)


class CropToInk:
    def __init__(self, threshold: int = 245, pad: int = 12, min_size: int = 50):
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


def load_labels_from_excel(excel_path: Path) -> dict[str, np.ndarray]:
    if not excel_path.exists():
        raise FileNotFoundError(f"Label file not found: {excel_path.resolve()}")

    df = pd.read_excel(excel_path, engine="openpyxl")

    # First 48 rows = criteria
    df_criteria = df.iloc[:48].copy()

    image_cols = [c for c in df_criteria.columns if isinstance(c, str) and "image" in c.lower()]
    if not image_cols:
        raise RuntimeError("No columns containing 'Image' found in labels spreadsheet headers.")

    label_map: dict[str, np.ndarray] = {}
    for col in image_cols:
        img_id = extract_id(col)
        if not img_id:
            continue
        y = pd.to_numeric(df_criteria[col], errors="coerce").to_numpy(dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0)
        if y.shape[0] == 48:
            label_map[img_id] = y

    if not label_map:
        raise RuntimeError("Loaded 0 label vectors. Check your Excel headers and first 48 rows.")
    return label_map


def list_images(folder: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not folder.exists():
        raise FileNotFoundError(f"Input image dir not found: {folder.resolve()}")

    out: dict[str, Path] = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            img_id = extract_id(p.name)
            if img_id:
                out[img_id] = p
    if not out:
        raise RuntimeError(f"No images found under: {folder.resolve()}")
    return out


def infer_in_channels(model: nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.in_channels)
    return 3


# -------------------------
# Optional HF backbone support
# -------------------------
def build_model(backbone: str, num_classes: int, pretrained: bool):
    backbone = str(backbone).strip()
    if backbone.lower().startswith("hf:"):
        from transformers import AutoModelForImageClassification

        repo = backbone.split(":", 1)[1].strip()
        hf = AutoModelForImageClassification.from_pretrained(repo, use_safetensors=True)

        # Replace head
        if hasattr(hf, "classifier") and isinstance(hf.classifier, nn.Module):
            in_features = getattr(hf.classifier, "in_features", None)
            if in_features is None and hasattr(hf.classifier, "weight"):
                in_features = hf.classifier.weight.shape[1]
            hf.classifier = nn.Linear(int(in_features), num_classes)
        elif hasattr(hf, "head") and isinstance(hf.head, nn.Module):
            in_features = getattr(hf.head, "in_features", None)
            if in_features is None and hasattr(hf.head, "weight"):
                in_features = hf.head.weight.shape[1]
            hf.head = nn.Linear(int(in_features), num_classes)
        else:
            raise RuntimeError("HF model head not found (expected .classifier or .head).")

        class HFWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(pixel_values=x).logits

        return HFWrapper(hf)

    return timm.create_model(backbone, pretrained=bool(pretrained), num_classes=num_classes)


# -------------------------
# Dataset
# -------------------------
class InferDataset(Dataset):
    def __init__(self, items: list[tuple[str, Path]], tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, path = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        return img_id, x


# -------------------------
# Threshold fitting
# -------------------------
def best_threshold_for_accuracy(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    """
    Returns (best_threshold, best_accuracy) for a single criterion.
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


def _resolve_under_model_dir(model_path: Path, maybe_rel: Path) -> Path:
    """If maybe_rel is relative, resolve it next to model_path."""
    return maybe_rel if maybe_rel.is_absolute() else (model_path.parent / maybe_rel)


def _get_predict_data_cfg(cfg: dict) -> dict:
    pdcfg = cfg.get("predict", {}).get("data", {})
    if pdcfg:
        return pdcfg
    return cfg.get("data", {})  # back-compat


def _get_labels_path_for_predict(cfg: dict) -> str | None:
    pl = cfg.get("predict", {}).get("labels", {})
    if isinstance(pl, dict) and pl.get("labels_path"):
        return pl.get("labels_path")

    tl = cfg.get("train", {}).get("data", {})
    if isinstance(tl, dict) and tl.get("labels_path"):
        return tl.get("labels_path")

    d = cfg.get("data", {})
    if isinstance(d, dict) and d.get("csv_path"):
        return d.get("csv_path")

    return None


def main():
    cfg = load_config("config.toml")

    # Predict/App settings
    predict_cfg = cfg.get("predict", {})
    data_cfg = _get_predict_data_cfg(cfg)

    model_path = Path(predict_cfg["model_path"])
    input_dir = Path(predict_cfg["input_image_dir"])

    out_json_cfg = Path(predict_cfg.get("threshold_vector_path", "threshold_vector.json"))
    out_json = _resolve_under_model_dir(model_path, out_json_cfg)

    # Data/Model settings
    labels_path = _get_labels_path_for_predict(cfg)
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
    label_map = load_labels_from_excel(label_file)
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
    if in_ch == 1:
        to_gray = transforms.Grayscale(num_output_channels=1)
        norm = transforms.Normalize([0.5], [0.5])
    else:
        to_gray = transforms.Grayscale(num_output_channels=3)
        norm = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])

    use_crop = bool(data_cfg.get("use_crop_to_ink", False))
    crop_threshold = int(data_cfg.get("crop_threshold", 245))
    crop_pad = int(data_cfg.get("crop_pad", 12))
    crop_min_size = int(data_cfg.get("crop_min_size", 50))

    tfm = []
    if use_crop:
        tfm.append(CropToInk(threshold=crop_threshold, pad=crop_pad, min_size=crop_min_size))
    tfm.extend([
        to_gray,
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    tfm = transforms.Compose(tfm)

    # DataLoader
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
            probs[offset:offset + bsz] = p

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