import os
import re
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
        self.threshold = threshold
        self.pad = pad
        self.min_size = min_size

    def __call__(self, img: Image.Image) -> Image.Image:
        g = img.convert("L")
        arr = np.array(g)
        mask = arr < self.threshold
        if mask.sum() < self.min_size:
            return img
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
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
def elementwise_accuracy(y_true: torch.Tensor, y_prob: torch.Tensor, thr: float) -> float:
    y_pred = (y_prob >= thr).float()
    return (y_pred.eq(y_true).float().mean().item())


@torch.no_grad()
def micro_f1(y_true: torch.Tensor, y_prob: torch.Tensor, thr: float) -> float:
    y_pred = (y_prob >= thr).int()
    y_true_i = y_true.int()

    tp = (y_pred & y_true_i).sum().item()
    fp = (y_pred & (1 - y_true_i)).sum().item()
    fn = ((1 - y_pred) & y_true_i).sum().item()
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def main():
    cfg = load_config("config.toml")

    device_pref = cfg["system"].get("device", "cuda")
    device = torch.device(device_pref if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    backbone = cfg["model"]["backbone"]
    num_classes = int(cfg["model"]["num_classes"])

    model_path = Path(cfg["predict"]["model_path"])
    input_dir = Path(cfg["predict"]["input_image_dir"])
    out_excel = Path(cfg["predict"]["output_excel"])
    thr = float(cfg["predict"].get("threshold", 0.5))

    label_file = Path(cfg["data"]["csv_path"])
    label_map = load_labels(label_file)  # may be empty if not found

    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    img_size = int(ckpt.get("img_size", cfg["data"].get("img_size", 384)))

    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        # CropToInk(threshold=245, pad=12),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    ])

    images = list_images(input_dir)
    items = [(img_id, images[img_id]) for img_id in sorted(images.keys())]
    print(f"Predicting {len(items)} images from: {input_dir}")

    # Windows safety
    num_workers = int(cfg["data"].get("num_workers", 0))
    if os.name == "nt" and num_workers > 0:
        num_workers = 0

    loader = DataLoader(InferDataset(items, tfm), batch_size=16, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    probs_by_id: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            img_ids, x = batch
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob = torch.sigmoid(logits).detach().cpu().numpy()  # (B,48)
            for i, img_id in enumerate(img_ids):
                probs_by_id[img_id] = prob[i].astype(np.float32)

        # Build output tables (rows=48 items, cols="Image 001"...)
    ordered_ids = sorted(probs_by_id.keys())
    cols = [f"Image {img_id}" for img_id in ordered_ids]

    pred_bin = np.stack(
        [(probs_by_id[img_id] >= thr).astype(np.float32) for img_id in ordered_ids],
        axis=1
    )  # (48, N)

    pred_prob = np.stack(
        [probs_by_id[img_id] for img_id in ordered_ids],
        axis=1
    )  # (48, N)

    df_pred = pd.DataFrame(pred_bin, columns=cols)
    df_prob = pd.DataFrame(pred_prob, columns=cols)

    df_pred.insert(0, "Item", [f"Item {i}" for i in range(1, 49)])
    df_prob.insert(0, "Item", [f"Item {i}" for i in range(1, 49)])

    # ---- Add Total score row per image (sum across 48 items)
    totals_01 = pred_bin.sum(axis=0)  # (N,)
    total_row_pred = pd.DataFrame([[ "Total", *totals_01.tolist() ]], columns=["Item", *cols])
    df_pred = pd.concat([df_pred, total_row_pred], ignore_index=True)

    # Optional but useful: sum of probabilities per image
    totals_prob = pred_prob.sum(axis=0)  # (N,)
    total_row_prob = pd.DataFrame([[ "Total_prob", *totals_prob.tolist() ]], columns=["Item", *cols])
    df_prob = pd.concat([df_prob, total_row_prob], ignore_index=True)

    # ---- Metrics (overall + per-item accuracy) if labels exist
    metrics_rows = []
    per_item_rows = []
    common_ids = sorted(set(probs_by_id.keys()) & set(label_map.keys()))

    metrics_rows.append(["threshold", thr])
    metrics_rows.append(["num_pred_images", len(probs_by_id)])
    metrics_rows.append(["num_labeled_images", len(common_ids)])

    if common_ids:
        # Build aligned arrays: (N,48) then transpose when needed
        y_true = np.stack([label_map[i] for i in common_ids], axis=0)  # (N,48)
        y_prob = np.stack([probs_by_id[i] for i in common_ids], axis=0)  # (N,48)

        y_true_t = torch.tensor(y_true, dtype=torch.float32)
        y_prob_t = torch.tensor(y_prob, dtype=torch.float32)

        acc_overall = elementwise_accuracy(y_true_t, y_prob_t, thr=thr)
        f1_overall = micro_f1(y_true_t, y_prob_t, thr=thr)

        print(f"Test/Folder metrics on {len(common_ids)} labeled images @thr={thr}:")
        print(f"  Elementwise accuracy: {acc_overall:.4f}")
        print(f"  Micro F1:             {f1_overall:.4f}")

        metrics_rows.append(["elementwise_accuracy_overall", acc_overall])
        metrics_rows.append(["micro_f1_overall", f1_overall])

        # Per-item accuracy: for each of 48 items, compare across N images
        y_pred = (y_prob >= thr).astype(np.float32)  # (N,48)
        for k in range(48):
            item_acc = float((y_pred[:, k] == y_true[:, k]).mean())
            per_item_rows.append([f"Item {k+1}", item_acc])

    else:
        print("No matching labels found for this input folder; exporting predictions only.")

    df_metrics_summary = pd.DataFrame(metrics_rows, columns=["metric", "value"])
    df_metrics_per_item = pd.DataFrame(per_item_rows, columns=["item", "accuracy"])

    out_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_pred.to_excel(writer, index=False, sheet_name="Predictions_0_1")
        df_prob.to_excel(writer, index=False, sheet_name="Probabilities_0_1")
        df_metrics_summary.to_excel(writer, index=False, sheet_name="Metrics_Summary")
        df_metrics_per_item.to_excel(writer, index=False, sheet_name="Metrics_PerItem")

    print(f"Saved: {out_excel}")


if __name__ == "__main__":
    main()