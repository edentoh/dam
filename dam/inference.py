import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
import timm

from .utils import extract_id


def list_images(folder: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: dict[str, Path] = {}
    if not folder.exists():
        raise FileNotFoundError(f"Input image dir not found: {folder}")
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            img_id = extract_id(p.name)
            if img_id:
                out[img_id] = p
    return out


def load_labels_lenient(label_path: Path) -> dict[str, np.ndarray]:
    """Load labels from CSV or Excel.

    This matches predict_to_excel.py's previous behavior:
      - returns {} if file missing
      - returns {} if no image columns found
      - uses first 48 rows
    """
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

    label_map: dict[str, np.ndarray] = {}
    for col in image_cols:
        img_id = extract_id(col)
        if not img_id:
            continue
        y = pd.to_numeric(df_criteria[col], errors="coerce").to_numpy(dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0)
        if y.shape[0] == 48:
            label_map[img_id] = y
    return label_map


def load_labels_from_excel_strict(excel_path: Path) -> dict[str, np.ndarray]:
    """Load labels from an Excel file; strict behavior used for threshold fitting."""
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


def infer_in_channels(model: nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.in_channels)
    return 3


def build_model(backbone: str, num_classes: int, pretrained: bool):
    """Build a model for inference.

    - If backbone starts with 'hf:', loads transformers AutoModelForImageClassification
    - Else uses timm.create_model

    This mirrors compute_threshold_vector.py.
    """
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


def build_default_timm_transforms(img_size: int, *, use_crop: bool, crop_to_ink, in_ch: int = 3):
    """Helper to build grayscale+resize+normalize transforms consistent with your scripts."""
    if in_ch == 1:
        to_gray = transforms.Grayscale(num_output_channels=1)
        norm = transforms.Normalize([0.5], [0.5])
    else:
        to_gray = transforms.Grayscale(num_output_channels=3)
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    tfm_ops = []
    if use_crop:
        tfm_ops.append(crop_to_ink)

    tfm_ops.extend(
        [
            to_gray,
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm,
        ]
    )

    return transforms.Compose(tfm_ops)


def safe_num_workers(num_workers: int) -> int:
    """Match existing Windows safety for DataLoader workers."""
    num_workers = int(num_workers)
    if os.name == "nt" and num_workers > 0:
        return 0
    return num_workers
