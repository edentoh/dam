import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional
from dam.utils.identifiers import extract_id

def list_images(folder: Path) -> Dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: Dict[str, Path] = {}
    if not folder.exists():
        raise FileNotFoundError(f"Input image dir not found: {folder}")
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            img_id = extract_id(p.name)
            if img_id:
                out[img_id] = p
    return out

def safe_num_workers(num_workers: int) -> int:
    num_workers = int(num_workers)
    if os.name == "nt" and num_workers > 0:
        return 0
    return num_workers

def load_labels_lenient(label_path: Path) -> Dict[str, np.ndarray]:
    """Lenient loader for predictions (ignores missing files/columns)."""
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
        if not img_id: continue
        y = pd.to_numeric(df_criteria[col], errors="coerce").to_numpy(dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0)
        if y.shape[0] == 48:
            label_map[img_id] = y
    return label_map

def load_labels_from_excel_strict(excel_path: Path) -> Dict[str, np.ndarray]:
    """Strict loader for threshold fitting."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Label file not found: {excel_path.resolve()}")

    df = pd.read_excel(excel_path, engine="openpyxl")
    df_criteria = df.iloc[:48].copy()

    image_cols = [c for c in df_criteria.columns if isinstance(c, str) and "image" in c.lower()]
    if not image_cols:
        raise RuntimeError("No columns containing 'Image' found in labels spreadsheet headers.")

    label_map = {}
    for col in image_cols:
        img_id = extract_id(col)
        if not img_id: continue
        y = pd.to_numeric(df_criteria[col], errors="coerce").to_numpy(dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0)
        if y.shape[0] == 48:
            label_map[img_id] = y

    if not label_map:
        raise RuntimeError("Loaded 0 label vectors. Check your Excel headers and first 48 rows.")
    return label_map

def get_labels_path_for_predict(cfg: dict) -> Union[str, None]:
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