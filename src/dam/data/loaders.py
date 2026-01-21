import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# Internal imports
from dam.utils.identifiers import extract_id  # Assumed to be created in the utils step
from .datasets import DAMDataset
from .transforms import build_transforms

class DataManager:
    """
    Loads labels + builds train/val dataloaders.
    Handles both Fixed Split and Cross-Validation modes.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.train_cfg = cfg.get("train", {})
        
        # Resolve data config (prefer [train.data], fall back to [data])
        self.data_cfg = self.train_cfg.get("data", {})
        if not self.data_cfg and "data" in cfg:
            self.data_cfg = cfg["data"]

        # Load Labels
        labels_path = self.data_cfg.get("labels_path", self.data_cfg.get("csv_path"))
        if labels_path is None:
            raise KeyError("Missing labels_path configuration (expected [train.data].labels_path)")
        
        self.label_map = self._load_labels(labels_path)

        # Set Root Directory
        img_root = self.data_cfg.get("img_root_dir")
        if img_root is None:
            raise KeyError("Missing img_root_dir configuration (expected [train.data].img_root_dir)")
        self.root_dir = Path(img_root)

    def _load_labels(self, path) -> dict:
        """Reads Excel/CSV and maps Image ID -> 48-float array."""
        # Use openpyxl for xlsx, standard pandas for csv
        path_str = str(path)
        if path_str.endswith(".xlsx") or path_str.endswith(".xls"):
            df = pd.read_excel(path, engine="openpyxl")
        else:
            df = pd.read_csv(path)

        # Identify columns containing "Image"
        image_cols = [c for c in df.columns if isinstance(c, str) and "image" in c.lower()]
        
        # Strict requirement: first 48 rows are the criteria
        df_criteria = df.iloc[:48].copy()
        
        label_map = {}
        for col in image_cols:
            img_id = extract_id(col)
            if not img_id:
                continue
            
            # Convert column to float array, coerce errors to 0
            y = pd.to_numeric(df_criteria[col], errors="coerce").to_numpy(dtype=np.float32)
            y = np.nan_to_num(y, nan=0.0)
            
            if y.shape[0] == 48:
                label_map[img_id] = y
                
        return label_map

    def _find_images(self, folder: Path) -> list:
        """Collects labeled images from a folder."""
        items = []
        if not folder.exists():
            return items

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

        for p in folder.rglob("*"):
            if (not p.is_file()) or (p.suffix.lower() not in exts):
                continue

            img_id = extract_id(p.name)
            # Only include image if we have a label for it
            if img_id and img_id in self.label_map:
                items.append((str(p), self.label_map[img_id], img_id))

        return items

    def _create_dataloaders(self, train_items, val_items):
        bs = int(self.train_cfg.get("batch_size", 16))
        nw = int(self.data_cfg.get("num_workers", 0))

        # Build transforms
        train_tfm = build_transforms(self.cfg, is_train=True)
        val_tfm = build_transforms(self.cfg, is_train=False)

        train_loader = DataLoader(
            DAMDataset(train_items, train_tfm),
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
        )
        val_loader = DataLoader(
            DAMDataset(val_items, val_tfm),
            batch_size=bs,
            shuffle=False,
            num_workers=0, # Validation usually strictly sequential/safer with 0 workers on some OS
            pin_memory=True,
        )
        return train_loader, val_loader

    def get_fixed_loaders(self):
        """
        Standard mode: expects 'train' and 'val' subfolders in img_root_dir.
        """
        train_items = self._find_images(self.root_dir / "train")
        val_items = self._find_images(self.root_dir / "val")
        print(f"[Data] Fixed Mode: {len(train_items)} Train, {len(val_items)} Val")
        
        if not train_items:
             print(f"[Warning] No training images found in {self.root_dir}/train")
        
        return self._create_dataloaders(train_items, val_items), train_items

    def get_cv_loaders(self, fold_idx, num_folds, seed=42):
        """
        Cross-Validation mode: Merges 'train' and 'val' folders, then splits by index.
        """
        all_items = self._find_images(self.root_dir / "train") + self._find_images(self.root_dir / "val")

        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_items))
        rng.shuffle(indices)

        folds = np.array_split(indices, num_folds)
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(num_folds) if i != fold_idx])

        train_items = [all_items[i] for i in train_idx]
        val_items = [all_items[i] for i in val_idx]

        print(f"[Data] CV Fold {fold_idx+1}/{num_folds}: {len(train_items)} Train, {len(val_items)} Val")
        return self._create_dataloaders(train_items, val_items), train_items