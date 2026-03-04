import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

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
        train_ds = DAMDataset(train_items, train_tfm)

        train_sampler = self._build_train_sampler(train_items)
        shuffle_train = train_sampler is None

        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=shuffle_train,
            sampler=train_sampler,
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

    def _build_train_sampler(self, train_items):
        sampler_cfg = self.train_cfg.get("sampler", {})
        if not bool(sampler_cfg.get("enabled", False)):
            return None
        if not train_items:
            return None

        ys = [np.asarray(it[1], dtype=np.float32) for it in train_items]
        y = np.stack(ys, axis=0)
        y_bin = (y > 0).astype(np.float32)
        n_images, num_classes = y_bin.shape

        pos_counts = y_bin.sum(axis=0)
        valid = pos_counts > 0

        alpha = float(sampler_cfg.get("alpha", 0.5))
        class_weights = np.ones(num_classes, dtype=np.float32)
        class_weights[valid] = np.power(n_images / pos_counts[valid], alpha)

        agg = str(sampler_cfg.get("aggregation", "max")).strip().lower()
        pos_per_sample = y_bin.sum(axis=1)
        if agg == "mean":
            denom = np.clip(pos_per_sample, 1.0, None)
            sample_weights = (y_bin * class_weights).sum(axis=1) / denom
        else:
            sample_weights = (y_bin * class_weights).max(axis=1)

        sample_weights[pos_per_sample <= 0] = 1.0

        min_weight = float(sampler_cfg.get("min_weight", 1.0))
        max_weight = float(sampler_cfg.get("max_weight", 4.0))
        sample_weights = np.clip(sample_weights, min_weight, max_weight).astype(np.float64)

        if bool(sampler_cfg.get("normalize", True)):
            mean_w = float(sample_weights.mean())
            if mean_w > 0:
                sample_weights = sample_weights / mean_w

        mult = float(sampler_cfg.get("num_samples_multiplier", 1.0))
        num_samples = max(1, int(round(len(train_items) * mult)))
        replacement = bool(sampler_cfg.get("replacement", True))

        print(
            "[Sampler] WeightedRandomSampler enabled | "
            f"alpha={alpha:.3f} agg={agg} min/max=({min_weight:.2f},{max_weight:.2f}) "
            f"num_samples={num_samples} replacement={replacement}"
        )

        return WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=replacement,
        )

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
