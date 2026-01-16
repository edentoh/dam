import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import extract_id

# ... (Keep CropToInk and DAMDataset classes exactly as they were) ...

class CropToInk:
    """Pre-processing: Crops image to ink bounding box."""
    def __init__(self, threshold: int = 245, pad: int = 12, min_size: int = 50):
        self.threshold = threshold
        self.pad = pad
        self.min_size = min_size

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


class DAMDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y, img_id = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.float32), img_id


class DataManager:
    """Loads labels + builds train/val dataloaders.

    Uses ONLY the training config section:
      - labels_path, img_root_dir, transforms, num_workers from [train.data]
      - batch_size from [train]

    Backwards compatible with the older config layout that used [data].
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.train_cfg = cfg.get("train", {})

        # Prefer the new layout: [train.data]
        self.data_cfg = self.train_cfg.get("data", {})

        # Back-compat: older layout used [data]
        if not self.data_cfg and "data" in cfg:
            self.data_cfg = cfg["data"]

        labels_path = self.data_cfg.get("labels_path", self.data_cfg.get("csv_path"))
        if labels_path is None:
            raise KeyError("Missing labels_path (expected [train.data].labels_path)")

        self.label_map = self._load_labels(labels_path)

        img_root = self.data_cfg.get("img_root_dir")
        if img_root is None:
            raise KeyError("Missing img_root_dir (expected [train.data].img_root_dir)")
        self.root_dir = Path(img_root)

    def _load_labels(self, path) -> dict:
        df = pd.read_excel(path, engine="openpyxl")
        image_cols = [c for c in df.columns if isinstance(c, str) and "image" in c.lower()]
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

    def _find_images(self, folder: Path) -> list:
        """Collect labeled images from folder (no sorting, matches your previous ordering)."""
        items = []
        if not folder.exists():
            return items

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for p in folder.rglob("*"):
            if (not p.is_file()) or (p.suffix.lower() not in exts):
                continue

            img_id = extract_id(p.name)
            if img_id and img_id in self.label_map:
                items.append((str(p), self.label_map[img_id], img_id))

        return items

    def _get_transforms(self, is_train: bool):
        cfg_d = self.data_cfg
        ops = []

        if cfg_d.get("use_crop_to_ink", False):
            ops.append(
                CropToInk(
                    cfg_d.get("crop_threshold", 245),
                    cfg_d.get("crop_pad", 12),
                    cfg_d.get("crop_min_size", 50),
                )
            )

        ops.append(transforms.Grayscale(num_output_channels=3))
        size = int(cfg_d.get("img_size", 384))

        if is_train:
            ops.extend(
                [
                    transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
                    transforms.RandomApply([transforms.ColorJitter(0.35, 0.35)], p=0.9),
                    transforms.RandomAffine(12, (0.06, 0.06), (0.9, 1.1), 6),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            ops.extend(
                [
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        return transforms.Compose(ops)

    def _create_dataloaders(self, train_items, val_items):
        bs = int(self.train_cfg.get("batch_size", 16))
        nw = int(self.data_cfg.get("num_workers", 0))

        train_loader = DataLoader(
            DAMDataset(train_items, self._get_transforms(is_train=True)),
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
        )
        val_loader = DataLoader(
            DAMDataset(val_items, self._get_transforms(is_train=False)),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, val_loader

    def get_fixed_loaders(self):
        """Standard mode: uses img_root_dir/train and img_root_dir/val."""
        train_items = self._find_images(self.root_dir / "train")
        val_items = self._find_images(self.root_dir / "val")
        print(f"[Data] Fixed Mode: {len(train_items)} Train, {len(val_items)} Val")
        return self._create_dataloaders(train_items, val_items), train_items

    def get_cv_loaders(self, fold_idx, num_folds, seed=42):
        """CV mode: merges train+val, shuffles, splits by index."""
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