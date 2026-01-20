#!/usr/bin/env python3
"""
Train a binary 'is_dam' classifier.

Expected labels CSV (header required):
  path,label
  img_dataset/train/foo.jpg,1
  false_images/bar.png,0

Typical usage:
  python scripts/train_is_dam.py --labels labels_gate/binary_labels.csv --out runs/is_dam_v1

Optional:
  --init-from <path_to_48label_ckpt>  (loads backbone weights; ignores head mismatch)
  --use-pos-weight                   (BCEWithLogits pos_weight = neg/pos)
  --balanced-sampler                 (WeightedRandomSampler)

This script is intentionally standalone (does not depend on dam/ internals).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

import timm
from timm.data import resolve_data_config, create_transform


# -------------------------
# Path robustness for "python scripts/..."
# -------------------------
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # usually beneficial for CNNs
    torch.backends.cudnn.deterministic = False


class CSVBinarizedImageDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        # Robust image open
        with Image.open(path) as im:
            im = im.convert("RGB")
        x = self.transform(im)
        # float label for BCE
        y = torch.tensor([float(y)], dtype=torch.float32)
        return x, y, str(path)


def read_labels_csv(csv_path: Path) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must have header columns: path,label")
        for row in reader:
            p = Path(row["path"]).expanduser()
            # allow relative paths relative to project root
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
            if p.suffix.lower() not in IMG_EXTS:
                continue
            try:
                y = int(row["label"])
            except Exception:
                continue
            if y not in (0, 1):
                continue
            if p.exists():
                items.append((p, y))
    if not items:
        raise ValueError(f"No valid image rows found in: {csv_path}")
    return items


def stratified_split(items: List[Tuple[Path, int]], val_frac: float, seed: int) -> Tuple[List, List]:
    pos = [it for it in items if it[1] == 1]
    neg = [it for it in items if it[1] == 0]
    rnd = random.Random(seed)
    rnd.shuffle(pos)
    rnd.shuffle(neg)

    npos_val = max(1, int(round(len(pos) * val_frac))) if len(pos) > 1 else 0
    nneg_val = max(1, int(round(len(neg) * val_frac))) if len(neg) > 1 else 0

    val = pos[:npos_val] + neg[:nneg_val]
    train = pos[npos_val:] + neg[nneg_val:]

    rnd.shuffle(train)
    rnd.shuffle(val)
    return train, val


def build_is_dam_model(backbone: str, pretrained: bool) -> nn.Module:
    # timm: num_classes=1 gives a single logit head
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=1)
    return model


def load_backbone_init(model: nn.Module, ckpt_path: Path) -> Dict[str, int]:
    """
    Load a checkpoint from your 48-label model (or any timm model) and ignore head mismatch.
    Works with checkpoints that store:
      - {"model": state_dict}
      - {"state_dict": state_dict}
      - raw state_dict
    Returns a small report.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # may already be state dict
        state = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")

    # Try direct load with strict=False
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"missing": len(missing), "unexpected": len(unexpected)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    n = 0

    tp = tn = fp = fn = 0

    bce = nn.BCEWithLogitsLoss(reduction="sum")

    for x, y, _paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = bce(logits, y)
        total_loss += float(loss.item())
        n += y.numel()

        probs = torch.sigmoid(logits)
        pred = (probs >= threshold).to(torch.int64)
        yy = (y >= 0.5).to(torch.int64)

        tp += int(((pred == 1) & (yy == 1)).sum().item())
        tn += int(((pred == 0) & (yy == 0)).sum().item())
        fp += int(((pred == 1) & (yy == 0)).sum().item())
        fn += int(((pred == 0) & (yy == 1)).sum().item())

    eps = 1e-9
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(eps, (tp + fp))
    rec = tp / max(eps, (tp + fn))
    f1 = (2 * prec * rec) / max(eps, (prec + rec))

    avg_loss = total_loss / max(1, n)

    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": (tp + tn + fp + fn),
    }


@dataclass
class TrainConfig:
    labels: str
    out: str
    backbone: str = "convnextv2_tiny"
    pretrained: int = 1
    init_from: str = ""
    img_size: int = 420
    batch_size: int = 32
    epochs: int = 10
    lr: float = 0.00001
    weight_decay: float = 0.1
    num_workers: int = 4
    seed: int = 1337
    val_frac: float = 0.2
    use_pos_weight: int = 0
    balanced_sampler: int = 0
    amp: int = 1
    grad_clip: float = 0.0
    threshold: float = 0.5
    best_metric: str = "f1"  # or "loss"
    patience: int = 0        # 0 disables early stopping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="CSV file with columns: path,label (label in {0,1})")
    ap.add_argument("--out", required=True, help="Output run directory (checkpoints, logs)")

    ap.add_argument("--backbone", default="convnextv2_tiny")
    ap.add_argument("--pretrained", type=int, default=1)

    ap.add_argument("--init-from", default="", help="Optional checkpoint to initialize backbone from (e.g., your 48-label model)")

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.2)

    ap.add_argument("--use-pos-weight", action="store_true", help="Use pos_weight = neg/pos in BCEWithLogitsLoss")
    ap.add_argument("--balanced-sampler", action="store_true", help="Use WeightedRandomSampler to balance classes per batch")

    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=0.0)

    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold used for val metrics (sigmoid >= threshold => positive)")
    ap.add_argument("--best-metric", choices=["f1", "loss"], default="f1")
    ap.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs (0 disables)")

    args = ap.parse_args()

    cfg = TrainConfig(
        labels=args.labels,
        out=args.out,
        backbone=args.backbone,
        pretrained=int(args.pretrained),
        init_from=args.init_from,
        img_size=int(args.img_size),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        use_pos_weight=int(bool(args.use_pos_weight)),
        balanced_sampler=int(bool(args.balanced_sampler)),
        amp=int(not args.no_amp),
        grad_clip=float(args.grad_clip),
        threshold=float(args.threshold),
        best_metric=str(args.best_metric),
        patience=int(args.patience),
    )

    set_seed(cfg.seed)

    out_dir = Path(cfg.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    items = read_labels_csv(Path(cfg.labels).expanduser().resolve())
    train_items, val_items = stratified_split(items, cfg.val_frac, cfg.seed)

    n_pos = sum(1 for _p, y in train_items if y == 1)
    n_neg = sum(1 for _p, y in train_items if y == 0)
    print(f"Train size: {len(train_items)} (pos={n_pos}, neg={n_neg})")
    print(f"Val size:   {len(val_items)}")

    model = build_is_dam_model(cfg.backbone, pretrained=bool(cfg.pretrained))
    if cfg.init_from:
        rep = load_backbone_init(model, Path(cfg.init_from).expanduser().resolve())
        print(f"Loaded init_from with strict=False: {rep}")

    model.to(device)

    # Create transforms consistent with timm
    data_cfg = resolve_data_config({}, model=model)
    data_cfg["input_size"] = (3, cfg.img_size, cfg.img_size)

    train_tf = create_transform(**data_cfg, is_training=True)
    val_tf = create_transform(**data_cfg, is_training=False)

    train_ds = CSVBinarizedImageDataset(train_items, transform=train_tf)
    val_ds = CSVBinarizedImageDataset(val_items, transform=val_tf)

    sampler = None
    shuffle = True
    if cfg.balanced_sampler:
        # weights inverse to class counts
        w_pos = 1.0 / max(1, n_pos)
        w_neg = 1.0 / max(1, n_neg)
        weights = [w_pos if y == 1 else w_neg for _p, y in train_items]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    pos_weight = None
    if cfg.use_pos_weight:
        # BCEWithLogitsLoss pos_weight expects a tensor of shape [1] for binary case
        pw = (n_neg / max(1, n_pos))
        pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)
        print(f"Using pos_weight = {pw:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))

    log_path = out_dir / "train_log.jsonl"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    best_score = -float("inf") if cfg.best_metric == "f1" else float("inf")
    bad_epochs = 0

    def is_better(metric: Dict[str, float]) -> bool:
        nonlocal best_score
        if cfg.best_metric == "f1":
            return metric["f1"] > best_score
        # loss
        return metric["loss"] < best_score

    def update_best(metric: Dict[str, float]) -> None:
        nonlocal best_score
        best_score = metric["f1"] if cfg.best_metric == "f1" else metric["loss"]

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        seen = 0

        for x, y, _paths in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg.amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = y.numel()
            running_loss += float(loss.item()) * bs
            seen += bs

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, val_loader, device=device, threshold=cfg.threshold)

        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "seconds": dt,
        }

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"Ep {epoch:03d} | "
            f"T_Loss {train_loss:.4f} | "
            f"V_Loss {val_metrics['loss']:.4f} | "
            f"V_Acc {val_metrics['acc']:.4f} | "
            f"V_F1 {val_metrics['f1']:.4f} | "
            f"({dt:.1f}s)"
        )

        # Save last
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": asdict(cfg),
                "val_metrics": val_metrics,
            },
            str(last_path),
        )

        # Save best
        if is_better(val_metrics):
            update_best(val_metrics)
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "val_metrics": val_metrics,
                },
                str(best_path),
            )
        else:
            bad_epochs += 1

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"Early stopping: no improvement for {cfg.patience} epochs.")
            break

    print(f"Done. Best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
