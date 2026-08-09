import argparse
import csv
import time
import math
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from timm.data import resolve_data_config, create_transform

# --- Modular Imports ---
from dam.core.config import load_config
from dam.modeling.builder import build_model
from dam.utils.seeding import seed_everything
from dam.utils.io import atomic_write_json, ensure_unique_run_dir
from dam.utils.image import load_rgb_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

class CSVBinarizedImageDataset(Dataset):
    """Dataset for binary classification (Path, 0/1)."""
    def __init__(self, items: List[Tuple[Path, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        im = load_rgb_image(path)
        x = self.transform(im)
        y = torch.tensor([float(y)], dtype=torch.float32)
        return x, y, str(path)


def read_labels_csv(csv_path: Path) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must have header columns: path,label")
        
        # Resolve paths relative to the CSV location or CWD
        root = csv_path.parent
        
        for row in reader:
            p_str = row["path"]
            p = Path(p_str).expanduser()
            
            # If relative, try resolving from project root or CSV dir
            if not p.is_absolute():
                # Try finding it relative to CWD first (standard behavior)
                if p.exists():
                    p = p.resolve()
                # Fallback: relative to CSV folder
                elif (root / p).exists():
                    p = (root / p).resolve()
            
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
    
    import random
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


def load_backbone_init(model: nn.Module, ckpt_path: Path) -> Dict[str, int]:
    """Loads weights from a pretrained 48-class model into this binary model."""
    print(f"[Init] Loading backbone from {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    else:
        state = ckpt

    # strict=False allows loading the backbone while ignoring the head mismatch
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"missing": len(missing), "unexpected": len(unexpected)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    n = 0
    tp = tn = fp = fn = 0
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    for x, y, _ in loader:
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
    }


def main():
    parser = argparse.ArgumentParser(description="Train Binary 'Is-DAM' Gate Model")
    parser.add_argument("--config", default="configs/config_gate.toml", help="Path to gating config")
    args = parser.parse_args()

    # 1. Load Configuration
    cfg = load_config(args.config)
    
    sys_cfg = cfg.get("system", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    seed = int(sys_cfg.get("seed", 1337))
    seed_everything(seed)

    # 2. Setup Directories
    runs_root = Path(sys_cfg.get("runs_dir", "runs"))
    run_name = sys_cfg.get("run_name", "is_dam_v1")
    out_dir = ensure_unique_run_dir(runs_root, run_name)
    
    print(f"-> Output directory: {out_dir}")
    atomic_write_json(out_dir / "train_config.json", cfg)

    device = torch.device(sys_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"-> Device: {device}")

    # 3. Data Preparation
    labels_csv = Path(train_cfg.get("labels_csv", "labels_gate/binary_labels.csv"))
    items = read_labels_csv(labels_csv)
    
    val_frac = float(train_cfg.get("val_frac", 0.2))
    train_items, val_items = stratified_split(items, val_frac, seed)
    
    print(f"-> Data: {len(train_items)} Train | {len(val_items)} Val")

    # 4. Model Building
    backbone = model_cfg.get("backbone", "convnextv2_tiny")
    pretrained = bool(model_cfg.get("pretrained", True))
    
    # We use the shared builder but force num_classes=1 for binary
    model = build_model(backbone, num_classes=1, pretrained=pretrained)
    
    # Optional initialization from your main DAM model
    init_from = model_cfg.get("init_from", "")
    if init_from:
        rep = load_backbone_init(model, Path(init_from))
        print(f"-> Initialized backbone: {rep}")

    model.to(device)

    # 5. Transforms & Loaders
    img_size = int(train_cfg.get("img_size", 224))
    
    # Using timm's default config for the backbone ensures correct mean/std/crop
    data_config = resolve_data_config({}, model=model)
    data_config["input_size"] = (3, img_size, img_size)

    train_tf = create_transform(**data_config, is_training=True)
    val_tf = create_transform(**data_config, is_training=False)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))

    sampler = None
    shuffle = True
    if bool(train_cfg.get("balanced_sampler", False)):
        n_pos = sum(1 for _, y in train_items if y == 1)
        n_neg = sum(1 for _, y in train_items if y == 0)
        w_pos = 1.0 / max(1, n_pos)
        w_neg = 1.0 / max(1, n_neg)
        weights = [w_pos if y == 1 else w_neg for _, y in train_items]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        CSVBinarizedImageDataset(train_items, train_tf),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        CSVBinarizedImageDataset(val_items, val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 6. Optimization
    pos_weight = None
    if bool(train_cfg.get("use_pos_weight", False)):
        n_pos = sum(1 for _, y in train_items if y == 1)
        n_neg = sum(1 for _, y in train_items if y == 0)
        pw = (n_neg / max(1, n_pos))
        pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)
        print(f"-> Using pos_weight: {pw:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    lr = float(train_cfg.get("learning_rate", 2e-4))
    wd = float(train_cfg.get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    use_amp = bool(train_cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    # 7. Training Loop
    epochs = int(train_cfg.get("epochs", 10))
    best_metric_name = train_cfg.get("best_metric", "f1")
    best_score = -float("inf") if best_metric_name == "f1" else float("inf")
    patience = int(train_cfg.get("patience", 0))
    
    threshold = float(train_cfg.get("threshold", 0.5))
    bad_epochs = 0
    log_path = out_dir / "train_log.jsonl"

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0

        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = y.numel()
            running_loss += float(loss.item()) * bs
            seen += bs

        train_loss = running_loss / max(1, seen)
        val_res = evaluate(model, val_loader, device, threshold)
        dt = time.time() - t0

        # Log
        row = {
            "epoch": epoch, 
            "train_loss": train_loss, 
            "val_loss": val_res['loss'],
            "val_f1": val_res['f1'],
            "val_acc": val_res['acc'],
            "seconds": dt
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(import_json_dumps := str(row).replace("'", '"') + "\n") # Quick json dump replacement

        print(f"Ep {epoch:02d} | T_Loss {train_loss:.4f} | V_Loss {val_res['loss']:.4f} | V_F1 {val_res['f1']:.4f} | V_Acc {val_res['acc']:.4f}")

        # Save Best
        is_best = (val_res["f1"] > best_score) if best_metric_name == "f1" else (val_res["loss"] < best_score)
        
        state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": cfg,
            "val_metrics": val_res
        }
        
        torch.save(state_dict, out_dir / "last.pt")
        
        if is_best:
            best_score = val_res["f1"] if best_metric_name == "f1" else val_res["loss"]
            bad_epochs = 0
            torch.save(state_dict, out_dir / "best.pt")
            print("  --> New Best!")
        else:
            bad_epochs += 1

        if patience > 0 and bad_epochs >= patience:
            print(f"Early stopping triggered after {bad_epochs} epochs without improvement.")
            break

    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")

if __name__ == "__main__":
    main()
