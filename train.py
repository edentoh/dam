import argparse
import torch
import json
try:
    import tomllib as toml
except ImportError:
    import tomli as toml
from pathlib import Path
from datetime import datetime

from utils import seed_everything, ensure_unique_run_dir, atomic_write_json
from data import DataManager
from model import ModelBuilder
from loss import LossFactory
from engine import Trainer


def _get_module_by_path(root, path: str):
    """Resolves dotted attribute paths (e.g., 'head.fc') against a module."""
    cur = root
    for part in str(path).split('.'):
        if not part:
            continue
        cur = getattr(cur, part)
    return cur


def _resolve_classifier_module(model):
    """Best-effort resolution of a model's classifier/head module.

    Supports timm's `get_classifier()` conventions:
      - returns an nn.Module
      - returns a dotted string path to the classifier
      - returns a list/tuple of modules or paths
    Returns a list of modules (possibly empty).
    """
    if not hasattr(model, "get_classifier"):
        return []

    cls = model.get_classifier()
    if cls is None:
        return []

    modules = []
    if isinstance(cls, str):
        try:
            modules.append(_get_module_by_path(model, cls))
        except Exception:
            return []
    elif isinstance(cls, (list, tuple)):
        for item in cls:
            if item is None:
                continue
            if isinstance(item, str):
                try:
                    modules.append(_get_module_by_path(model, item))
                except Exception:
                    continue
            else:
                modules.append(item)
    else:
        modules.append(cls)

    # Filter to modules that look like nn.Modules
    out = []
    for m in modules:
        if hasattr(m, "parameters"):
            out.append(m)
    return out


def build_optimizer(cfg: dict, model):
    """Creates AdamW optimizer.

    If `train.use_discriminative_lr=true`, uses param groups:
      - backbone: base_lr * backbone_lr_mult
      - head:     base_lr * head_lr_mult

    Weight-decay filtering (enabled by default):
      - apply `weight_decay` only to params with ndim > 1 and not ending with `.bias`
      - set `weight_decay=0.0` for biases and 1D params (e.g., LayerNorm/BatchNorm weights)
    """
    train_cfg = cfg.get("train", {})
    base_lr = float(train_cfg.get("learning_rate", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    use_wd_filter = bool(train_cfg.get("use_weight_decay_filtering", True))

    def is_no_decay(name: str, p: torch.nn.Parameter) -> bool:
        if not use_wd_filter:
            return False
        if name.endswith(".bias"):
            return True
        # Most normalization weights are 1D; excluding all 1D params is a common, robust rule.
        if getattr(p, "ndim", None) == 1:
            return True
        return False

    use_disc = bool(train_cfg.get("use_discriminative_lr", False))
    if not use_disc:
        # Single LR, but still apply weight-decay filtering if enabled.
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if is_no_decay(n, p) else decay).append(p)

        # If something went wrong, fall back to the simplest optimizer construction.
        if len(decay) == 0 and len(no_decay) == 0:
            return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

        param_groups = []
        if len(decay) > 0:
            param_groups.append({"params": decay, "lr": base_lr, "weight_decay": weight_decay, "name": "all_decay"})
        if len(no_decay) > 0:
            param_groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0, "name": "all_no_decay"})
        return torch.optim.AdamW(param_groups)

    backbone_mult = float(train_cfg.get("backbone_lr_mult", 0.1))
    head_mult = float(train_cfg.get("head_lr_mult", 1.0))

    head_modules = _resolve_classifier_module(model)
    head_params = []
    for m in head_modules:
        head_params.extend(list(m.parameters()))

    head_param_ids = {id(p) for p in head_params}

    # Build backbone/head parameter lists with names so we can apply weight-decay filtering.
    backbone_named, head_named = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_param_ids:
            head_named.append((n, p))
        else:
            backbone_named.append((n, p))

    backbone_params = [p for _, p in backbone_named]
    head_params = [p for _, p in head_named]

    # Fallback safety: if for any reason head params are empty, revert to single-group.
    if len(head_params) == 0 or len(backbone_params) == 0:
        print("[Optimizer] Warning: Could not split backbone/head params reliably. Falling back to single LR.")
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Split each group into (decay / no_decay) to implement weight-decay filtering.
    bb_decay, bb_no_decay = [], []
    for n, p in backbone_named:
        (bb_no_decay if is_no_decay(n, p) else bb_decay).append(p)
    hd_decay, hd_no_decay = [], []
    for n, p in head_named:
        (hd_no_decay if is_no_decay(n, p) else hd_decay).append(p)

    bb_lr = base_lr * backbone_mult
    hd_lr = base_lr * head_mult

    param_groups = []
    if len(bb_decay) > 0:
        param_groups.append({"params": bb_decay, "lr": bb_lr, "weight_decay": weight_decay, "name": "backbone_decay"})
    if len(bb_no_decay) > 0:
        param_groups.append({"params": bb_no_decay, "lr": bb_lr, "weight_decay": 0.0, "name": "backbone_no_decay"})
    if len(hd_decay) > 0:
        param_groups.append({"params": hd_decay, "lr": hd_lr, "weight_decay": weight_decay, "name": "head_decay"})
    if len(hd_no_decay) > 0:
        param_groups.append({"params": hd_no_decay, "lr": hd_lr, "weight_decay": 0.0, "name": "head_no_decay"})

    opt = torch.optim.AdamW(param_groups)

    print(
        f"[Optimizer] Discriminative LR enabled | "
        f"backbone_lr={bb_lr:.2e} (x{backbone_mult}) | "
        f"head_lr={hd_lr:.2e} (x{head_mult}) | "
        f"wd_filter={'on' if use_wd_filter else 'off'}"
    )
    return opt


def load_config(path):
    with open(path, "rb") as f:
        return toml.load(f)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def summarize_cv(base_dir: Path, fold_dirs: list[Path], cfg: dict, num_folds: int, cv_seed: int):
    """Builds and saves a CV summary JSON (mean/std across folds) into base_dir."""
    metric_key = cfg['train'].get('metric_for_best', 'val_f1_micro')

    fold_rows = []
    for i, fd in enumerate(fold_dirs, start=1):
        best_meta_path = fd / "best_model_metadata.json"
        history_path = fd / "history.json"

        if best_meta_path.exists():
            best_meta = load_json(best_meta_path)
            epoch_metrics = best_meta.get("epoch_metrics", {})
            fold_rows.append({
                "fold": i,
                "fold_dir": str(fd),
                "best_epoch": int(best_meta.get("best_epoch", epoch_metrics.get("epoch", 0) or 0)),
                "metric_for_best": str(best_meta.get("metric_for_best", metric_key)),
                "best_metric_val": _safe_float(best_meta.get("best_metric_val", 0.0)),
                "val_f1_micro": _safe_float(epoch_metrics.get("val_f1_micro", 0.0)),
                "val_f1_macro": _safe_float(epoch_metrics.get("val_f1_macro", 0.0)),
                "val_acc": _safe_float(epoch_metrics.get("val_acc", 0.0)),
                "val_loss": _safe_float(epoch_metrics.get("val_loss", 0.0)),
                "train_loss": _safe_float(epoch_metrics.get("train_loss", 0.0)),
            })
            continue

        # Fallback: reconstruct best epoch from history.json
        if history_path.exists():
            hist = load_json(history_path)
            epochs = hist.get("epochs", [])
            if epochs:
                key = metric_key
                best_row = max(epochs, key=lambda r: _safe_float(r.get(key, 0.0)))
                fold_rows.append({
                    "fold": i,
                    "fold_dir": str(fd),
                    "best_epoch": int(best_row.get("epoch", 0) or 0),
                    "metric_for_best": str(metric_key),
                    "best_metric_val": _safe_float(best_row.get(metric_key, 0.0)),
                    "val_f1_micro": _safe_float(best_row.get("val_f1_micro", 0.0)),
                    "val_f1_macro": _safe_float(best_row.get("val_f1_macro", 0.0)),
                    "val_acc": _safe_float(best_row.get("val_acc", 0.0)),
                    "val_loss": _safe_float(best_row.get("val_loss", 0.0)),
                    "train_loss": _safe_float(best_row.get("train_loss", 0.0)),
                })
                continue

        fold_rows.append({
            "fold": i,
            "fold_dir": str(fd),
            "error": "Missing best_model_metadata.json and history.json",
        })

    import numpy as np

    def agg(field: str):
        vals = [r[field] for r in fold_rows if isinstance(r, dict) and field in r and isinstance(r[field], (int, float))]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "n": int(arr.size),
        }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir),
        "num_folds": int(num_folds),
        "cv_seed": int(cv_seed),
        "metric_for_best": str(metric_key),
        "folds": fold_rows,
        "aggregate": {
            "best_epoch": agg("best_epoch"),
            "best_metric_val": agg("best_metric_val"),
            "val_f1_micro": agg("val_f1_micro"),
            "val_f1_macro": agg("val_f1_macro"),
            "val_acc": agg("val_acc"),
            "val_loss": agg("val_loss"),
            "train_loss": agg("train_loss"),
        },
    }

    atomic_write_json(base_dir / "cv_summary.json", summary)
    print(f"\n-> Saved CV summary: {base_dir / 'cv_summary.json'}")


def train_single_run(cfg, run_dir, train_loader, val_loader, train_items, device, fold_info=None):
    """
    Executes one complete training cycle (setup model -> train -> save).
    """
    model = ModelBuilder.build(cfg, device)

    criterion = LossFactory.get(cfg, train_items=train_items, device=device)
    optimizer = build_optimizer(cfg, model)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg['train'].get('lr_milestones', [15, 30]),
        gamma=cfg['train'].get('lr_gamma', 0.1)
    )

    trainer = Trainer(
        model=model,
        loaders=(train_loader, val_loader),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        run_dir=run_dir
    )

    print(f"\n=== Starting Run: {run_dir.name} {f'({fold_info})' if fold_info else ''} ===")
    trainer.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    system_seed = cfg['system'].get('seed', 42)
    seed_everything(system_seed)

    device = torch.device(cfg['system'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    base_run_name = cfg['system'].get('run_name', 'default_run')
    runs_root = Path(cfg['system'].get('runs_dir', 'runs'))

    dm = DataManager(cfg)

    cv_cfg = cfg.get('train', {}).get('cv', cfg.get('cv', {}))
    if cv_cfg.get('enabled', False):
        num_folds = int(cv_cfg.get('num_runs', 5))
        cv_seed = cv_cfg.get('seed', system_seed)

        base_dir = ensure_unique_run_dir(runs_root, f"{base_run_name}_CV")

        print(f"-> Cross-Validation Enabled: {num_folds} folds.")
        print(f"-> CV Shuffle Seed: {cv_seed}")
        print(f"-> Saving to {base_dir}")

        atomic_write_json(base_dir / "cv_config.json", cfg)

        fold_dirs = []
        for fold in range(num_folds):
            fold_dir = base_dir / f"fold_{fold+1}"
            fold_dir.mkdir(exist_ok=True)
            fold_dirs.append(fold_dir)

            (train_loader, val_loader), train_items = dm.get_cv_loaders(
                fold_idx=fold,
                num_folds=num_folds,
                seed=cv_seed
            )

            train_single_run(
                cfg, fold_dir, train_loader, val_loader,
                train_items, device, fold_info=f"Fold {fold+1}/{num_folds}"
            )

        summarize_cv(base_dir, fold_dirs, cfg, num_folds=num_folds, cv_seed=cv_seed)

    else:
        run_dir = ensure_unique_run_dir(runs_root, base_run_name)
        (train_loader, val_loader), train_items = dm.get_fixed_loaders()
        train_single_run(cfg, run_dir, train_loader, val_loader, train_items, device)


if __name__ == "__main__":
    main()
