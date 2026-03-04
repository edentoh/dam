import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
import numpy as np

# --- New Modular Imports ---
from dam.core.config import load_config
from dam.data.loaders import DataManager
from dam.modeling.builder import ModelBuilder
from dam.training.engine import Trainer
from dam.training.losses import LossFactory
from dam.training.optimizers import build_optimizer
from dam.utils.io import atomic_write_json, ensure_unique_run_dir
from dam.utils.seeding import seed_everything
from dam.utils.label_stats import compute_and_save_label_stats


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _pick_best_epoch_row(epochs: list[dict], metric_key: str) -> dict | None:
    if not epochs:
        return None
    if metric_key == "val_loss":
        return min(epochs, key=lambda r: _safe_float(r.get("val_loss", float("inf")), float("inf")))
    return max(epochs, key=lambda r: _safe_float(r.get(metric_key, 0.0)))


def summarize_cv(base_dir: Path, fold_dirs: list[Path], cfg: dict, num_folds: int, cv_seed: int):
    """
    Aggregates results from multiple fold directories into a single summary JSON.
    """
    metric_key = cfg["train"].get("metric_for_best", "val_f1_micro")

    fold_rows = []
    for i, fd in enumerate(fold_dirs, start=1):
        best_meta_path = fd / "best_model_metadata.json"
        history_path = fd / "history.json"
        hist = load_json(history_path) if history_path.exists() else {}
        epochs = hist.get("epochs", []) if isinstance(hist, dict) else []

        if best_meta_path.exists():
            best_meta = load_json(best_meta_path)
            best_epoch = int(best_meta.get("best_epoch", 0) or 0)
            metric_for_best = str(best_meta.get("metric_for_best", metric_key))
            epoch_metrics = best_meta.get("epoch_metrics")

            if (not isinstance(epoch_metrics, dict) or not epoch_metrics) and epochs:
                if best_epoch > 0:
                    epoch_metrics = next((r for r in epochs if int(r.get("epoch", 0) or 0) == best_epoch), None)
                if not epoch_metrics:
                    epoch_metrics = _pick_best_epoch_row(epochs, metric_for_best)

            if not isinstance(epoch_metrics, dict):
                epoch_metrics = {}

            fold_rows.append({
                "fold": i,
                "fold_dir": str(fd),
                "best_epoch": int(best_meta.get("best_epoch", epoch_metrics.get("epoch", 0) or 0)),
                "metric_for_best": metric_for_best,
                "best_metric_val": _safe_float(
                    best_meta.get("best_metric_val", epoch_metrics.get(metric_for_best, 0.0))
                ),
                "val_f1_micro": _safe_float(epoch_metrics.get("val_f1_micro", 0.0)),
                "val_f1_macro": _safe_float(epoch_metrics.get("val_f1_macro", 0.0)),
                "val_map_micro": _safe_float(epoch_metrics.get("val_map_micro", 0.0)),
                "val_map_macro": _safe_float(epoch_metrics.get("val_map_macro", 0.0)),
                "val_acc": _safe_float(epoch_metrics.get("val_acc", 0.0)),
                "val_loss": _safe_float(epoch_metrics.get("val_loss", 0.0)),
                "train_loss": _safe_float(epoch_metrics.get("train_loss", 0.0)),
            })
            continue

        # Fallback: reconstruct best epoch from history.json
        if epochs:
            key = metric_key
            best_row = _pick_best_epoch_row(epochs, key)
            if not best_row:
                continue
            fold_rows.append({
                "fold": i,
                "fold_dir": str(fd),
                "best_epoch": int(best_row.get("epoch", 0) or 0),
                "metric_for_best": str(metric_key),
                "best_metric_val": _safe_float(best_row.get(metric_key, 0.0)),
                "val_f1_micro": _safe_float(best_row.get("val_f1_micro", 0.0)),
                "val_f1_macro": _safe_float(best_row.get("val_f1_macro", 0.0)),
                "val_map_micro": _safe_float(best_row.get("val_map_micro", 0.0)),
                "val_map_macro": _safe_float(best_row.get("val_map_macro", 0.0)),
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
            "val_map_micro": agg("val_map_micro"),
            "val_map_macro": agg("val_map_macro"),
            "val_acc": agg("val_acc"),
            "val_loss": agg("val_loss"),
            "train_loss": agg("train_loss"),
        },
    }

    atomic_write_json(base_dir / "cv_summary.json", summary)
    print(f"\n-> Saved CV summary: {base_dir / 'cv_summary.json'}")


def train_single_run(cfg, run_dir, train_loader, val_loader, train_items, device, fold_info=None):
    """Executes one complete training cycle (setup model -> train -> save)."""
    # 0. Persist training-only label stats (class counts + correlations)
    try:
        compute_and_save_label_stats(train_items, cfg, run_dir)
    except Exception as e:
        print(f"[Stats] Warning: failed to compute label stats: {e}")

    # 1. Build Model (Handles backbone + pose weights)
    model = ModelBuilder.build(cfg, device)

    # 2. Setup Loss & Optimizer
    criterion = LossFactory.get(cfg, train_items=train_items, device=device)
    optimizer = build_optimizer(cfg, model)

    # 3. Setup Scheduler
    train_cfg = cfg.get("train", {})
    scheduler_name = str(train_cfg.get("lr_scheduler", "multistep")).strip().lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("cosine_t_max", train_cfg.get("epochs", 1))),
            eta_min=float(train_cfg.get("cosine_eta_min", 0.0)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_cfg.get("lr_milestones", [15, 30]),
            gamma=float(train_cfg.get("lr_gamma", 0.1)),
        )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        loaders=(train_loader, val_loader),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        run_dir=run_dir,
    )

    print(f"\n=== Starting Run: {run_dir.name} {f'({fold_info})' if fold_info else ''} ===")
    trainer.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_score.toml")
    args = parser.parse_args()

    # Load Config & Seed
    cfg = load_config(args.config)
    system_seed = cfg["system"].get("seed", 42)
    seed_everything(system_seed)

    device = torch.device(cfg["system"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    # Paths
    base_run_name = cfg["system"].get("run_name", "default_run")
    runs_root = Path(cfg["system"].get("runs_dir", "runs"))

    # Init Data Manager
    dm = DataManager(cfg)

    # Check CV Mode
    cv_cfg = cfg.get("train", {}).get("cv", cfg.get("cv", {}))
    if cv_cfg.get("enabled", False):
        num_folds = int(cv_cfg.get("num_runs", 5))
        cv_seed = cv_cfg.get("seed", system_seed)

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
                seed=cv_seed,
            )

            train_single_run(
                cfg,
                fold_dir,
                train_loader,
                val_loader,
                train_items,
                device,
                fold_info=f"Fold {fold+1}/{num_folds}",
            )

        summarize_cv(base_dir, fold_dirs, cfg, num_folds=num_folds, cv_seed=cv_seed)

    else:
        # Fixed Split Mode
        run_dir = ensure_unique_run_dir(runs_root, base_run_name)
        (train_loader, val_loader), train_items = dm.get_fixed_loaders()
        train_single_run(cfg, run_dir, train_loader, val_loader, train_items, device)


if __name__ == "__main__":
    main()
