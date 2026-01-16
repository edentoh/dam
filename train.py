import argparse
import torch
try:
    import tomllib as toml
except ImportError:
    import tomli as toml
from pathlib import Path

from utils import seed_everything, ensure_unique_run_dir, atomic_write_json
from data import DataManager
from model import ModelBuilder
from loss import LossFactory
from engine import Trainer

def load_config(path):
    with open(path, "rb") as f:
        return toml.load(f)

def train_single_run(cfg, run_dir, train_loader, val_loader, train_items, device, fold_info=None):
    """
    Executes one complete training cycle (setup model -> train -> save).
    """
    # 1. Build Model
    model = ModelBuilder.build(cfg, device)
    
    # 2. Loss & Optimizer
    criterion = LossFactory.get(cfg, train_items=train_items, device=device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg['train']['learning_rate'], 
        weight_decay=cfg['train']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=cfg['train'].get('lr_milestones', [15, 30]), 
        gamma=cfg['train'].get('lr_gamma', 0.1)
    )
    
    # 3. Trainer
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
    
    # Global System Seed (for weight init, etc.)
    system_seed = cfg['system'].get('seed', 42)
    seed_everything(system_seed)
    
    device = torch.device(cfg['system'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    
    # Base Run Directory
    base_run_name = cfg['system'].get('run_name', 'default_run')
    runs_root = Path(cfg['system'].get('runs_dir', 'runs'))
    
    # Data Manager
    dm = DataManager(cfg)
    
    # --- BRANCH: CV vs Standard ---
    cv_cfg = cfg.get('train', {}).get('cv', cfg.get('cv', {}))
    if cv_cfg.get('enabled', False):
        # CROSS VALIDATION MODE
        num_folds = int(cv_cfg.get('num_runs', 5))
        
        # Use specific CV seed if provided, else fallback to system seed
        cv_seed = cv_cfg.get('seed', system_seed)
        
        base_dir = ensure_unique_run_dir(runs_root, f"{base_run_name}_CV")
        
        print(f"-> Cross-Validation Enabled: {num_folds} folds.")
        print(f"-> CV Shuffle Seed: {cv_seed}")
        print(f"-> Saving to {base_dir}")
        
        # Save master config
        atomic_write_json(base_dir / "cv_config.json", cfg)
        
        for fold in range(num_folds):
            # Create sub-directory for this fold
            fold_dir = base_dir / f"fold_{fold+1}"
            fold_dir.mkdir(exist_ok=True)
            
            # Get Data using CV SEED
            (train_loader, val_loader), train_items = dm.get_cv_loaders(
                fold_idx=fold, 
                num_folds=num_folds, 
                seed=cv_seed 
            )
            
            # Train
            train_single_run(
                cfg, fold_dir, train_loader, val_loader, 
                train_items, device, fold_info=f"Fold {fold+1}/{num_folds}"
            )
            
    else:
        # STANDARD MODE (Fixed Train/Val folders)
        run_dir = ensure_unique_run_dir(runs_root, base_run_name)
        
        (train_loader, val_loader), train_items = dm.get_fixed_loaders()
        
        train_single_run(
            cfg, run_dir, train_loader, val_loader, 
            train_items, device
        )

if __name__ == "__main__":
    main()
