from datetime import datetime
import torch
import torch.nn as nn

from dam.utils.io import atomic_write_json
from dam.modeling.utils import resolve_classifier_modules
from .metrics import calculate_metrics

def _freeze_backbone(model):
    """Freeze all params except the classifier/head."""
    for p in model.parameters():
        p.requires_grad = False

    head_modules = resolve_classifier_modules(model)
    if not head_modules:
        # Fallback: if we can't find head, unfreeze everything to be safe
        for p in model.parameters():
            p.requires_grad = True
        return

    for m in head_modules:
        for p in m.parameters():
            p.requires_grad = True

def _unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

class Trainer:
    def __init__(self, model, loaders, criterion, optimizer, scheduler, cfg, device, run_dir):
        self.model = model
        self.train_loader, self.val_loader = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        
        self.threshold = cfg["train"].get("metric_threshold", cfg["train"].get("threshold", 0.5))

        self.history = {
            "created_at": datetime.now().isoformat(),
            "config": cfg,
            "epochs": [],
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for x, y, _ in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        ys, ps = [], []

        for x, y, _ in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            ys.append(y.cpu())
            ps.append(torch.sigmoid(logits).cpu())

        y_true = torch.cat(ys)
        y_prob = torch.cat(ps)

        micro_f1, macro_f1, acc = calculate_metrics(y_true, y_prob, self.threshold)

        return (total_loss / len(self.val_loader.dataset)), acc, micro_f1, macro_f1

    def run(self):
        epochs = self.cfg["train"]["epochs"]
        best_metric = -1.0
        metric_key = self.cfg["train"].get("metric_for_best", "val_f1_micro")
        best_path = self.run_dir / "best.pth"

        # Hybrid schedule: freeze backbone for N epochs
        freeze_epochs = int(self.cfg["train"].get("freeze_backbone_epochs", 0) or 0)
        is_frozen = False

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for ep in range(1, epochs + 1):
            # Hybrid Freeze Logic
            if freeze_epochs > 0 and ep <= freeze_epochs:
                if not is_frozen:
                    _freeze_backbone(self.model)
                    is_frozen = True
                    print(f"[Hybrid] Backbone frozen (epochs 1-{freeze_epochs}).")
            else:
                if is_frozen:
                    _unfreeze_all(self.model)
                    is_frozen = False
                    print(f"[Hybrid] Backbone unfrozen from epoch {ep} onward.")

            t_loss = self.train_epoch()
            v_loss, v_acc, v_micro, v_macro = self.validate()

            self.scheduler.step()
            
            # Log LR
            lr_groups = []
            for i, pg in enumerate(self.optimizer.param_groups):
                name = pg.get("name", f"group_{i}")
                lr_groups.append({"name": str(name), "lr": float(pg.get("lr", 0.0))})

            lr_display = " | ".join([f"{g['name']} {g['lr']:.2e}" for g in lr_groups])

            print(
                f"Ep {ep:03d} | Lr {lr_display} | "
                f"T_Loss {t_loss:.4f} | V_Loss {v_loss:.4f} | "
                f"V_F1(mac) {v_macro:.4f} | V_F1(mic) {v_micro:.4f} | V_Acc {v_acc:.4f}"
            )

            # Record History
            row = {
                "epoch": ep,
                "train_loss": t_loss,
                "val_loss": v_loss,
                "val_f1_micro": v_micro,
                "val_f1_macro": v_macro,
                "val_acc": v_acc,
                "lr_groups": lr_groups,
                "backbone_frozen": bool(is_frozen),
            }
            self.history["epochs"].append(row)
            atomic_write_json(self.run_dir / "history.json", self.history)

            # Checkpoint Best
            current_val = v_macro if metric_key == "val_f1_macro" else v_micro

            if current_val > best_metric:
                best_metric = float(current_val)
                
                ckpt = {
                    "epoch": int(ep),
                    "model_state": self.model.state_dict(),
                    "metric_name": str(metric_key),
                    "best_metric_val": float(best_metric),
                    "epoch_metrics": row,
                    "config": self.cfg,
                    "img_size": int(self.cfg.get("train", {}).get("data", {}).get("img_size", 384))
                }
                torch.save(ckpt, best_path)

                best_meta = {
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "run_dir": str(self.run_dir),
                    "best_epoch": int(ep),
                    "best_metric_val": float(best_metric),
                    "checkpoint_path": str(best_path),
                }
                atomic_write_json(self.run_dir / "best_model_metadata.json", best_meta)
                print(f"  --> New Best {metric_key}: {best_metric:.4f} saved")

        print("Training Finished.")