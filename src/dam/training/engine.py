from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        aux_cfg = cfg.get("aux_head", {})
        self.aux_enabled = bool(aux_cfg.get("enabled", False))
        self.aux_weight = float(aux_cfg.get("weight", 0.0))
        self.aux_loss_type = str(aux_cfg.get("loss", "mse")).lower()
        self.aux_normalize = bool(aux_cfg.get("normalize_target", True))

        self.history = {
            "created_at": datetime.now().isoformat(),
            "config": cfg,
            "epochs": [],
        }

    def _split_outputs(self, outputs):
        if isinstance(outputs, dict):
            return outputs.get("logits", outputs), outputs.get("aux", None)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0] if len(outputs) > 0 else outputs
            aux = outputs[1] if len(outputs) > 1 else None
            return logits, aux
        return outputs, None

    def _aux_loss(self, aux_pred, y_true):
        if aux_pred is None:
            return None
        target = y_true.sum(dim=1)
        if self.aux_normalize and y_true.shape[1] > 0:
            target = target / float(y_true.shape[1])
        aux_pred = aux_pred.view(-1)
        if self.aux_loss_type == "l1":
            return F.l1_loss(aux_pred, target)
        return F.mse_loss(aux_pred, target)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_aux = 0.0
        aux_count = 0
        
        for x, y, _ in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            outputs = self.model(x)
            logits, aux = self._split_outputs(outputs)
            loss_main = self.criterion(logits, y)
            loss = loss_main

            aux_loss = None
            if self.aux_enabled and self.aux_weight > 0 and aux is not None:
                aux_loss = self._aux_loss(aux, y)
                if aux_loss is not None:
                    loss = loss + self.aux_weight * aux_loss
                    total_aux += aux_loss.item() * x.size(0)
                    aux_count += x.size(0)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_aux = (total_aux / aux_count) if aux_count > 0 else None
        return avg_loss, avg_aux

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_aux = 0.0
        aux_count = 0
        ys, ps = [], []

        for x, y, _ in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            logits, aux = self._split_outputs(outputs)
            loss_main = self.criterion(logits, y)
            loss = loss_main

            if self.aux_enabled and self.aux_weight > 0 and aux is not None:
                aux_loss = self._aux_loss(aux, y)
                if aux_loss is not None:
                    loss = loss + self.aux_weight * aux_loss
                    total_aux += aux_loss.item() * x.size(0)
                    aux_count += x.size(0)

            total_loss += loss.item() * x.size(0)

            ys.append(y.cpu())
            ps.append(torch.sigmoid(logits).cpu())

        y_true = torch.cat(ys)
        y_prob = torch.cat(ps)

        micro_f1, macro_f1, acc, map_macro, map_micro = calculate_metrics(y_true, y_prob, self.threshold)

        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_aux = (total_aux / aux_count) if aux_count > 0 else None

        return avg_loss, acc, micro_f1, macro_f1, map_macro, map_micro, avg_aux

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

            t_loss, t_aux = self.train_epoch()
            v_loss, v_acc, v_micro, v_macro, v_map_macro, v_map_micro, v_aux = self.validate()

            self.scheduler.step()
            
            # Log LR
            lr_groups = []
            for i, pg in enumerate(self.optimizer.param_groups):
                name = pg.get("name", f"group_{i}")
                lr_groups.append({"name": str(name), "lr": float(pg.get("lr", 0.0))})

            lr_display = " | ".join([f"{g['name']} {g['lr']:.2e}" for g in lr_groups])

            aux_str = ""
            if t_aux is not None or v_aux is not None:
                t_aux_str = f"{t_aux:.4f}" if t_aux is not None else "NA"
                v_aux_str = f"{v_aux:.4f}" if v_aux is not None else "NA"
                aux_str = f" | T_Aux {t_aux_str} | V_Aux {v_aux_str}"

            print(
                f"Ep {ep:03d} | Lr {lr_display} | "
                f"T_Loss {t_loss:.4f} | V_Loss {v_loss:.4f}{aux_str} | "
                f"V_F1(mac) {v_macro:.4f} | V_F1(mic) {v_micro:.4f} | "
                f"V_mAP(mac) {v_map_macro:.4f} | V_mAP(mic) {v_map_micro:.4f} | V_Acc {v_acc:.4f}"
            )

            # Record History
            row = {
                "epoch": ep,
                "train_loss": t_loss,
                "train_aux_loss": t_aux,
                "val_loss": v_loss,
                "val_aux_loss": v_aux,
                "val_f1_micro": v_micro,
                "val_f1_macro": v_macro,
                "val_map_macro": v_map_macro,
                "val_map_micro": v_map_micro,
                "val_acc": v_acc,
                "lr_groups": lr_groups,
                "backbone_frozen": bool(is_frozen),
            }
            self.history["epochs"].append(row)
            atomic_write_json(self.run_dir / "history.json", self.history)

            # Checkpoint Best
            if metric_key == "val_f1_macro":
                current_val = v_macro
            elif metric_key == "val_map_macro":
                current_val = v_map_macro
            elif metric_key == "val_map_micro":
                current_val = v_map_micro
            else:
                current_val = v_micro

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
