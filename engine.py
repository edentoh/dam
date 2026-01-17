import torch
import torch.nn as nn
from datetime import datetime
from utils import atomic_write_json


def _get_module_by_path(root, path: str):
    """Resolves dotted attribute paths (e.g., 'head.fc') against a module."""
    cur = root
    for part in str(path).split('.'):
        if not part:
            continue
        cur = getattr(cur, part)
    return cur


def _resolve_classifier_modules(model):
    """Best-effort resolution of a model's classifier/head modules.

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

    out = []
    for m in modules:
        if hasattr(m, "parameters"):
            out.append(m)
    return out


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes Micro F1, Macro F1, and Element-wise Accuracy.
    """
    y_pred = (y_prob >= threshold).float()

    # --- Micro F1 ---
    # Global TP, FP, FN
    tp_micro = (y_pred * y_true).sum()
    denom_micro = (y_pred + y_true).sum()
    micro_f1 = (2 * tp_micro / denom_micro).item() if denom_micro > 0 else 0.0

    # --- Macro F1 ---
    # Per-class TP, FP, FN
    tp = (y_pred * y_true).sum(dim=0)
    fp = (y_pred * (1 - y_true)).sum(dim=0)
    fn = ((1 - y_pred) * y_true).sum(dim=0)

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_per_class = 2 * (precision * recall) / (precision + recall + eps)
    macro_f1 = f1_per_class.mean().item()

    # --- Accuracy ---
    acc = (y_pred == y_true).float().mean().item()

    return micro_f1, macro_f1, acc


def _freeze_backbone(model):
    """Freeze all params except the classifier/head."""
    for p in model.parameters():
        p.requires_grad = False

    head_modules = _resolve_classifier_modules(model)
    # Best-effort: if we can't resolve the head, do nothing (safer than freezing everything).
    if not head_modules:
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
        self.threshold = cfg['train'].get('metric_threshold', cfg['train'].get('threshold', 0.5))

        # History tracking
        self.history = {
            "created_at": datetime.now().isoformat(),
            "config": cfg,
            "epochs": []
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
        epochs = self.cfg['train']['epochs']
        best_metric = -1.0
        metric_key = self.cfg['train'].get('metric_for_best', 'val_f1_micro')
        best_path = self.run_dir / "best.pth"

        # Hybrid schedule: freeze backbone for N epochs, then unfreeze and continue fine-tuning.
        freeze_epochs = int(self.cfg['train'].get('freeze_backbone_epochs', 0) or 0)
        is_frozen = False

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for ep in range(1, epochs + 1):
            # Apply hybrid freeze/unfreeze policy at the start of the epoch.
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
            lr_groups = []
            for i, pg in enumerate(self.optimizer.param_groups):
                name = pg.get('name', f'group_{i}')
                lr_groups.append({"name": str(name), "lr": float(pg.get('lr', 0.0))})

            if len(lr_groups) == 1:
                lr_display = f"{lr_groups[0]['lr']:.2e}"
            else:
                lr_display = " | ".join([f"{g['name']} {g['lr']:.2e}" for g in lr_groups])

            print(
                f"Ep {ep:03d} | Lr {lr_display} | "
                f"T_Loss {t_loss:.4f} | V_Loss {v_loss:.4f} | "
                f"V_F1(mac) {v_macro:.4f} | V_F1(mic) {v_micro:.4f} | V_Acc {v_acc:.4f}"
            )

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

            current_val = v_macro if metric_key == 'val_f1_macro' else v_micro

            if current_val > best_metric:
                best_metric = float(current_val)

                ckpt = {
                    "epoch": int(ep),
                    "model_state": self.model.state_dict(),
                    "metric_name": str(metric_key),
                    "best_metric_val": float(best_metric),
                    "epoch_metrics": row,
                    "learning_rate": [g["lr"] for g in lr_groups],
                    "learning_rate_groups": lr_groups,
                    "metric_threshold": float(self.threshold),
                    "config": self.cfg,
                }
                torch.save(ckpt, best_path)

                best_meta = {
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "run_dir": str(self.run_dir),
                    "best_epoch": int(ep),
                    "metric_for_best": str(metric_key),
                    "best_metric_val": float(best_metric),
                    "checkpoint_path": str(best_path),
                    "history_path": str(self.run_dir / "history.json"),
                    "learning_rate": [g["lr"] for g in lr_groups],
                    "learning_rate_groups": lr_groups,
                    "metric_threshold": float(self.threshold),
                    "epoch_metrics": row,
                    "config": self.cfg,
                }
                atomic_write_json(self.run_dir / "best_model_metadata.json", best_meta)

                print(f"  --> New Best {metric_key}: {best_metric:.4f} saved (+ metadata)")

        print("Training Finished.")