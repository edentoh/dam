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
        self.aux_decay_after_unfreeze = bool(aux_cfg.get("decay_after_unfreeze", True))
        self.aux_post_warmup_decay_factor = float(aux_cfg.get("post_warmup_decay_factor", 0.1))
        self.aux_use_pcgrad = bool(aux_cfg.get("use_pcgrad", True))
        self.grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))

        amp_dtype_cfg = str(cfg["train"].get("amp_dtype", "fp16")).lower()
        amp_dtype_map = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        self.amp_dtype = amp_dtype_map.get(amp_dtype_cfg, torch.float16)
        self.use_amp = bool(cfg["train"].get("use_amp", True)) and self.device.type == "cuda"
        self.use_scaler = self.use_amp and self.amp_dtype == torch.float16
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_scaler)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

        self.classifier_head_param_ids = set()
        for m in resolve_classifier_modules(self.model):
            for p in m.parameters():
                self.classifier_head_param_ids.add(id(p))
        self.aux_head_param_ids = set()
        if hasattr(self.model, "aux_head") and self.model.aux_head is not None:
            for p in self.model.aux_head.parameters():
                self.aux_head_param_ids.add(id(p))

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

    @staticmethod
    def _project_if_conflict(g_a, g_b):
        if g_a is None:
            return None
        if g_b is None:
            return g_a
        dot = torch.dot(g_a.reshape(-1), g_b.reshape(-1))
        if dot >= 0:
            return g_a
        denom = g_b.pow(2).sum().clamp_min(1e-12)
        return g_a - (dot / denom) * g_b

    def _current_aux_weight(self, epoch: int, freeze_epochs: int) -> float:
        if not self.aux_enabled or self.aux_weight <= 0:
            return 0.0
        w = float(self.aux_weight)
        if self.aux_decay_after_unfreeze and freeze_epochs > 0 and epoch > freeze_epochs:
            w *= float(self.aux_post_warmup_decay_factor)
        return w

    def _get_backbone_params(self):
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "parameters"):
            return [p for p in self.model.backbone.parameters() if p.requires_grad]
        out = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in self.classifier_head_param_ids or pid in self.aux_head_param_ids:
                continue
            out.append(p)
        return out

    def _get_aux_head_params(self):
        if not (hasattr(self.model, "aux_head") and self.model.aux_head is not None):
            return []
        return [p for p in self.model.aux_head.parameters() if p.requires_grad]

    def _apply_aux_grads(self, aux_loss, aux_weight_step):
        if aux_loss is None or aux_weight_step <= 0:
            return

        backbone_params = self._get_backbone_params()
        aux_head_params = self._get_aux_head_params()
        if not backbone_params and not aux_head_params:
            return

        aux_targets = backbone_params + aux_head_params
        aux_scaled = (aux_weight_step * aux_loss) / self.grad_accum_steps
        aux_grads = torch.autograd.grad(aux_scaled, aux_targets, allow_unused=True)

        grad_scale = float(self.scaler.get_scale()) if self.use_scaler else 1.0
        num_backbone = len(backbone_params)
        aux_backbone = []
        for g in aux_grads[:num_backbone]:
            if g is None:
                aux_backbone.append(None)
            else:
                aux_backbone.append(g * grad_scale)

        if self.aux_use_pcgrad and num_backbone > 0:
            for p, g_aux in zip(backbone_params, aux_backbone):
                g_main = p.grad
                if g_main is None and g_aux is None:
                    continue
                if g_main is None:
                    p.grad = g_aux
                    continue
                if g_aux is None:
                    continue

                g_main_proj = self._project_if_conflict(g_main, g_aux)
                g_aux_proj = self._project_if_conflict(g_aux, g_main)
                p.grad = g_main_proj + g_aux_proj
        else:
            for p, g_aux in zip(backbone_params, aux_backbone):
                if g_aux is None:
                    continue
                if p.grad is None:
                    p.grad = g_aux
                else:
                    p.grad = p.grad + g_aux

        for p, g_aux in zip(aux_head_params, aux_grads[num_backbone:]):
            if g_aux is None:
                continue
            g_aux = g_aux * grad_scale
            if p.grad is None:
                p.grad = g_aux
            else:
                p.grad = p.grad + g_aux

    def train_epoch(self, aux_weight_step: float):
        self.model.train()
        total_loss = 0.0
        total_aux = 0.0
        aux_count = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, (x, y, _) in enumerate(self.train_loader, start=1):
            x, y = x.to(self.device), y.to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(x)
                logits, aux = self._split_outputs(outputs)
                loss_main = self.criterion(logits, y)
                loss = loss_main

                aux_loss = None
                if self.aux_enabled and aux_weight_step > 0 and aux is not None:
                    aux_loss = self._aux_loss(aux, y)
                    if aux_loss is not None:
                        loss = loss + aux_weight_step * aux_loss
                        total_aux += aux_loss.item() * x.size(0)
                        aux_count += x.size(0)

            total_loss += loss.item() * x.size(0)
            loss_main_for_backward = loss_main / self.grad_accum_steps
            use_aux_grad = aux_loss is not None and aux_weight_step > 0
            if self.use_scaler:
                self.scaler.scale(loss_main_for_backward).backward(retain_graph=use_aux_grad)
            else:
                loss_main_for_backward.backward(retain_graph=use_aux_grad)

            if use_aux_grad:
                self._apply_aux_grads(aux_loss=aux_loss, aux_weight_step=aux_weight_step)

            is_step_boundary = (
                (step_idx % self.grad_accum_steps == 0) or
                (step_idx == len(self.train_loader))
            )
            if is_step_boundary:
                if self.use_scaler:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_aux = (total_aux / aux_count) if aux_count > 0 else None
        return avg_loss, avg_aux

    @torch.no_grad()
    def validate(self, aux_weight_step: float):
        self.model.eval()
        total_loss = 0.0
        total_aux = 0.0
        aux_count = 0
        ys, ps = [], []

        for x, y, _ in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(x)
                logits, aux = self._split_outputs(outputs)
                loss_main = self.criterion(logits, y)
                loss = loss_main

                if self.aux_enabled and aux_weight_step > 0 and aux is not None:
                    aux_loss = self._aux_loss(aux, y)
                    if aux_loss is not None:
                        loss = loss + aux_weight_step * aux_loss
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
        if self.use_amp:
            amp_name = "fp16" if self.amp_dtype == torch.float16 else "bf16"
            print(f"[AMP] Enabled ({amp_name}); GradScaler={'on' if self.use_scaler else 'off'}")
        if self.grad_accum_steps > 1:
            print(f"[Accum] Gradient accumulation steps: {self.grad_accum_steps}")
        if self.aux_enabled and self.aux_weight > 0:
            mode = "PCGrad" if self.aux_use_pcgrad else "sum-grad"
            print(
                f"[Aux] Enabled | mode={mode} | base_weight={self.aux_weight:.4f} | "
                f"post_warmup_decay_factor={self.aux_post_warmup_decay_factor:.4f}"
            )

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

            aux_weight_step = self._current_aux_weight(ep, freeze_epochs)
            t_loss, t_aux = self.train_epoch(aux_weight_step=aux_weight_step)
            v_loss, v_acc, v_micro, v_macro, v_map_macro, v_map_micro, v_aux = self.validate(aux_weight_step=aux_weight_step)

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
                "aux_weight_step": float(aux_weight_step),
            }
            self.history["epochs"].append(row)
            atomic_write_json(self.run_dir / "history.json", self.history)

            # Checkpoint Best
            if metric_key == "val_loss":
                current_val = -float(v_loss)
            elif metric_key == "val_f1_macro":
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
                    "metric_for_best": str(metric_key),
                    "best_metric_val": float(best_metric),
                    "epoch_metrics": row,
                    "checkpoint_path": str(best_path),
                }
                atomic_write_json(self.run_dir / "best_model_metadata.json", best_meta)
                print(f"  --> New Best {metric_key}: {best_metric:.4f} saved")

        print("Training Finished.")
