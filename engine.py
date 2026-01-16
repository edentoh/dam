<<<<<<< HEAD
import torch
import torch.nn as nn
from datetime import datetime
from utils import atomic_write_json

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
        
        # Calculate both Micro and Macro
        micro_f1, macro_f1, acc = calculate_metrics(y_true, y_prob, self.threshold)
        
        return (total_loss / len(self.val_loader.dataset)), acc, micro_f1, macro_f1

    def run(self):
        epochs = self.cfg['train']['epochs']
        best_metric = -1.0
        # Determine which metric controls "Best Model" saving (default to Micro if not specified)
        metric_key = self.cfg['train'].get('metric_for_best', 'val_f1_micro')
        best_path = self.run_dir / "best.pth"
        
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for ep in range(1, epochs + 1):
            t_loss = self.train_epoch()
            v_loss, v_acc, v_micro, v_macro = self.validate()
            
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            
            # --- UPDATED DISPLAY: Shows Macro F1 ---
            print(f"Ep {ep:03d} | Lr {curr_lr:.2e} | T_Loss {t_loss:.4f} | V_Loss {v_loss:.4f} | V_F1(mac) {v_macro:.4f} | V_F1(mic) {v_micro:.4f} | V_Acc {v_acc:.4f}")
            
            # --- UPDATED HISTORY: Logs Macro F1 ---
            row = {
                "epoch": ep, 
                "train_loss": t_loss, 
                "val_loss": v_loss, 
                "val_f1_micro": v_micro, 
                "val_f1_macro": v_macro,  # <--- Saved to JSON
                "val_acc": v_acc
            }
            self.history["epochs"].append(row)
            atomic_write_json(self.run_dir / "history.json", self.history)
            
            # Decide which metric to track for "best"
            current_val = v_macro if metric_key == 'val_f1_macro' else v_micro
            
            if current_val > best_metric:
                best_metric = float(current_val)

                # Save checkpoint (include full epoch metrics too)
                ckpt = {
                    "epoch": int(ep),
                    "model_state": self.model.state_dict(),
                    "metric_name": str(metric_key),
                    "best_metric_val": float(best_metric),
                    "epoch_metrics": row,                 # train_loss, val_loss, val_f1_micro, val_f1_macro, val_acc
                    "learning_rate": float(curr_lr),
                    "metric_threshold": float(self.threshold),
                    "config": self.cfg,
                }
                torch.save(ckpt, best_path)

                # Save best metadata JSON (human-readable)
                best_meta = {
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "run_dir": str(self.run_dir),
                    "best_epoch": int(ep),
                    "metric_for_best": str(metric_key),
                    "best_metric_val": float(best_metric),
                    "checkpoint_path": str(best_path),
                    "history_path": str(self.run_dir / "history.json"),
                    "learning_rate": float(curr_lr),
                    "metric_threshold": float(self.threshold),
                    "epoch_metrics": row,
                    "config": self.cfg,
                }
                atomic_write_json(self.run_dir / "best_model_metadata.json", best_meta)

                print(f"  --> New Best {metric_key}: {best_metric:.4f} saved (+ metadata)")

        print("Training Finished.")
=======
import torch
import torch.nn as nn
from datetime import datetime
from utils import atomic_write_json

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
        
        # Calculate both Micro and Macro
        micro_f1, macro_f1, acc = calculate_metrics(y_true, y_prob, self.threshold)
        
        return (total_loss / len(self.val_loader.dataset)), acc, micro_f1, macro_f1

    def run(self):
        epochs = self.cfg['train']['epochs']
        best_metric = -1.0
        # Determine which metric controls "Best Model" saving (default to Micro if not specified)
        metric_key = self.cfg['train'].get('metric_for_best', 'val_f1_micro')
        best_path = self.run_dir / "best.pth"
        
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for ep in range(1, epochs + 1):
            t_loss = self.train_epoch()
            v_loss, v_acc, v_micro, v_macro = self.validate()
            
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            
            # --- UPDATED DISPLAY: Shows Macro F1 ---
            print(f"Ep {ep:03d} | Lr {curr_lr:.2e} | T_Loss {t_loss:.4f} | V_Loss {v_loss:.4f} | V_F1(mac) {v_macro:.4f} | V_F1(mic) {v_micro:.4f} | V_Acc {v_acc:.4f}")
            
            # --- UPDATED HISTORY: Logs Macro F1 ---
            row = {
                "epoch": ep, 
                "train_loss": t_loss, 
                "val_loss": v_loss, 
                "val_f1_micro": v_micro, 
                "val_f1_macro": v_macro,  # <--- Saved to JSON
                "val_acc": v_acc
            }
            self.history["epochs"].append(row)
            atomic_write_json(self.run_dir / "history.json", self.history)
            
            # Decide which metric to track for "best"
            current_val = v_macro if metric_key == 'val_f1_macro' else v_micro
            
            if current_val > best_metric:
                best_metric = float(current_val)

                # Save checkpoint (include full epoch metrics too)
                ckpt = {
                    "epoch": int(ep),
                    "model_state": self.model.state_dict(),
                    "metric_name": str(metric_key),
                    "best_metric_val": float(best_metric),
                    "epoch_metrics": row,                 # train_loss, val_loss, val_f1_micro, val_f1_macro, val_acc
                    "learning_rate": float(curr_lr),
                    "metric_threshold": float(self.threshold),
                    "config": self.cfg,
                }
                torch.save(ckpt, best_path)

                # Save best metadata JSON (human-readable)
                best_meta = {
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "run_dir": str(self.run_dir),
                    "best_epoch": int(ep),
                    "metric_for_best": str(metric_key),
                    "best_metric_val": float(best_metric),
                    "checkpoint_path": str(best_path),
                    "history_path": str(self.run_dir / "history.json"),
                    "learning_rate": float(curr_lr),
                    "metric_threshold": float(self.threshold),
                    "epoch_metrics": row,
                    "config": self.cfg,
                }
                atomic_write_json(self.run_dir / "best_model_metadata.json", best_meta)

                print(f"  --> New Best {metric_key}: {best_metric:.4f} saved (+ metadata)")

        print("Training Finished.")
>>>>>>> d01af25 (update files)
