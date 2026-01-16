import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()

class LossFactory:
    @staticmethod
    def get(cfg: dict, train_items: list = None, device: str = 'cpu'):
        loss_name = cfg['train'].get('loss', 'bce').lower()
        
        if loss_name == 'asl':
            return AsymmetricLoss(
                gamma_neg=float(cfg['train'].get('asl_gamma_neg', 4.0)),
                gamma_pos=float(cfg['train'].get('asl_gamma_pos', 1.0)),
                clip=float(cfg['train'].get('asl_clip', 0.05))
            )
            
        elif loss_name == 'bce':
            # 1. Check for Manual Weights (List or Scalar)
            manual_weight = cfg['train'].get('pos_weight', None)
            
            if manual_weight is not None:
                print(f"[Loss] Using manual pos_weight from config: {manual_weight}")
                # Handle list vs scalar
                if isinstance(manual_weight, (list, tuple)):
                    pos_weight = torch.tensor(manual_weight, device=device, dtype=torch.float32)
                else:
                    pos_weight = torch.tensor(float(manual_weight), device=device, dtype=torch.float32)
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # 2. Check for Auto-Calculation
            use_weighted = bool(cfg['train'].get('use_weighted_loss', False))
            if use_weighted and train_items:
                clamp_val = float(cfg['train'].get('pos_weight_clamp', 10.0))
                print(f"[Loss] Calculating pos_weights from data (clamp={clamp_val})...")
                pos_weight = LossFactory._calculate_pos_weights(train_items, clamp_val).to(device)
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # 3. Default (No weights)
            return nn.BCEWithLogitsLoss()
        
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    @staticmethod
    def _calculate_pos_weights(items, max_weight=10.0):
        y = np.stack([it[1] for it in items], axis=0)
        pos = y.sum(axis=0)
        neg = y.shape[0] - pos
        w = np.ones_like(pos, dtype=np.float32)
        mask = pos > 0
        w[mask] = (neg[mask] / pos[mask]).astype(np.float32)
        
        # Clamp to avoid exploding gradients on very rare classes
        w = np.clip(w, 1.0, max_weight)
        return torch.tensor(w, dtype=torch.float32)
