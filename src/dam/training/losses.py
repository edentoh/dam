from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.
    Reduces the penalty for negative samples (easy negatives) to focus on hard positives.
    """
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None

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

        if self.class_weight is None:
            return -loss.mean()

        w = self.class_weight.view(1, -1)
        loss = loss * w
        denom = w.sum()
        if denom.item() <= 0:
            return loss.sum() * 0.0
        return -loss.sum() / (denom * x.shape[0])


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss that supports:
      - pos_weight (positive class weighting)
      - class_weight (per-class weight applied to both pos/neg terms)
    class_weight entries set to 0.0 effectively mask a class from training.
    """
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None

    def forward(self, x, y):
        loss = F.binary_cross_entropy_with_logits(
            x,
            y,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if self.class_weight is None:
            return loss.mean()

        w = self.class_weight.view(1, -1)
        loss = loss * w
        denom = w.sum()
        if denom.item() <= 0:
            return loss.sum() * 0.0
        return loss.sum() / (denom * x.shape[0])

class LossFactory:
    """
    Factory to create loss functions based on config.
    Supports: BCEWithLogitsLoss (weighted/unweighted) and AsymmetricLoss.
    """
    @staticmethod
    def get(cfg: dict, train_items: list = None, device: str = 'cpu'):
        loss_name = cfg['train'].get('loss', 'bce').lower()

        def build_class_weight():
            if not train_items:
                return None

            if not bool(cfg['train'].get('use_class_masking', True)):
                return None

            min_pos_for_loss = int(cfg['train'].get('min_pos_for_loss', 1))
            rare_pos_threshold = int(cfg['train'].get('rare_pos_threshold', 10))
            rare_weight_factor = float(cfg['train'].get('rare_weight_factor', 1.0))

            y = np.stack([it[1] for it in train_items], axis=0)
            pos_counts = (y > 0).sum(axis=0)

            weights = np.ones_like(pos_counts, dtype=np.float32)
            mask = pos_counts < min_pos_for_loss
            rare = (pos_counts <= rare_pos_threshold) & (~mask)

            weights[mask] = 0.0
            if not np.isclose(rare_weight_factor, 1.0):
                weights[rare] = rare_weight_factor

            if np.allclose(weights, 1.0):
                return None

            print(
                f"[Loss] Class-weighting: masked={int(mask.sum())}, "
                f"rare={int(rare.sum())}, rare_weight={rare_weight_factor}"
            )
            return torch.tensor(weights, device=device, dtype=torch.float32)

        class_weight = build_class_weight()

        if loss_name == 'asl':
            return AsymmetricLoss(
                gamma_neg=float(cfg['train'].get('asl_gamma_neg', 4.0)),
                gamma_pos=float(cfg['train'].get('asl_gamma_pos', 1.0)),
                clip=float(cfg['train'].get('asl_clip', 0.05)),
                class_weight=class_weight,
            )

        elif loss_name == 'bce':
            # 1. Check for Manual Weights (List or Scalar)
            manual_weight = cfg['train'].get('pos_weight', None)

            if manual_weight is not None:
                print(f"[Loss] Using manual pos_weight from config: {manual_weight}")
                if isinstance(manual_weight, (list, tuple)):
                    pos_weight = torch.tensor(manual_weight, device=device, dtype=torch.float32)
                else:
                    pos_weight = torch.tensor(float(manual_weight), device=device, dtype=torch.float32)
                return WeightedBCEWithLogitsLoss(pos_weight=pos_weight, class_weight=class_weight)

            # 2. Check for Auto-Calculation
            use_weighted = bool(cfg['train'].get('use_weighted_loss', False))
            if use_weighted and train_items:
                clamp_val = float(cfg['train'].get('pos_weight_clamp', 10.0))
                print(f"[Loss] Calculating pos_weights from data (clamp={clamp_val})...")
                pos_weight = LossFactory._calculate_pos_weights(train_items, clamp_val).to(device)
                return WeightedBCEWithLogitsLoss(pos_weight=pos_weight, class_weight=class_weight)

            # 3. Default (No weights)
            if class_weight is not None:
                return WeightedBCEWithLogitsLoss(class_weight=class_weight)
            return nn.BCEWithLogitsLoss()

        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    @staticmethod
    def _calculate_pos_weights(items, max_weight=10.0):
        # Extract all labels from the dataset items
        y = np.stack([it[1] for it in items], axis=0)
        pos = y.sum(axis=0)
        neg = y.shape[0] - pos
        w = np.ones_like(pos, dtype=np.float32)
        
        mask = pos > 0
        w[mask] = (neg[mask] / pos[mask]).astype(np.float32)

        # Clamp to avoid exploding gradients on very rare classes
        w = np.clip(w, 1.0, max_weight)
        return torch.tensor(w, dtype=torch.float32)
