import torch
import torch.nn as nn
from dam.modeling.utils import resolve_classifier_modules

def build_optimizer(cfg: dict, model: nn.Module):
    """
    Creates AdamW optimizer with support for:
      1. Weight Decay Filtering (no decay on bias/norm layers).
      2. Discriminative Learning Rates (different LR for backbone vs head).
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
        # Most normalization weights are 1D
        if getattr(p, "ndim", None) == 1:
            return True
        return False

    use_disc = bool(train_cfg.get("use_discriminative_lr", False))
    if not use_disc:
        # Standard Single LR
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if is_no_decay(n, p) else decay).append(p)

        param_groups = []
        if len(decay) > 0:
            param_groups.append({"params": decay, "lr": base_lr, "weight_decay": weight_decay, "name": "all_decay"})
        if len(no_decay) > 0:
            param_groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0, "name": "all_no_decay"})
        
        # Fallback if filtering yielded nothing (e.g. empty model?)
        if not param_groups:
             return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
             
        return torch.optim.AdamW(param_groups)

    # --- Discriminative LR Logic ---
    backbone_mult = float(train_cfg.get("backbone_lr_mult", 0.1))
    head_mult = float(train_cfg.get("head_lr_mult", 1.0))

    head_modules = resolve_classifier_modules(model)
    head_params = []
    for m in head_modules:
        head_params.extend(list(m.parameters()))

    head_param_ids = {id(p) for p in head_params}

    backbone_named, head_named = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_param_ids:
            head_named.append((n, p))
        else:
            backbone_named.append((n, p))

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

    print(
        f"[Optimizer] Discriminative LR enabled | "
        f"backbone_lr={bb_lr:.2e} (x{backbone_mult}) | "
        f"head_lr={hd_lr:.2e} (x{head_mult})"
    )
    return torch.optim.AdamW(param_groups)