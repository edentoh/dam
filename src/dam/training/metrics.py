import torch
import numpy as np
from sklearn.metrics import average_precision_score

def calculate_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes Micro F1, Macro F1, Element-wise Accuracy, and mAP (macro/micro).
    Args:
        y_true (Tensor): Ground truth labels (0 or 1)
        y_prob (Tensor): Sigmoid probabilities (0.0 to 1.0)
        threshold (float): Decision threshold
    """
    y_pred = (y_prob >= threshold).float()

    # --- Micro F1 ---
    tp_micro = (y_pred * y_true).sum()
    denom_micro = (y_pred + y_true).sum()
    micro_f1 = (2 * tp_micro / denom_micro).item() if denom_micro > 0 else 0.0

    # --- Macro F1 ---
    # Calculate per-class TP, FP, FN
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

    # --- mAP (average precision) ---
    # Convert to numpy for sklearn
    y_true_np = y_true.detach().cpu().numpy()
    y_prob_np = y_prob.detach().cpu().numpy()
    try:
        # For macro mAP, ignore classes with no positive samples in y_true
        pos_counts = y_true_np.sum(axis=0)
        valid = pos_counts > 0
        if valid.any():
            map_macro = float(average_precision_score(y_true_np[:, valid], y_prob_np[:, valid], average="macro"))
        else:
            map_macro = 0.0
        map_micro = float(average_precision_score(y_true_np, y_prob_np, average="micro"))
    except Exception:
        map_macro = 0.0
        map_micro = 0.0

    return micro_f1, macro_f1, acc, map_macro, map_micro
