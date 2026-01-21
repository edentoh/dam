import torch

def calculate_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes Micro F1, Macro F1, and Element-wise Accuracy.
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

    return micro_f1, macro_f1, acc