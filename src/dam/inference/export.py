import pandas as pd
import numpy as np
import torch
from pathlib import Path

def _elementwise_accuracy_vector(y_true: np.ndarray, y_prob: np.ndarray, thr_vec: np.ndarray) -> float:
    y_pred = (y_prob >= thr_vec.reshape(1, -1)).astype(np.int32)
    return float((y_pred == y_true.astype(np.int32)).mean())

def _micro_f1_vector(y_true: np.ndarray, y_prob: np.ndarray, thr_vec: np.ndarray) -> float:
    y_pred = (y_prob >= thr_vec.reshape(1, -1)).astype(np.int32)
    y_true_i = y_true.astype(np.int32)
    tp = int((y_pred & y_true_i).sum())
    fp = int((y_pred & (1 - y_true_i)).sum())
    fn = int(((1 - y_pred) & y_true_i).sum())
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0

def save_predictions_to_excel(
    probs_by_id: dict, 
    label_map: dict, 
    thr_vec: np.ndarray, 
    thr_info: dict, 
    out_path: Path
):
    """
    Generates the Excel report with Predictions, Probabilities, and Metrics.
    """
    ordered_ids = sorted(probs_by_id.keys())
    cols = [f"Image {img_id}" for img_id in ordered_ids]

    # pred_prob: (48, N)
    pred_prob = np.stack([probs_by_id[img_id] for img_id in ordered_ids], axis=1)

    # Apply per-item thresholds
    pred_bin = (pred_prob >= thr_vec.reshape(-1, 1)).astype(np.float32)

    df_pred = pd.DataFrame(pred_bin, columns=cols)
    df_prob = pd.DataFrame(pred_prob, columns=cols)

    df_pred.insert(0, "Item", [f"Item {i}" for i in range(1, 49)])
    df_prob.insert(0, "Item", [f"Item {i}" for i in range(1, 49)])

    # Total score per image
    totals_01 = pred_bin.sum(axis=0)
    total_row_pred = pd.DataFrame([["Total", *totals_01.tolist()]], columns=["Item", *cols])
    df_pred = pd.concat([df_pred, total_row_pred], ignore_index=True)

    # Metrics logic
    metrics_rows = []
    per_item_rows = []
    common_ids = sorted(set(probs_by_id.keys()) & set(label_map.keys()))
    num_classes = len(thr_vec)
    thr_scalar = thr_vec[0] # Approximate for summary if uniform

    metrics_rows.append(["threshold_mode", thr_info["threshold_mode"]])
    metrics_rows.append(["threshold_vector_path", thr_info["threshold_vector_path"]])
    metrics_rows.append(["num_pred_images", len(probs_by_id)])
    metrics_rows.append(["num_labeled_images", len(common_ids)])

    if common_ids:
        y_true = np.stack([label_map[i] for i in common_ids], axis=0).astype(np.float32)
        y_prob_labeled = np.stack([probs_by_id[i] for i in common_ids], axis=0).astype(np.float32)

        acc_overall = _elementwise_accuracy_vector(y_true, y_prob_labeled, thr_vec)
        f1_overall = _micro_f1_vector(y_true, y_prob_labeled, thr_vec)

        metrics_rows.append(["elementwise_accuracy_overall", acc_overall])
        metrics_rows.append(["micro_f1_overall", f1_overall])

        # Per-item metrics
        y_pred_labeled = (y_prob_labeled >= thr_vec.reshape(1, -1)).astype(np.int32)
        y_true_i = y_true.astype(np.int32)

        for k in range(num_classes):
            item_acc = float((y_pred_labeled[:, k] == y_true_i[:, k]).mean())
            per_item_rows.append([f"Item {k+1}", float(thr_vec[k]), item_acc])
    else:
        metrics_rows.append(["info", "No matching labels found for metrics."])

    df_metrics_summary = pd.DataFrame(metrics_rows, columns=["metric", "value"])
    df_metrics_per_item = pd.DataFrame(per_item_rows, columns=["item", "threshold", "accuracy"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_pred.to_excel(writer, index=False, sheet_name="Predictions_0_1")
        df_prob.to_excel(writer, index=False, sheet_name="Probabilities_0_1")
        df_metrics_summary.to_excel(writer, index=False, sheet_name="Metrics_Summary")
        df_metrics_per_item.to_excel(writer, index=False, sheet_name="Metrics_PerItem")
    
    print(f"Saved predictions to: {out_path}")