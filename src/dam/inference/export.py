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

def _macro_f1_vector(y_true: np.ndarray, y_prob: np.ndarray, thr_vec: np.ndarray) -> float:
    y_pred = (y_prob >= thr_vec.reshape(1, -1)).astype(np.int32)
    y_true_i = y_true.astype(np.int32)
    tp = (y_pred & y_true_i).sum(axis=0)
    fp = (y_pred & (1 - y_true_i)).sum(axis=0)
    fn = ((1 - y_pred) & y_true_i).sum(axis=0)
    denom = (2 * tp + fp + fn)
    f1 = np.where(denom > 0, (2 * tp / denom), 0.0)
    return float(f1.mean())

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

    # Ground-truth sheet (aligned to prediction columns)
    df_gt = None
    if label_map:
        gt = np.full((len(thr_vec), len(ordered_ids)), np.nan, dtype=np.float32)
        for j, img_id in enumerate(ordered_ids):
            if img_id in label_map:
                gt[:, j] = label_map[img_id]
        df_gt = pd.DataFrame(gt, columns=cols)
        df_gt.insert(0, "Item", [f"Item {i}" for i in range(1, 49)])
        total_gt = np.nansum(gt, axis=0)
        total_row_gt = pd.DataFrame([["Total", *total_gt.tolist()]], columns=["Item", *cols])
        df_gt = pd.concat([df_gt, total_row_gt], ignore_index=True)

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
        micro_f1_overall = _micro_f1_vector(y_true, y_prob_labeled, thr_vec)
        macro_f1_overall = _macro_f1_vector(y_true, y_prob_labeled, thr_vec)

        metrics_rows.append(["elementwise_accuracy_overall", acc_overall])
        metrics_rows.append(["micro_f1_overall", micro_f1_overall])
        metrics_rows.append(["macro_f1_overall", macro_f1_overall])

        # Per-item metrics
        y_pred_labeled = (y_prob_labeled >= thr_vec.reshape(1, -1)).astype(np.int32)
        y_true_i = y_true.astype(np.int32)

        for k in range(num_classes):
            item_acc = float((y_pred_labeled[:, k] == y_true_i[:, k]).mean())
            tp = int(((y_pred_labeled[:, k] == 1) & (y_true_i[:, k] == 1)).sum())
            fp = int(((y_pred_labeled[:, k] == 1) & (y_true_i[:, k] == 0)).sum())
            fn = int(((y_pred_labeled[:, k] == 0) & (y_true_i[:, k] == 1)).sum())
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            per_item_rows.append([f"Item {k+1}", float(thr_vec[k]), item_acc, prec, rec, f1])
    else:
        metrics_rows.append(["info", "No matching labels found for metrics."])

    df_metrics_summary = pd.DataFrame(metrics_rows, columns=["metric", "value"])
    df_metrics_per_item = pd.DataFrame(per_item_rows, columns=["item", "threshold", "accuracy", "precision", "recall", "f1"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_pred.to_excel(writer, index=False, sheet_name="Predictions_0_1")
        df_prob.to_excel(writer, index=False, sheet_name="Probabilities_0_1")
        if df_gt is not None:
            df_gt.to_excel(writer, index=False, sheet_name="GroundTruth_0_1")
        df_metrics_summary.to_excel(writer, index=False, sheet_name="Metrics_Summary")
        df_metrics_per_item.to_excel(writer, index=False, sheet_name="Metrics_PerItem")
    
    print(f"Saved predictions to: {out_path}")
