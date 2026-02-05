from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np

from dam.core.constants import CRITERIA
from dam.utils.io import atomic_write_json


def _criteria_names(num_classes: int) -> list[str]:
    if isinstance(CRITERIA, list) and len(CRITERIA) >= num_classes:
        return [str(x) for x in CRITERIA[:num_classes]]
    return [f"criterion_{i+1}" for i in range(num_classes)]


def _labels_from_train_items(train_items: list) -> np.ndarray:
    if not train_items:
        return np.zeros((0, 0), dtype=np.int32)

    ys = []
    for it in train_items:
        y = np.asarray(it[1], dtype=np.float32)
        ys.append(y)

    y_arr = np.stack(ys, axis=0)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)

    return (y_arr > 0).astype(np.int32)


def _phi_from_counts(n11: int, n10: int, n01: int, n00: int) -> float:
    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denom <= 0:
        return 0.0
    return float((n11 * n00 - n10 * n01) / np.sqrt(denom))


def compute_class_stats(
    y_bin: np.ndarray,
    min_pos_for_loss: int,
    rare_pos_threshold: int,
) -> dict:
    n_images = int(y_bin.shape[0])
    num_classes = int(y_bin.shape[1]) if y_bin.ndim > 1 else 0

    if num_classes == 0:
        return {
            "num_images": n_images,
            "num_classes": 0,
            "pos_counts": [],
            "pos_rates": [],
            "zero_pos_indices": [],
            "mask_loss_indices": [],
            "rare_indices": [],
        }

    pos_counts = y_bin.sum(axis=0).astype(int)
    pos_rates = (pos_counts / max(n_images, 1)).astype(float)

    zero_pos_indices = np.where(pos_counts == 0)[0].tolist()
    mask_loss_indices = np.where(pos_counts < int(min_pos_for_loss))[0].tolist()
    rare_indices = np.where(pos_counts <= int(rare_pos_threshold))[0].tolist()

    return {
        "num_images": n_images,
        "num_classes": num_classes,
        "pos_counts": pos_counts.tolist(),
        "pos_rates": pos_rates.tolist(),
        "zero_pos_indices": zero_pos_indices,
        "mask_loss_indices": mask_loss_indices,
        "rare_indices": rare_indices,
    }


def compute_correlation_edges(
    y_bin: np.ndarray,
    min_support: int,
    min_lift: float,
    names: list[str] | None = None,
) -> list[dict]:
    if y_bin.size == 0:
        return []

    n_images, num_classes = y_bin.shape
    pos_counts = y_bin.sum(axis=0).astype(int)

    edges = []
    for i in range(num_classes):
        if pos_counts[i] == 0:
            continue

        ai = y_bin[:, i]
        for j in range(i + 1, num_classes):
            if pos_counts[j] == 0:
                continue

            n11 = int(((ai == 1) & (y_bin[:, j] == 1)).sum())
            if n11 < min_support:
                continue

            n10 = int(pos_counts[i] - n11)
            n01 = int(pos_counts[j] - n11)
            n00 = int(n_images - n11 - n10 - n01)

            p_i = pos_counts[i] / n_images
            p_j = pos_counts[j] / n_images
            p_ij = n11 / n_images
            if p_i <= 0 or p_j <= 0:
                continue

            lift = p_ij / (p_i * p_j)
            if lift < min_lift:
                continue

            p_i_given_j = n11 / pos_counts[j] if pos_counts[j] > 0 else 0.0
            p_j_given_i = n11 / pos_counts[i] if pos_counts[i] > 0 else 0.0
            phi = _phi_from_counts(n11, n10, n01, n00)

            edges.append({
                "i": int(i),
                "j": int(j),
                "name_i": names[i] if names else f"criterion_{i+1}",
                "name_j": names[j] if names else f"criterion_{j+1}",
                "support": int(n11),
                "p_i": float(p_i),
                "p_j": float(p_j),
                "p_i_given_j": float(p_i_given_j),
                "p_j_given_i": float(p_j_given_i),
                "lift": float(lift),
                "phi": float(phi),
            })

    return edges


def compute_and_save_label_stats(train_items: list, cfg: dict, run_dir: Path):
    y_bin = _labels_from_train_items(train_items)
    if y_bin.size == 0:
        print("[Stats] No training items; skipping label stats.")
        return None, None

    n_images, num_classes = y_bin.shape
    names = _criteria_names(num_classes)

    train_cfg = cfg.get("train", {})
    min_pos_for_loss = int(train_cfg.get("min_pos_for_loss", 1))
    rare_pos_threshold = int(train_cfg.get("rare_pos_threshold", 10))
    rare_weight_factor = float(train_cfg.get("rare_weight_factor", 1.0))

    class_stats = compute_class_stats(
        y_bin=y_bin,
        min_pos_for_loss=min_pos_for_loss,
        rare_pos_threshold=rare_pos_threshold,
    )
    class_stats.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "criteria": names,
        "min_pos_for_loss": int(min_pos_for_loss),
        "rare_pos_threshold": int(rare_pos_threshold),
        "rare_weight_factor": float(rare_weight_factor),
    })

    run_dir = Path(run_dir)
    atomic_write_json(run_dir / "class_stats.json", class_stats)

    corr_cfg = cfg.get("correlation", {})
    min_support = int(corr_cfg.get("min_support", 5))
    min_lift = float(corr_cfg.get("min_lift", 2.0))

    edges = compute_correlation_edges(
        y_bin=y_bin,
        min_support=min_support,
        min_lift=min_lift,
        names=names,
    )

    corr_stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_images": int(n_images),
        "num_classes": int(num_classes),
        "criteria": names,
        "min_support": int(min_support),
        "min_lift": float(min_lift),
        "edges": edges,
    }

    atomic_write_json(run_dir / "correlation_stats.json", corr_stats)

    print(f"[Stats] Saved class_stats.json and correlation_stats.json to {run_dir}")
    return class_stats, corr_stats
