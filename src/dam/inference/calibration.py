from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from dam.inference.thresholds import resolve_under_model_dir


class CorrelationCalibrator:
    def __init__(self, cfg: dict, model_path: Path, num_classes: int):
        self.enabled = bool(cfg.get("predict", {}).get("use_correlation_calibration", False))
        self.num_classes = int(num_classes)
        self.mode = str(cfg.get("correlation", {}).get("mode", "p_rare_given_common"))

        self.zero_pos_indices: list[int] = []
        self.rare_indices: list[int] = []
        self.edges: list[dict] = []

        pred_cfg = cfg.get("predict", {})
        class_stats_path = Path(pred_cfg.get("class_stats_path", "class_stats.json"))
        corr_path = Path(pred_cfg.get("correlation_stats_path", "correlation_stats.json"))

        self.class_stats_path = resolve_under_model_dir(Path(model_path), class_stats_path)
        self.corr_path = resolve_under_model_dir(Path(model_path), corr_path)

        self._load_class_stats()
        self._load_correlation_stats()

        if self.enabled and not self.edges:
            print("[Calib] Enabled but no valid correlation edges found.")

    def _load_class_stats(self) -> None:
        if not self.class_stats_path.exists():
            print(f"[Calib] class_stats.json not found: {self.class_stats_path}")
            return

        try:
            with open(self.class_stats_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[Calib] Failed to read class stats: {e}")
            return

        def _clean_indices(vals) -> list[int]:
            out = []
            for v in vals or []:
                try:
                    i = int(v)
                except Exception:
                    continue
                if 0 <= i < self.num_classes:
                    out.append(i)
            return out

        self.zero_pos_indices = _clean_indices(obj.get("zero_pos_indices"))
        self.rare_indices = _clean_indices(obj.get("rare_indices"))

    def _load_correlation_stats(self) -> None:
        if not self.corr_path.exists():
            if self.enabled:
                print(f"[Calib] correlation_stats.json not found: {self.corr_path}")
            return

        try:
            with open(self.corr_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[Calib] Failed to read correlation stats: {e}")
            return

        edges = obj.get("edges", [])
        if not isinstance(edges, list):
            return

        for e in edges:
            try:
                i = int(e.get("i"))
                j = int(e.get("j"))
            except Exception:
                continue
            if not (0 <= i < self.num_classes and 0 <= j < self.num_classes):
                continue
            self.edges.append({
                "i": i,
                "j": j,
                "p_i_given_j": float(e.get("p_i_given_j", 0.0)),
                "p_j_given_i": float(e.get("p_j_given_i", 0.0)),
            })

    def apply(self, probs: np.ndarray) -> np.ndarray:
        if probs is None:
            return probs

        arr = np.asarray(probs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != self.num_classes:
            return probs

        # Always zero-out criteria with 0 positives if available
        if self.zero_pos_indices:
            arr[:, self.zero_pos_indices] = 0.0

        if not self.enabled:
            return arr

        if not self.edges or not self.rare_indices:
            return arr

        rare_set = set(self.rare_indices)

        for b in range(arr.shape[0]):
            p = arr[b]
            for e in self.edges:
                i = e["i"]
                j = e["j"]
                if i in rare_set and j not in rare_set:
                    cond = e["p_i_given_j"] if self.mode == "p_rare_given_common" else max(e["p_i_given_j"], e["p_j_given_i"])
                    boost = p[j] * cond
                    if boost > p[i]:
                        p[i] = boost
                elif j in rare_set and i not in rare_set:
                    cond = e["p_j_given_i"] if self.mode == "p_rare_given_common" else max(e["p_j_given_i"], e["p_i_given_j"])
                    boost = p[i] * cond
                    if boost > p[j]:
                        p[j] = boost

            arr[b] = np.clip(p, 0.0, 1.0)

        return arr
