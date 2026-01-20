"""gating.py

Lightweight, non-ML gating for Draw-A-Man (DAM) uploads.

This module implements multiple cheap gates:
  1) Image-level heuristics (paper/ink/blank/tiny checks).
  2) Prediction-level heuristics (uncertainty + optional flip consistency).

The design is intentionally conservative: it aims to reject obvious non-DAM
inputs (photos, documents, blank pages, tiny images) while allowing a broad
range of DAM drawings.

Configuration
-------------
All functions accept an optional `cfg` dict. Missing keys fall back to
sensible defaults.

Image-level cfg keys:
  - min_side_px (int): minimum width/height.
  - white_thr (int): grayscale threshold for "near-white".
  - ink_thr (int): grayscale threshold for "ink" (darker-than).
  - min_ink_frac (float): reject if ink coverage below this (blank).
  - max_ink_frac (float): reject if ink coverage above this (overfilled).
  - min_mean_gray (float): reject if overall image is too dark.
  - min_soft_passes (int): number of soft gates required.
  - min_white_frac (float): soft gate (paper-like background).
  - min_edge_density (float): soft gate (line-art structure).
  - edge_thr (float): threshold on simple gradient magnitude.

Prediction-level cfg keys:
  - min_margin (float): mean(|p-0.5|) must exceed this.
  - use_tta_flip (bool): whether to compute flip predictions for stability.
  - max_flip_l1 (float): mean(|p - p_flip|) must be below this OR margin high.
  - margin_bypass_flip (float): if margin >= this, skip flip stability failure.

Return format
-------------
Each gate returns a dict: {"ok": bool, "code": str, "message": str, "metrics": dict}

CLI
---
  python gating.py --image path/to/file.png

This prints JSON gate results for image-level heuristics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GateResult:
    ok: bool
    code: str
    message: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "code": str(self.code),
            "message": str(self.message),
            "metrics": dict(self.metrics or {}),
        }


def _cfg_get(cfg: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    if not cfg:
        return default
    v = cfg.get(key, default)
    return default if v is None else v


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    # rgb: (H,W,3) float32 in [0,255]
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _rgb_to_saturation(rgb: np.ndarray) -> np.ndarray:
    # Approx HSV saturation: (max-min)/max
    rgb01 = rgb / 255.0
    mx = rgb01.max(axis=-1)
    mn = rgb01.min(axis=-1)
    sat = (mx - mn) / (mx + 1e-6)
    return sat.astype(np.float32)


def _edge_density(gray: np.ndarray, edge_thr: float) -> float:
    # Very cheap gradient magnitude thresholding (no conv libs required).
    gx = np.abs(np.diff(gray, axis=1))  # (H,W-1)
    gy = np.abs(np.diff(gray, axis=0))  # (H-1,W)

    # Pad back to (H,W) by repeating last column/row.
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")

    g = np.maximum(gx, gy)
    return float((g > edge_thr).mean())


def compute_image_features(img: Image.Image, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[0], rgb.shape[1]

    gray = _rgb_to_gray(rgb)
    sat = _rgb_to_saturation(rgb)

    # --- Adaptive Thresholding Start ---
    # Instead of hardcoded 245, we estimate the background brightness.
    # We assume the top 5% of pixels represent the paper color (ignores ink).
    bg_est = np.percentile(gray, 95)
    
    # "Ink" is defined as significantly darker than the paper background.
    # Gap of 45 intensity levels helps ignore faint shadows and faded text.
    default_ink_thr = max(0, int(bg_est - 45))
    
    # "White" is defined as close to the paper background.
    default_white_thr = max(0, int(bg_est - 20))
    
    ink_thr = int(_cfg_get(cfg, "ink_thr", default_ink_thr))
    white_thr = int(_cfg_get(cfg, "white_thr", default_white_thr))
    # --- Adaptive Thresholding End ---

    edge_thr = float(_cfg_get(cfg, "edge_thr", 18.0))

    white_frac = float((gray >= white_thr).mean())
    ink_frac = float((gray <= ink_thr).mean())
    mean_gray = float(gray.mean())
    sat_mean = float(sat.mean())
    edge_den = _edge_density(gray, edge_thr=edge_thr)

    # Colorfulness (Hasler-Susstrunk) - useful diagnostic only.
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    std_rg, std_yb = float(rg.std()), float(yb.std())
    mean_rg, mean_yb = float(np.abs(rg).mean()), float(np.abs(yb).mean())
    colorfulness = float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

    return {
        "width": w,
        "height": h,
        "white_frac": white_frac,
        "ink_frac": ink_frac,
        "mean_gray": mean_gray,
        "sat_mean": sat_mean,
        "edge_density": edge_den,
        "colorfulness": colorfulness,
        "white_thr": white_thr,
        "ink_thr": ink_thr,
        "edge_thr": edge_thr,
        "bg_est": float(bg_est),
    }


def gate_image(img: Image.Image, cfg: Optional[Dict[str, Any]] = None) -> GateResult:
    """Image-level gating (no model required)."""

    feats = compute_image_features(img, cfg)

    min_side_px = int(_cfg_get(cfg, "min_side_px", 256))
    min_ink_frac = float(_cfg_get(cfg, "min_ink_frac", 0.002))
    max_ink_frac = float(_cfg_get(cfg, "max_ink_frac", 0.35))
    
    # Lowered default minimum brightness to handle indoor lighting better (was 90.0)
    min_mean_gray = float(_cfg_get(cfg, "min_mean_gray", 60.0))

    # Soft gates: require at least N to pass.
    min_white_frac = float(_cfg_get(cfg, "min_white_frac", 0.45))
    min_edge_density = float(_cfg_get(cfg, "min_edge_density", 0.004))
    min_soft_passes = int(_cfg_get(cfg, "min_soft_passes", 1))

    w, h = int(feats["width"]), int(feats["height"])
    if min(w, h) < min_side_px:
        return GateResult(
            ok=False,
            code="too_small",
            message=f"Image is too small ({w}x{h}). Please upload a clearer photo of the drawing.",
            metrics=feats,
        )

    if feats["mean_gray"] < min_mean_gray:
        return GateResult(
            ok=False,
            code="too_dark",
            message="Image is too dark. Please retake in better lighting with the paper well lit.",
            metrics=feats,
        )

    if feats["ink_frac"] < min_ink_frac:
        return GateResult(
            ok=False,
            code="blank_or_low_ink",
            message="Image appears blank or too faint. Please upload the DAM drawing (clear pen/pencil strokes).",
            metrics=feats,
        )

    # --- Shadow Bypass Logic ---
    if feats["ink_frac"] > max_ink_frac:
        # If ink is too high, it might be a massive shadow or bad lighting.
        # Real "overfilled" images (like a photo of a room) usually have complex textures (high edge density).
        # Shadows are smooth (low edge density).
        # If edge_density is low enough, we assume it's a lighting artifact and PASS it.
        shadow_edge_limit = 0.06
        if feats["edge_density"] < shadow_edge_limit:
            # Pass with a note in metrics (implicitly allowed)
            feats["shadow_bypass"] = True
        else:
            return GateResult(
                ok=False,
                code="too_much_ink",
                message="Image appears too filled/dark (possibly not a paper drawing). Please upload a clear DAM drawing on paper.",
                metrics=feats,
            )

    soft_passes = []

    # Soft gate 1: paper-like background.
    soft_passes.append(feats["white_frac"] >= min_white_frac)

    # Soft gate 2: line-art structure.
    soft_passes.append(feats["edge_density"] >= min_edge_density)

    soft_pass_count = int(sum(bool(x) for x in soft_passes))
    feats["soft_pass_count"] = soft_pass_count
    feats["min_soft_passes"] = min_soft_passes

    if soft_pass_count < min_soft_passes:
        return GateResult(
            ok=False,
            code="not_drawing_like",
            message="Upload looks unlike a paper line drawing. Please upload a DAM drawing (paper + pen/pencil/crayon).",
            metrics=feats,
        )

    return GateResult(
        ok=True,
        code="ok",
        message="OK",
        metrics=feats,
    )


def gate_predictions(
    prob: np.ndarray,
    prob_flip: Optional[np.ndarray] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> GateResult:
    """Prediction-level gating.

    Intended to reject out-of-domain inputs where the checklist model is highly
    uncertain and/or unstable under a simple flip augmentation.
    """

    prob = np.asarray(prob, dtype=np.float32).reshape(-1)

    min_margin = float(_cfg_get(cfg, "min_margin", 0.06))
    use_tta_flip = bool(_cfg_get(cfg, "use_tta_flip", True))
    max_flip_l1 = float(_cfg_get(cfg, "max_flip_l1", 0.15))
    margin_bypass_flip = float(_cfg_get(cfg, "margin_bypass_flip", 0.12))

    margin = float(np.mean(np.abs(prob - 0.5)))

    metrics: Dict[str, Any] = {
        "margin": margin,
        "min_margin": min_margin,
    }

    if margin < min_margin:
        return GateResult(
            ok=False,
            code="low_confidence",
            message="Model is not confident this is a valid DAM drawing. Please upload a clear drawing of a person on paper.",
            metrics=metrics,
        )

    if use_tta_flip and prob_flip is not None:
        prob_flip = np.asarray(prob_flip, dtype=np.float32).reshape(-1)
        if prob_flip.shape != prob.shape:
            return GateResult(
                ok=False,
                code="tta_shape_mismatch",
                message="Internal error: TTA prediction shape mismatch.",
                metrics={**metrics, "prob_shape": list(prob.shape), "prob_flip_shape": list(prob_flip.shape)},
            )

        flip_l1 = float(np.mean(np.abs(prob - prob_flip)))
        metrics.update({"flip_l1": flip_l1, "max_flip_l1": max_flip_l1, "margin_bypass_flip": margin_bypass_flip})

        if flip_l1 > max_flip_l1 and margin < margin_bypass_flip:
            return GateResult(
                ok=False,
                code="unstable_under_flip",
                message="Upload looks out-of-domain (unstable prediction). Please retake the photo straight-on with good lighting.",
                metrics=metrics,
            )

    return GateResult(ok=True, code="ok", message="OK", metrics=metrics)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Run DAM image gating on a single image.")
    p.add_argument("--image", required=True, help="Path to an image file")
    args = p.parse_args()

    img = Image.open(args.image).convert("RGB")
    res = gate_image(img).to_dict()
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())