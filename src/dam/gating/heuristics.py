import numpy as np
from PIL import Image
from typing import Any, Dict, Optional

from .common import GateResult, _cfg_get

def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)

def _rgb_to_saturation(rgb: np.ndarray) -> np.ndarray:
    rgb01 = rgb / 255.0
    mx = rgb01.max(axis=-1)
    mn = rgb01.min(axis=-1)
    sat = (mx - mn) / (mx + 1e-6)
    return sat.astype(np.float32)

def _edge_density(gray: np.ndarray, edge_thr: float) -> float:
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
    g = np.maximum(gx, gy)
    return float((g > edge_thr).mean())

def compute_image_features(img: Image.Image, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[0], rgb.shape[1]

    gray = _rgb_to_gray(rgb)
    sat = _rgb_to_saturation(rgb)

    # Adaptive thresholds
    bg_est = np.percentile(gray, 95)
    default_ink_thr = max(0, int(bg_est - 45))
    default_white_thr = max(0, int(bg_est - 20))
    
    ink_thr = int(_cfg_get(cfg, "ink_thr", default_ink_thr))
    white_thr = int(_cfg_get(cfg, "white_thr", default_white_thr))
    edge_thr = float(_cfg_get(cfg, "edge_thr", 18.0))

    return {
        "width": w,
        "height": h,
        "white_frac": float((gray >= white_thr).mean()),
        "ink_frac": float((gray <= ink_thr).mean()),
        "mean_gray": float(gray.mean()),
        "sat_mean": float(sat.mean()),
        "edge_density": _edge_density(gray, edge_thr=edge_thr),
        "white_thr": white_thr,
        "ink_thr": ink_thr,
        "bg_est": float(bg_est),
    }

def gate_image_heuristics(img: Image.Image, cfg: Optional[Dict[str, Any]] = None) -> GateResult:
    """
    Checks basic image quality (size, brightness, ink density).
    Rejects blank pages, dark photos, or non-drawings.
    """
    feats = compute_image_features(img, cfg)

    min_side_px = int(_cfg_get(cfg, "min_side_px", 256))
    min_mean_gray = float(_cfg_get(cfg, "min_mean_gray", 60.0))
    min_ink_frac = float(_cfg_get(cfg, "min_ink_frac", 0.002))
    max_ink_frac = float(_cfg_get(cfg, "max_ink_frac", 0.35))
    
    # Soft gates
    min_white_frac = float(_cfg_get(cfg, "min_white_frac", 0.45))
    min_edge_density = float(_cfg_get(cfg, "min_edge_density", 0.004))
    min_soft_passes = int(_cfg_get(cfg, "min_soft_passes", 1))

    w, h = int(feats["width"]), int(feats["height"])

    if min(w, h) < min_side_px:
        return GateResult(False, "too_small", f"Image too small ({w}x{h}).", feats)

    if feats["mean_gray"] < min_mean_gray:
        return GateResult(False, "too_dark", "Image is too dark.", feats)

    if feats["ink_frac"] < min_ink_frac:
        return GateResult(False, "blank_or_low_ink", "Image appears blank or too faint.", feats)

    if feats["ink_frac"] > max_ink_frac:
        # Shadow Bypass: If edges are low, it's likely a shadow, not complex texture
        if feats["edge_density"] < 0.06:
            feats["shadow_bypass"] = True
        else:
            return GateResult(False, "too_much_ink", "Image too filled/dark.", feats)

    soft_passes = 0
    if feats["white_frac"] >= min_white_frac: soft_passes += 1
    if feats["edge_density"] >= min_edge_density: soft_passes += 1
    
    feats["soft_pass_count"] = soft_passes
    
    if soft_passes < min_soft_passes:
        return GateResult(False, "not_drawing_like", "Does not look like a line drawing.", feats)

    return GateResult(True, "ok", "OK", feats)