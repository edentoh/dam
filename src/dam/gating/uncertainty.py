import numpy as np
from typing import Any, Dict, Optional
from .common import GateResult, _cfg_get

def check_prediction_stability(
    prob: np.ndarray,
    prob_flip: Optional[np.ndarray] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> GateResult:
    """
    Checks if model prediction is confident and stable (consistent across horizontal flips).
    """
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)

    min_margin = float(_cfg_get(cfg, "min_margin", 0.06))
    max_flip_l1 = float(_cfg_get(cfg, "max_flip_l1", 0.15))
    margin_bypass_flip = float(_cfg_get(cfg, "margin_bypass_flip", 0.12))

    # Margin = Distance from 0.5 (uncertainty)
    margin = float(np.mean(np.abs(prob - 0.5)))

    metrics = {
        "margin": margin,
        "min_margin": min_margin,
    }

    if margin < min_margin:
        return GateResult(
            False, 
            "low_confidence", 
            "Model is not confident this is a valid drawing.", 
            metrics
        )

    if prob_flip is not None:
        prob_flip = np.asarray(prob_flip, dtype=np.float32).reshape(-1)
        flip_l1 = float(np.mean(np.abs(prob - prob_flip)))
        metrics["flip_l1"] = flip_l1

        # If prediction flips significantly and we aren't super confident, reject.
        if flip_l1 > max_flip_l1 and margin < margin_bypass_flip:
            return GateResult(
                False, 
                "unstable_under_flip", 
                "Prediction unstable under rotation/flip.", 
                metrics
            )

    return GateResult(True, "ok", "OK", metrics)