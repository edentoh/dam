"""
Gating module.
Contains logic for filtering invalid inputs via heuristics (image metrics),
uncertainty (prediction stability), and ML classifiers (binary gate).
"""
from .common import GateResult
from .heuristics import gate_image_heuristics, compute_image_features
from .uncertainty import check_prediction_stability
from .ml_gate import BinaryGate

__all__ = [
    "GateResult",
    "gate_image_heuristics",
    "compute_image_features",
    "check_prediction_stability",
    "BinaryGate",
]