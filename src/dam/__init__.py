"""
DAM (Draw-A-Man) Package.
A modular deep learning library for assessing developmental age from drawings.
"""

__version__ = "1.0.0"

# Expose key components for cleaner imports (e.g., `from dam import DAMPredictor`)
from .inference.predictor import DAMPredictor
from .core.config import load_config
from .gating.heuristics import gate_image_heuristics
from .gating.ml_gate import BinaryGate

__all__ = [
    "DAMPredictor",
    "load_config",
    "gate_image_heuristics",
    "BinaryGate",
]