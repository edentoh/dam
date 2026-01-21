"""
Modeling module.
Handles model instantiation, weight loading, and architecture utilities.
"""
from .builder import ModelBuilder, build_model
from .utils import resolve_classifier_modules, infer_in_channels

__all__ = [
    "ModelBuilder",
    "build_model",
    "resolve_classifier_modules",
    "infer_in_channels",
]