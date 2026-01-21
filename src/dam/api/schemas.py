from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ItemPrediction(BaseModel):
    item: int
    prob: float
    threshold: float
    pass_: int = Field(..., alias="pass")  # "pass" is a reserved keyword in Python

class IsDamResult(BaseModel):
    prob: float
    threshold: float
    pass_: int = Field(..., alias="pass")

class GatingMetrics(BaseModel):
    image: Dict[str, Any]
    prediction: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    device: str
    backbone: str
    img_size: int
    num_classes: int
    threshold_mode: str
    threshold_vector_path: str
    threshold_scalar_fallback: float
    require_threshold_vector: bool
    rate_limit: Dict[str, int]
    max_upload_bytes: int
    gating: Dict[str, Any]
    is_dam: Dict[str, Any]

class PredictResponse(BaseModel):
    filename: str
    threshold_mode: str
    threshold_vector_path: str
    threshold_scalar_fallback: float
    total_score: int
    items: List[ItemPrediction]
    
    # Optional fields (included based on configuration/success)
    is_dam_prob: Optional[float] = None
    is_dam: Optional[IsDamResult] = None
    gating: Optional[GatingMetrics] = None