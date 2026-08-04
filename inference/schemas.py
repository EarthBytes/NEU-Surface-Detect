"""Request and response schemas for the inference API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: List[str] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class MonitoringSummaryResponse(BaseModel):
    enabled: bool
    log_path: str
    prediction_count: int
    average_confidence: Optional[float] = None
    average_brightness: Optional[float] = None
    low_confidence_rate: Optional[float] = None
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    recent_alerts: List[Dict[str, str]] = Field(default_factory=list)
