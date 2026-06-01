from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    checkpoint_exists: bool
    metrics_exists: bool
    device: str


class ProbabilityItem(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: list[ProbabilityItem]
    gradcam_image: str
    model_name: str
    image_size: int
    device: str
