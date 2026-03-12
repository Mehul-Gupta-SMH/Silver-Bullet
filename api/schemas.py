"""Pydantic schemas for the SilverBullet FastAPI layer."""

from pydantic import BaseModel


class PairRequest(BaseModel):
    text1: str
    text2: str


class BatchRequest(BaseModel):
    pairs: list[list[str]]


class PredictionResult(BaseModel):
    prediction: int
    probability: float
    text1: str
    text2: str


class BatchResponse(BaseModel):
    results: list[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
