"""Pydantic request/response models for the SilverBullet API."""

from pydantic import BaseModel, Field


class PairRequest(BaseModel):
    text1: str = Field(..., description="First text input")
    text2: str = Field(..., description="Second text input")


class PairResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction: 1=similar, 0=not similar")
    probability: float = Field(..., description="Similarity score in [0, 1]")
    text1: str
    text2: str


class BatchRequest(BaseModel):
    pairs: list[list[str]] = Field(
        ...,
        description="List of [text1, text2] pairs",
        min_length=1,
    )


class BatchResponse(BaseModel):
    results: list[PairResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
