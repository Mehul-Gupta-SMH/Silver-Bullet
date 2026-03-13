"""Pydantic request/response models for the SilverBullet API."""

from pydantic import BaseModel, Field


class PairRequest(BaseModel):
    text1: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="First text (source / reference)",
        examples=["The sky is blue."],
    )
    text2: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Second text (hypothesis / generated output)",
        examples=["The sky appears blue in colour."],
    )


class PairResponse(BaseModel):
    prediction: int = Field(
        ...,
        description="Binary prediction: 1 = similar, 0 = different",
        examples=[1],
    )
    probability: float = Field(
        ...,
        description="Similarity score in [0, 1]",
        examples=[0.87],
    )


class BatchRequest(BaseModel):
    pairs: list[list[str]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of [text1, text2] string pairs to score (max 100 per request)",
        examples=[[["The sky is blue.", "The sky appears blue in colour."]]]
    )


class BatchResponse(BaseModel):
    results: list[PairResponse] = Field(
        ...,
        description="Similarity scores, one per input pair",
    )


class BreakdownResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction: 1 = similar, 0 = different")
    probability: float = Field(..., description="Overall similarity score in [0, 1]")
    sentences1: list[str] = Field(..., description="Sentences split from text1")
    sentences2: list[str] = Field(..., description="Sentences split from text2")
    alignment: list[list[float]] = Field(
        ...,
        description="n×m semantic cosine-similarity matrix (sentences1 × sentences2)",
    )
    divergent_in_1: list[int] = Field(
        ...,
        description="Indices into sentences1 whose best alignment to any sentence in text2 is below 0.5",
    )
    divergent_in_2: list[int] = Field(
        ...,
        description="Indices into sentences2 whose best alignment to any sentence in text1 is below 0.5",
    )
    feature_scores: dict[str, float] = Field(
        ...,
        description="Per-feature-group mean best-match score (higher = more similar)",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status", examples=["ok"])
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
