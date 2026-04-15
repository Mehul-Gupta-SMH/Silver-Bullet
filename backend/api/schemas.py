"""Pydantic request/response models for the SilverBullet API."""

from typing import Literal

from pydantic import BaseModel, Field

# Evaluation mode — controls which trained model is used for scoring.
# Each mode targets a distinct use case with its own training distribution.
EvaluationMode = Literal[
    "model-vs-model",          # Agreement between two LLM outputs (same prompt)
    "reference-vs-generated",  # Faithfulness of generated answer vs ground-truth reference
    "context-vs-generated",    # Groundedness of generated answer against source context (RAG/hallucination)
]

_MODE_DESCRIPTION = (
    "Evaluation mode — selects the mode-specific trained model.\n"
    "• `model-vs-model`: score agreement between two LLM outputs for the same prompt\n"
    "• `reference-vs-generated`: score faithfulness of a generated answer against a reference\n"
    "• `context-vs-generated`: detect hallucinations by checking if an answer is grounded in context"
)


class PairRequest(BaseModel):
    text1: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="First text (source / reference / context)",
        examples=["The sky is blue."],
    )
    text2: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Second text (hypothesis / generated output)",
        examples=["The sky appears blue in colour."],
    )
    mode: EvaluationMode = Field(
        "context-vs-generated",
        description=_MODE_DESCRIPTION,
    )


class PairResponse(BaseModel):
    prediction: int = Field(
        ...,
        description="Binary prediction: 1 = similar/faithful/grounded, 0 = different/unfaithful/hallucinated",
        examples=[1],
    )
    probability: float = Field(
        ...,
        description="Score in [0, 1]",
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
    mode: EvaluationMode = Field(
        "context-vs-generated",
        description=_MODE_DESCRIPTION,
    )


class BatchResponse(BaseModel):
    results: list[PairResponse] = Field(
        ...,
        description="Similarity scores, one per input pair",
    )


class MisalignmentReason(BaseModel):
    label: str = Field(..., description="Short title for the misalignment signal")
    description: str = Field(..., description="Plain-English explanation of the signal")
    severity: str = Field(..., description="'high' | 'medium' | 'low'")
    signal: str = Field(..., description="Feature group that triggered the reason")


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
    min_alignment: float = Field(
        0.0,
        description="Minimum alignment score across all sentence pairs (weakest link)",
    )
    min_alignment_pair: list[int] = Field(
        default_factory=list,
        description="[i, j] indices of the weakest-aligned sentence pair",
    )
    feature_scores: dict[str, float] = Field(
        ...,
        description="Per-feature-group mean best-match score (higher = more similar)",
    )
    misalignment_reasons: list[MisalignmentReason] = Field(
        default_factory=list,
        description="Rule-based diagnostics explaining why texts diverge, ranked by severity",
    )


class BatchBreakdownRequest(BaseModel):
    pairs: list[list[str]] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of [text1, text2] string pairs (max 10 per request — breakdown reruns the full pipeline per pair)",
        examples=[[["The sky is blue.", "The sky appears blue in colour."]]]
    )
    mode: EvaluationMode = Field(
        "context-vs-generated",
        description=_MODE_DESCRIPTION,
    )


class BatchBreakdownResponse(BaseModel):
    results: list[BreakdownResponse] = Field(
        ...,
        description="Breakdown analysis, one per input pair",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status", examples=["ok"])
    model_loaded: bool = Field(..., description="Whether at least one mode model is loaded and ready")
    models: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-mode model load state — keys are evaluation mode IDs",
        examples=[{"model-vs-model": True, "reference-vs-generated": True, "context-vs-generated": True}],
    )


# ---------------------------------------------------------------------------
# Jury (LLM-as-judge) schemas
# ---------------------------------------------------------------------------

class JuryQuestion(BaseModel):
    """A single binary yes/no question posed to the LLM jury."""

    question: str = Field(..., description="The question text posed to the LLM")
    answer: Literal["yes", "no"] = Field(..., description="LLM answer: 'yes' or 'no'")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM's self-reported confidence in the answer (0–1)",
    )
    reasoning: str = Field(..., description="One-sentence explanation from the LLM")


class JuryResult(BaseModel):
    """Aggregated result from the LLM jury evaluation of a single text pair."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregated faithfulness score in [0, 1] (higher = more faithful/grounded)",
    )
    verdict: Literal["faithful", "hallucinated"] = Field(
        ...,
        description="'faithful' if score >= 0.5, otherwise 'hallucinated'",
    )
    questions: list[JuryQuestion] = Field(
        ...,
        description="Per-question breakdown of the jury's evaluation",
    )
    model_used: str = Field(
        ...,
        description="Name of the LLM used as the jury (e.g. 'gpt-4o-mini')",
    )


class JuryRequest(BaseModel):
    """Request body for a single jury evaluation."""

    text1: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="First text (source / reference / context)",
        examples=["The Eiffel Tower is located in Paris, France."],
    )
    text2: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Second text (hypothesis / generated output)",
        examples=["The Eiffel Tower can be found in Paris, France."],
    )
    mode: EvaluationMode = Field(
        "context-vs-generated",
        description=_MODE_DESCRIPTION,
    )


class JuryBatchRequest(BaseModel):
    """Request body for a batch jury evaluation (up to 10 pairs)."""

    pairs: list[JuryRequest] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "List of text pairs to evaluate (max 10 per request — each pair makes one LLM call)"
        ),
    )


class JuryBatchResponse(BaseModel):
    """Response body for a batch jury evaluation."""

    results: list[JuryResult] = Field(
        ...,
        description="Jury evaluation results, one per input pair in the same order",
    )
