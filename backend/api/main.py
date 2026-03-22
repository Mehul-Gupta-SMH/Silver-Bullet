"""SilverBullet FastAPI application factory."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from backend.api.dependencies import _MODE_ENV, get_predictor
from backend.api.middleware import LoggingMiddleware, RequestIDMiddleware
from backend.api.schemas import (
    BatchBreakdownRequest,
    BatchBreakdownResponse,
    BatchRequest,
    BatchResponse,
    BreakdownResponse,
    HealthResponse,
    PairRequest,
    PairResponse,
)


# ---------------------------------------------------------------------------
# Lifespan — startup model check
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Fail fast on startup if MODEL_PATH is explicitly set but the file is missing."""
    model_env = os.environ.get("MODEL_PATH")
    if model_env:
        model_path = Path(model_env)
        if not model_path.exists():
            raise RuntimeError(
                f"MODEL_PATH={model_env!r} does not exist or is not readable. "
                "Ensure the file path points to a valid checkpoint."
            )
    yield


# ---------------------------------------------------------------------------
# Rate limiter
#   Global default: 60/minute (all prediction endpoints)
#   Breakdown endpoints override to a tighter limit (expensive — reruns pipeline)
# ---------------------------------------------------------------------------

def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SilverBullet — LLM Evaluation API",
    description=(
        "Real-time LLM evaluation benchmark. "
        "Given two texts, returns a faithfulness / agreement score in **[0, 1]** "
        "using a Conv2D model trained on 16 multi-signal feature maps "
        "(semantic, lexical, NLI, entity, LCS).\n\n"
        "**Use cases:** hallucination detection, model-vs-model agreement, RAG groundedness.\n\n"
        "Interactive docs: `/api/v1/docs` · ReDoc: `/api/v1/redoc`"
    ),
    version="1.0.0",
    contact={
        "name": "SilverBullet",
        "url": "https://github.com/Mehul-Gupta-SMH/Silver-Bullet",
    },
    openapi_tags=[
        {"name": "health", "description": "Service health and readiness checks"},
        {"name": "prediction", "description": "LLM evaluation scoring endpoints"},
    ],
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(getattr(request, "state", None), "request_id", None)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_default_origins = ["http://localhost:5173"]
_extra = os.environ.get("ALLOWED_ORIGINS", "")
_allowed_origins = _default_origins + [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    response_description="Service status and model load state",
)
async def health() -> HealthResponse:
    """Return service status and per-mode model load state. No authentication required."""
    models: dict[str, bool] = {}
    for mode in _MODE_ENV:
        try:
            get_predictor(mode)
            models[mode] = True
        except Exception:
            models[mode] = False
    return HealthResponse(status="ok", model_loaded=any(models.values()), models=models)


@app.post(
    "/api/v1/predict/pair",
    response_model=PairResponse,
    tags=["prediction"],
    summary="Evaluate a single LLM output",
    response_description="Faithfulness / agreement score in [0, 1] with binary verdict",
    responses={422: {"description": "Validation error — text empty or exceeds 10 000 characters"}},
)
async def predict_pair(
    request: Request,
    body: PairRequest,
) -> PairResponse:
    """Score the similarity between two texts using the model for the requested evaluation mode."""
    predictor = get_predictor(body.mode)
    result = predictor.predict_pair(body.text1, body.text2)
    return PairResponse(**result)


@app.post(
    "/api/v1/predict/pair/breakdown",
    response_model=BreakdownResponse,
    tags=["prediction"],
    summary="Sentence-level divergence analysis for a text pair",
    response_description=(
        "Per-sentence alignment matrix, divergence indices, and feature-group scores. "
        "More expensive than /predict/pair — runs the full feature pipeline without cache."
    ),
    responses={422: {"description": "Validation error — text empty or exceeds 10 000 characters"}},
)
@limiter.limit("20/minute")
async def predict_pair_breakdown(
    request: Request,
    body: PairRequest,
) -> BreakdownResponse:
    """Return a full divergence breakdown: sentence splits, alignment matrix,
    divergent sentence indices, and per-feature-group similarity scores."""
    predictor = get_predictor(body.mode)
    result = predictor.predict_pair_breakdown(body.text1, body.text2)
    return BreakdownResponse(**result)


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Evaluate a batch of LLM outputs",
    response_description="List of faithfulness / agreement scores, one per input pair",
    responses={422: {"description": "Validation error — batch exceeds 100 pairs or pairs are malformed"}},
)
async def predict_batch(
    request: Request,
    body: BatchRequest,
) -> BatchResponse:
    """Score a list of text pairs using the model for the requested evaluation mode. Max 100 pairs."""
    predictor = get_predictor(body.mode)
    results = predictor.predict_batch(body.pairs)
    return BatchResponse(results=[PairResponse(**r) for r in results])


@app.post(
    "/api/v1/predict/batch/breakdown",
    response_model=BatchBreakdownResponse,
    tags=["prediction"],
    summary="Sentence-level divergence analysis for a batch of text pairs",
    response_description=(
        "Per-sentence alignment matrix, divergence indices, and feature-group scores for each pair. "
        "Expensive — reruns the full feature pipeline per pair. Max 10 pairs."
    ),
    responses={422: {"description": "Validation error — batch exceeds 10 pairs or pairs are malformed"}},
)
@limiter.limit("10/minute")
async def predict_batch_breakdown(
    request: Request,
    body: BatchBreakdownRequest,
) -> BatchBreakdownResponse:
    """Return full divergence breakdowns for up to 10 text pairs using the mode-specific model."""
    predictor = get_predictor(body.mode)
    results = predictor.predict_batch_breakdown(body.pairs)
    return BatchBreakdownResponse(results=[BreakdownResponse(**r) for r in results])
