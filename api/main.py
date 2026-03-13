"""SilverBullet FastAPI application factory."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.dependencies import get_predictor
from api.middleware import LoggingMiddleware, RequestIDMiddleware
from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PairRequest,
    PairResponse,
)
from predict import SimilarityPredictor


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
# Rate limiter (infrastructure kept; per-endpoint decorators omitted due to
# slowapi/FastAPI signature-inspection incompatibility — add middleware later)
# ---------------------------------------------------------------------------

def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SilverBullet Similarity API",
    description=(
        "Learned text similarity / faithfulness scorer. "
        "Given two texts, returns a score in **[0, 1]**.\n\n"
        "Interactive docs: `/api/v1/docs` · ReDoc: `/api/v1/redoc`"
    ),
    version="1.0.0",
    contact={
        "name": "SilverBullet",
        "url": "https://github.com/Mehul-Gupta-SMH/Silver-Bullet",
    },
    openapi_tags=[
        {"name": "health", "description": "Service health and readiness checks"},
        {"name": "prediction", "description": "Text similarity scoring endpoints"},
    ],
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)


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
    """Return service status. No authentication required."""
    try:
        get_predictor()
        model_loaded = True
    except Exception:
        model_loaded = False
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.post(
    "/api/v1/predict/pair",
    response_model=PairResponse,
    tags=["prediction"],
    summary="Score a single text pair",
    response_description="Similarity score in [0, 1] with binary prediction",
    responses={422: {"description": "Validation error — text empty or exceeds 10 000 characters"}},
)
async def predict_pair(
    request: Request,
    body: PairRequest,
    predictor: Annotated[SimilarityPredictor, Depends(get_predictor)],
) -> PairResponse:
    """Score the similarity between two texts."""
    result = predictor.predict_pair(body.text1, body.text2)
    return PairResponse(**result)


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Score a batch of text pairs",
    response_description="List of similarity scores, one per input pair",
    responses={422: {"description": "Validation error — batch exceeds 100 pairs or pairs are malformed"}},
)
async def predict_batch(
    request: Request,
    body: BatchRequest,
    predictor: Annotated[SimilarityPredictor, Depends(get_predictor)],
) -> BatchResponse:
    """Score a list of text pairs. Max 100 pairs per request."""
    results = predictor.predict_batch(body.pairs)
    return BatchResponse(results=[PairResponse(**r) for r in results])
