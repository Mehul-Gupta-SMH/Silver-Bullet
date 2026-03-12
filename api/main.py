"""SilverBullet FastAPI application factory."""

import os
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


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a JSON 429 response consistent with FastAPI's error envelope."""
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SilverBullet",
    description="Learned text-similarity / faithfulness scorer",
    version="1.0.0",
)

# Attach the slowapi limiter and its 429 handler.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

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

# ---------------------------------------------------------------------------
# Custom middleware (order matters: outermost = first to run)
# ---------------------------------------------------------------------------
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Health-check endpoint — no rate limiting applied."""
    try:
        get_predictor()
        model_loaded = True
    except Exception:
        model_loaded = False
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.post("/api/v1/predict/pair", response_model=PairResponse, tags=["predict"])
@limiter.limit("30/minute")
async def predict_pair(
    request: Request,
    body: PairRequest,
    predictor: Annotated[SimilarityPredictor, Depends(get_predictor)],
) -> PairResponse:
    """Score a single text pair. Rate-limited to 30 requests/minute per IP."""
    result = predictor.predict_pair(body.text1, body.text2)
    return PairResponse(**result)


@app.post("/api/v1/predict/batch", response_model=BatchResponse, tags=["predict"])
@limiter.limit("30/minute")
async def predict_batch(
    request: Request,
    body: BatchRequest,
    predictor: Annotated[SimilarityPredictor, Depends(get_predictor)],
) -> BatchResponse:
    """Score a list of text pairs. Rate-limited to 30 requests/minute per IP."""
    results = predictor.predict_batch(body.pairs)
    return BatchResponse(results=[PairResponse(**r) for r in results])
