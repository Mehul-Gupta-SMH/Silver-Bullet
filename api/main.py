"""FastAPI serving layer for the SilverBullet SimilarityPredictor."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is importable before pulling in predict.py
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Depends, FastAPI, HTTPException, status  # noqa: E402

from api.dependencies import get_predictor  # noqa: E402
from api.schemas import (  # noqa: E402
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PairRequest,
    PredictionResult,
)
from predict import SimilarityPredictor  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting SilverBullet API")
    yield


app = FastAPI(title="SilverBullet API", version="1.0.0", lifespan=lifespan)


def _ensure_model_loaded(predictor: SimilarityPredictor | None) -> SimilarityPredictor:
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure MODEL_PATH points to a valid checkpoint.",
        )
    return predictor


@app.get("/api/v1/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_loaded = get_predictor() is not None
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.post("/api/v1/predict/pair", response_model=PredictionResult)
def predict_pair(
    request: PairRequest, predictor: SimilarityPredictor | None = Depends(get_predictor)
) -> PredictionResult:
    predictor = _ensure_model_loaded(predictor)
    result = predictor.predict_pair(request.text1, request.text2)
    return PredictionResult(**result)


@app.post("/api/v1/predict/batch", response_model=BatchResponse)
def predict_batch(
    request: BatchRequest, predictor: SimilarityPredictor | None = Depends(get_predictor)
) -> BatchResponse:
    predictor = _ensure_model_loaded(predictor)
    results = predictor.predict_batch(request.pairs)
    return BatchResponse(results=[PredictionResult(**item) for item in results])
