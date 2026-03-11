"""Shared dependencies for the FastAPI layer."""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so predict.py is importable when the api package is executed directly.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import SimilarityPredictor  # noqa: E402

logger = logging.getLogger(__name__)


def _get_model_path() -> Path:
    """Resolve the model path from the environment with a sensible default."""
    return Path(os.environ.get("MODEL_PATH", "best_model.pth"))


@lru_cache(maxsize=1)
def _load_predictor() -> Optional[SimilarityPredictor]:
    """Load the SimilarityPredictor once; return None if the model file is missing."""
    model_path = _get_model_path()
    if not model_path.exists():
        logger.warning("MODEL_PATH %s does not exist; predictor not loaded", model_path)
        return None

    try:
        return SimilarityPredictor(str(model_path))
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to load model from %s", model_path)
        return None


def get_predictor() -> Optional[SimilarityPredictor]:
    """FastAPI dependency to retrieve the singleton predictor instance."""
    return _load_predictor()
