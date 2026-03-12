"""Singleton model loader for the SilverBullet API."""

from functools import lru_cache
from pathlib import Path
import os

from predict import SimilarityPredictor


@lru_cache(maxsize=1)
def get_predictor() -> SimilarityPredictor:
    """Load and cache the SimilarityPredictor singleton.

    The model path is read from the MODEL_PATH env var, defaulting to
    'best_model.pth' relative to the project root.
    """
    model_path = Path(os.environ.get("MODEL_PATH", "best_model.pth"))
    return SimilarityPredictor(model_path=model_path)
