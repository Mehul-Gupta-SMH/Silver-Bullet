"""Singleton model loader for the SilverBullet API."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# Ensure the project root is importable so predict.py can be found
# whether the server is started from inside or outside the repo.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import SimilarityPredictor  # noqa: E402


@lru_cache(maxsize=1)
def get_predictor() -> SimilarityPredictor:
    """Load and cache the SimilarityPredictor singleton.

    Model path is read from MODEL_PATH env var, defaulting to 'best_model.pth'
    relative to the working directory.
    """
    model_path = Path(os.environ.get("MODEL_PATH", "best_model.pth"))
    return SimilarityPredictor(model_path=str(model_path))
