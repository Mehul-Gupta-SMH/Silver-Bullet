"""Singleton model loaders for the SilverBullet API — one per evaluation mode."""

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

# Env-var name for each mode's model checkpoint.
# Falls back to MODEL_PATH (legacy) then best_model.pth if unset or file absent.
_MODE_ENV: dict[str, str] = {
    "model-vs-model":         "MODEL_PATH_MODEL_VS_MODEL",
    "reference-vs-generated": "MODEL_PATH_REFERENCE_VS_GENERATED",
    "context-vs-generated":   "MODEL_PATH_CONTEXT_VS_GENERATED",
}

# Default checkpoint names used when the env var is not set
_MODE_DEFAULT: dict[str, str] = {
    "model-vs-model":         "models/model-vs-model.pth",
    "reference-vs-generated": "models/reference-vs-generated.pth",
    "context-vs-generated":   "models/context-vs-generated.pth",
}

# General fallback (backwards compat)
_FALLBACK = os.environ.get("MODEL_PATH", "best_model.pth")


def _resolve_model_path(mode: str) -> str:
    """Return the checkpoint path for *mode*, falling back to the general model."""
    env_key = _MODE_ENV.get(mode)
    if env_key and os.environ.get(env_key):
        return os.environ[env_key]
    mode_default = Path(_MODE_DEFAULT.get(mode, _FALLBACK))
    if mode_default.exists():
        return str(mode_default)
    return _FALLBACK


@lru_cache(maxsize=3)
def get_predictor(mode: str = "context-vs-generated") -> SimilarityPredictor:
    """Load and cache the SimilarityPredictor for *mode* (one instance per mode).

    Resolution order for the checkpoint path:
      1. Env var  MODEL_PATH_{MODE_UPPER}  (e.g. MODEL_PATH_CONTEXT_VS_GENERATED)
      2. models/{mode}.pth  (default per-mode path, created by train.py --mode)
      3. best_model.pth / MODEL_PATH  (legacy general model — used when a mode-specific
         checkpoint has not been trained yet)
    """
    path = _resolve_model_path(mode)
    return SimilarityPredictor(model_path=path)
