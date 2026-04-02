"""Singleton model loaders for the SilverBullet API — one per evaluation mode."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from backend.predict import SimilarityPredictor
from backend.model_hub import ensure_checkpoint

# Env-var name for each mode's model checkpoint.
# Falls back to MODEL_PATH (legacy) then best_model.pth if unset or file absent.
_MODE_ENV: dict[str, str] = {
    "model-vs-model":         "MODEL_PATH_MODEL_VS_MODEL",
    "reference-vs-generated": "MODEL_PATH_REFERENCE_VS_GENERATED",
    "context-vs-generated":   "MODEL_PATH_CONTEXT_VS_GENERATED",
}

# Default checkpoint names used when the env var is not set.
# New layout: models/{mode}/best.pth  (preferred)
# Legacy layout: models/{mode}.pth    (fallback for old checkpoints)
_MODE_DEFAULT: dict[str, str] = {
    "model-vs-model":         "models/model-vs-model/best.pth",
    "reference-vs-generated": "models/reference-vs-generated/best.pth",
    "context-vs-generated":   "models/context-vs-generated/best.pth",
}

_MODE_LEGACY: dict[str, str] = {
    "model-vs-model":         "models/model-vs-model.pth",
    "reference-vs-generated": "models/reference-vs-generated.pth",
    "context-vs-generated":   "models/context-vs-generated.pth",
}

# General fallback (backwards compat)
_FALLBACK = os.environ.get("MODEL_PATH", "best_model.pth")


def _resolve_model_path(mode: str) -> str:
    """Return the checkpoint path for *mode*, falling back to the general model.

    Resolution order:
      1. Env var  MODEL_PATH_{MODE_UPPER}
      2. models/{mode}/best.pth  (new layout)
      3. models/{mode}.pth       (legacy flat layout)
      4. best_model.pth / MODEL_PATH  (general fallback)
    """
    env_key = _MODE_ENV.get(mode)
    if env_key and os.environ.get(env_key):
        return os.environ[env_key]
    preferred = Path(_MODE_DEFAULT.get(mode, _FALLBACK))
    if preferred.exists():
        return str(preferred)
    legacy = Path(_MODE_LEGACY.get(mode, _FALLBACK))
    if legacy.exists():
        return str(legacy)
    return _FALLBACK


@lru_cache(maxsize=3)
def get_predictor(mode: str = "context-vs-generated") -> SimilarityPredictor:
    """Load and cache the SimilarityPredictor for *mode* (one instance per mode).

    Resolution order for the checkpoint path:
      1. Env var  MODEL_PATH_{MODE_UPPER}  (e.g. MODEL_PATH_CONTEXT_VS_GENERATED)
      2. models/{mode}/best.pth  (new layout — created by train.py --mode)
      3. models/{mode}.pth       (legacy flat layout — backward compat)
      4. best_model.pth / MODEL_PATH  (general fallback)

    If the preferred path (step 2) is absent and SB_HF_REPO_ID is set,
    the checkpoint is downloaded from HuggingFace Hub before loading.
    """
    preferred = _MODE_DEFAULT.get(mode, _FALLBACK)
    ensure_checkpoint(mode, preferred)   # no-op if file exists or Hub not configured
    path = _resolve_model_path(mode)
    return SimilarityPredictor(model_path=path)
