"""Pytest fixtures for SilverBullet API tests."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import huggingface_hub

# Prevent network login during test collection; resolveEntity imports `login` at module import time.
huggingface_hub.login = lambda *args, **kwargs: None

# Import app and the original get_predictor at module level — BEFORE any test
# patches are applied.  This ensures that:
#   1. api.main is fully imported with the real get_predictor function, so
#      Depends(get_predictor) in endpoint signatures captures the real function.
#   2. _original_get_predictor is the real function object, which is the correct
#      key for app.dependency_overrides.
from api.main import app  # noqa: E402 — must be first import of api.main
from api.dependencies import get_predictor as _original_get_predictor


@pytest.fixture
def mock_predictor():
    """Replace get_predictor so no model file is needed at test time."""
    mock_instance = MagicMock()
    mock_instance.predict_pair.return_value = {
        "prediction": 1,
        "probability": 0.85,
    }
    mock_instance.predict_batch.return_value = [
        {"prediction": 1, "probability": 0.85}
    ]

    with patch("api.dependencies.get_predictor", return_value=mock_instance), \
         patch("api.main.get_predictor", return_value=mock_instance):
        yield mock_instance


@pytest.fixture
def client(mock_predictor):
    """FastAPI TestClient with a mocked predictor — no real model needed."""
    # Use the original (pre-patch) function object so FastAPI finds the override
    # when resolving Depends(get_predictor) in endpoint signatures.
    app.dependency_overrides[_original_get_predictor] = lambda: mock_predictor
    yield TestClient(app)
    app.dependency_overrides.clear()
