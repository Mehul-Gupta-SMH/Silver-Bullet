"""Pytest fixtures for SilverBullet API tests."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_predictor():
    """Patch get_predictor so no model file is needed.

    get_predictor is an lru_cache singleton used both directly in the health
    endpoint and via FastAPI Depends() in the predict endpoints.  We patch it
    at both import locations so the TestClient never tries to load a .pth file.
    """
    mock_instance = MagicMock()
    mock_instance.predict_pair.return_value = {
        "prediction": 1,
        "probability": 0.85,
        "text1": "Hello world",
        "text2": "Hi there",
    }
    mock_instance.predict_batch.return_value = [
        {
            "prediction": 1,
            "probability": 0.85,
            "text1": "Hello world",
            "text2": "Hi there",
        }
    ]

    with patch("api.dependencies.get_predictor", return_value=mock_instance) as p1, \
         patch("api.main.get_predictor", return_value=mock_instance) as p2:
        yield mock_instance


@pytest.fixture
def client(mock_predictor):
    """FastAPI TestClient with a mocked predictor — no real model needed."""
    from api.main import app
    # Override the FastAPI dependency so Depends(get_predictor) resolves to
    # the mock instance instead of trying to load best_model.pth.
    from api.dependencies import get_predictor as _real_get_predictor
    app.dependency_overrides[_real_get_predictor] = lambda: mock_predictor
    yield TestClient(app)
    app.dependency_overrides.clear()
