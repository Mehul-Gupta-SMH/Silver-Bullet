"""Pytest fixtures for SilverBullet API tests."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import huggingface_hub

# Prevent network login during test collection; resolveEntity imports `login` at module import time.
huggingface_hub.login = lambda *args, **kwargs: None

from api.main import app  # noqa: E402


@pytest.fixture
def mock_predictor():
    """Patch get_predictor in api.main so no model file is needed at test time.

    Since endpoints now call get_predictor(mode) directly (not via Depends),
    we patch the function in api.main to return the same mock regardless of mode.
    """
    mock_instance = MagicMock()
    mock_instance.predict_pair.return_value = {
        "prediction": 1,
        "probability": 0.85,
    }
    mock_instance.predict_pair_breakdown.return_value = {
        "prediction": 1,
        "probability": 0.85,
        "sentences1": ["Hello world"],
        "sentences2": ["Hi there"],
        "alignment": [[0.9]],
        "divergent_in_1": [],
        "divergent_in_2": [],
        "feature_scores": {"Semantic (mxbai)": 0.9},
    }
    mock_instance.predict_batch.return_value = [
        {"prediction": 1, "probability": 0.85}
    ]
    mock_instance.predict_batch_breakdown.return_value = [
        {
            "prediction": 1,
            "probability": 0.85,
            "sentences1": ["Hello world"],
            "sentences2": ["Hi there"],
            "alignment": [[0.9]],
            "divergent_in_1": [],
            "divergent_in_2": [],
            "feature_scores": {"Semantic (mxbai)": 0.9},
        }
    ]

    # get_predictor(mode) is called directly in endpoint handlers — patch it in api.main.
    with patch("api.main.get_predictor", return_value=mock_instance):
        yield mock_instance


@pytest.fixture
def client(mock_predictor):
    """FastAPI TestClient with a mocked predictor — no real model needed."""
    yield TestClient(app)
