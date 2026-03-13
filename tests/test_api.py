"""API endpoint tests for SilverBullet.

These tests use a mocked SimilarityPredictor so no trained model file is
required.  All test coverage is against the HTTP contract (status codes,
response shapes, input validation, CORS headers).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import _original_get_predictor


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    """GET /api/v1/health should return 200 with status='ok'."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_health_reports_model_loaded_true_when_predictor_available(client):
    """Health response should set model_loaded=True when predictor loads."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_health_reports_model_not_loaded_on_predictor_failure(monkeypatch):
    """If predictor raises on load, health should stay 200 with model_loaded=False."""

    def _boom():
        raise RuntimeError("model missing")

    monkeypatch.setattr("api.main.get_predictor", _boom)
    monkeypatch.setattr("api.dependencies.get_predictor", _boom)

    with TestClient(app) as temp_client:
        response = temp_client.get("/api/v1/health")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


# ---------------------------------------------------------------------------
# Predict pair — happy path
# ---------------------------------------------------------------------------

def test_predict_pair_happy_path(client):
    """POST /api/v1/predict/pair with valid inputs should return 200 and include a probability."""
    response = client.post(
        "/api/v1/predict/pair",
        json={"text1": "Hello world", "text2": "Hi there"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "probability" in body


# ---------------------------------------------------------------------------
# Predict pair — validation errors (422 Unprocessable Entity)
# ---------------------------------------------------------------------------

def test_predict_pair_empty_text1_returns_422(client):
    """POST with an empty text1 should be rejected with 422."""
    response = client.post(
        "/api/v1/predict/pair",
        json={"text1": "", "text2": "hello"},
    )
    assert response.status_code == 422


def test_predict_pair_empty_text2_returns_422(client):
    """POST with an empty text2 should be rejected with 422."""
    response = client.post(
        "/api/v1/predict/pair",
        json={"text1": "hello", "text2": ""},
    )
    assert response.status_code == 422


def test_predict_pair_text_too_long_returns_422(client):
    """POST with text1 exceeding 10 000 characters should be rejected with 422."""
    long_text = "a" * 10_001
    response = client.post(
        "/api/v1/predict/pair",
        json={"text1": long_text, "text2": "hello"},
    )
    assert response.status_code == 422


def test_predict_pair_error_returns_500_and_request_id(mock_predictor):
    """If predictor throws, the global handler should return 500 and echo X-Request-ID."""
    mock_predictor.predict_pair.side_effect = ValueError("explode")

    app.dependency_overrides[_original_get_predictor] = lambda: mock_predictor
    try:
        with TestClient(app, raise_server_exceptions=False) as error_client:
            response = error_client.post(
                "/api/v1/predict/pair",
                json={"text1": "Hello", "text2": "Hi"},
                headers={"X-Request-ID": "req-123"},
            )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 500
    assert "detail" in body
    assert body["detail"] == "explode"
    assert "request_id" in body


# ---------------------------------------------------------------------------
# Predict batch — validation errors
# ---------------------------------------------------------------------------

def test_predict_batch_too_many_pairs_returns_422(client):
    """POST /api/v1/predict/batch with 101 pairs should be rejected with 422."""
    pairs = [["text one", "text two"]] * 101
    response = client.post(
        "/api/v1/predict/batch",
        json={"pairs": pairs},
    )
    assert response.status_code == 422


def test_predict_batch_happy_path(client, mock_predictor):
    """POST /api/v1/predict/batch should return mocked predictor results."""
    mock_predictor.predict_batch.return_value = [
        {"prediction": 1, "probability": 0.91},
        {"prediction": 0, "probability": 0.12},
    ]
    payload = [["alpha", "beta"], ["gamma", "delta"]]

    response = client.post("/api/v1/predict/batch", json={"pairs": payload})

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == mock_predictor.predict_batch.return_value
    mock_predictor.predict_batch.assert_called_once_with(payload)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

def test_cors_header_present(client):
    """A request from the allowed dev origin should receive the CORS header."""
    response = client.get(
        "/api/v1/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
