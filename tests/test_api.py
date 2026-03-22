"""API endpoint tests for SilverBullet.

These tests use a mocked SimilarityPredictor so no trained model file is
required.  All test coverage is against the HTTP contract (status codes,
response shapes, input validation, CORS headers).
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.api.main import app


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


def test_health_reports_per_mode_models(client):
    """Health response should include a 'models' dict with per-mode load state."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    expected_modes = {"model-vs-model", "reference-vs-generated", "context-vs-generated"}
    assert set(body["models"].keys()) == expected_modes


def test_health_reports_model_not_loaded_on_predictor_failure(monkeypatch):
    """If predictor raises on load, health should stay 200 with model_loaded=False."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("model missing")

    monkeypatch.setattr("backend.api.main.get_predictor", _boom)

    with TestClient(app) as temp_client:
        response = temp_client.get("/api/v1/health")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_health_response_keys_are_strict(client):
    """Health payload should expose only the documented keys."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert set(response.json().keys()) == {"status", "model_loaded", "models"}


# ---------------------------------------------------------------------------
# Predict pair — happy path
# ---------------------------------------------------------------------------

def test_predict_pair_happy_path(client):
    """POST /api/v1/predict/pair with valid inputs should return 200 and include a probability."""
    response = client.post(
        "/api/v1/predict/pair",
        json={"text1": "Hello world", "text2": "Hi there", "mode": "context-vs-generated"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "probability" in body


def test_predict_pair_default_mode_is_accepted(client):
    """POST without 'mode' should use the default and return 200."""
    response = client.post(
        "/api/v1/predict/pair",
        json={"text1": "Hello world", "text2": "Hi there"},
    )
    assert response.status_code == 200


def test_predict_pair_invalid_mode_returns_422(client):
    """POST with an unknown mode value should be rejected with 422."""
    response = client.post(
        "/api/v1/predict/pair",
        json={"text1": "Hello", "text2": "Hi", "mode": "invalid-mode"},
    )
    assert response.status_code == 422


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


def test_predict_pair_missing_text1_returns_422(client):
    """POST missing text1 should be rejected with 422."""
    response = client.post(
        "/api/v1/predict/pair",
        json={"text2": "hello"},
    )
    assert response.status_code == 422


def test_predict_pair_missing_text2_returns_422(client):
    """POST missing text2 should be rejected with 422."""
    response = client.post(
        "/api/v1/predict/pair",
        json={"text1": "hello"},
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

    with patch("backend.api.main.get_predictor", return_value=mock_predictor), \
         TestClient(app, raise_server_exceptions=False) as error_client:
        response = error_client.post(
            "/api/v1/predict/pair",
            json={"text1": "Hello", "text2": "Hi"},
            headers={"X-Request-ID": "req-123"},
        )

    body = response.json()
    assert response.status_code == 500
    assert "detail" in body
    assert body["detail"] == "explode"
    assert "request_id" in body


# ---------------------------------------------------------------------------
# Predict pair breakdown
# ---------------------------------------------------------------------------

def test_predict_pair_breakdown_happy_path(client, mock_predictor):
    """POST /predict/pair/breakdown should return the full breakdown payload."""
    expected = {
        "prediction": 1,
        "probability": 0.75,
        "sentences1": ["Hello world"],
        "sentences2": ["Hi there"],
        "alignment": [[0.8]],
        "divergent_in_1": [],
        "divergent_in_2": [],
        "feature_scores": {"Semantic (mxbai)": 0.8},
    }
    mock_predictor.predict_pair_breakdown.return_value = expected

    response = client.post(
        "/api/v1/predict/pair/breakdown",
        json={"text1": "Hello world", "text2": "Hi there"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == expected


def test_predict_pair_breakdown_error_returns_500_and_request_id(mock_predictor):
    """If predictor throws, /predict/pair/breakdown should surface 500 and include request ID."""
    mock_predictor.predict_pair_breakdown.side_effect = ValueError("explode breakdown")

    with patch("backend.api.main.get_predictor", return_value=mock_predictor), \
         TestClient(app, raise_server_exceptions=False) as error_client:
        response = error_client.post(
            "/api/v1/predict/pair/breakdown",
            json={"text1": "Hello", "text2": "Hi"},
            headers={"X-Request-ID": "req-456"},
        )

    body = response.json()
    assert response.status_code == 500
    assert body["detail"] == "explode breakdown"
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


def test_predict_batch_empty_pairs_returns_422(client):
    """POST /api/v1/predict/batch with an empty list should be rejected with 422."""
    response = client.post("/api/v1/predict/batch", json={"pairs": []})
    assert response.status_code == 422


def test_predict_batch_malformed_pair_returns_422(client):
    """POST /api/v1/predict/batch with a non-pair element should be rejected with 422."""
    response = client.post("/api/v1/predict/batch", json={"pairs": ["just one string"]})
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
# Predict batch breakdown
# ---------------------------------------------------------------------------

def test_predict_batch_breakdown_happy_path(client, mock_predictor):
    """POST /api/v1/predict/batch/breakdown should return mocked breakdown results."""
    payload = [["Hello world", "Hi there"]]
    response = client.post("/api/v1/predict/batch/breakdown", json={"pairs": payload})

    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert "alignment" in result
    assert "divergent_in_1" in result
    assert "feature_scores" in result
    mock_predictor.predict_batch_breakdown.assert_called_once_with(payload)


def test_predict_batch_breakdown_too_many_pairs_returns_422(client):
    """POST /api/v1/predict/batch/breakdown with 11 pairs should be rejected with 422."""
    pairs = [["text one", "text two"]] * 11
    response = client.post("/api/v1/predict/batch/breakdown", json={"pairs": pairs})
    assert response.status_code == 422


def test_predict_batch_breakdown_empty_pairs_returns_422(client):
    """POST /api/v1/predict/batch/breakdown with an empty list should be rejected with 422."""
    response = client.post("/api/v1/predict/batch/breakdown", json={"pairs": []})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limit_handler_returns_429():
    """_rate_limit_handler should return 429 with the expected body."""
    import json
    from unittest.mock import MagicMock
    from starlette.requests import Request

    from backend.api.main import _rate_limit_handler

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/predict/pair",
        "query_string": b"",
        "headers": [],
    }
    fake_request = Request(scope)
    fake_exc = MagicMock()

    response = _rate_limit_handler(fake_request, fake_exc)

    assert response.status_code == 429
    assert json.loads(response.body) == {"detail": "Rate limit exceeded"}


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
