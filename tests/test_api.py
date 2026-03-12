"""API endpoint tests for SilverBullet.

These tests use a mocked SimilarityPredictor so no trained model file is
required.  All test coverage is against the HTTP contract (status codes,
response shapes, input validation, CORS headers).
"""

import pytest


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    """GET /api/v1/health should return 200 with status='ok'."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


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
