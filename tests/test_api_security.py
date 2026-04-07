"""Tests for API security: auth, rate limiting, CORS, input validation."""
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def set_api_key():
    """Ensure a known API key for tests."""
    with patch.dict(os.environ, {"RAG_API_KEY": "test-key-12345"}):
        # Re-import to pick up env var
        import importlib
        import src.api.security as sec
        importlib.reload(sec)
        yield


@pytest.fixture
def mock_retriever():
    """Mock retriever that returns fake chunks."""
    mock = MagicMock()
    mock.embedder.get_or_create_collection.return_value.count.return_value = 95
    chunk = MagicMock()
    chunk.text = "Test chunk content about aged care standards."
    chunk.chunk_id = "test-chunk-001"
    chunk.to_metadata.return_value = {"document_title": "Test Doc"}
    chunk.score = 0.9
    mock.retrieve.return_value = [chunk]
    return mock


@pytest.fixture
def mock_generator():
    """Mock generator that returns a fake response."""
    mock = MagicMock()
    resp = MagicMock()
    resp.question = "test question"
    resp.answer = "test answer"
    resp.model = "claude-sonnet-4-5-20250929"
    resp.input_tokens = 100
    resp.output_tokens = 50
    resp.cost_usd = 0.001
    mock.generate.return_value = resp
    mock.model = "claude-sonnet-4-5-20250929"
    return mock


@pytest.fixture
def client(mock_retriever, mock_generator):
    """Test client with mocked retriever/generator."""
    import src.api.app as api_app
    import src.api.security as sec
    import importlib

    # Reload security module to pick up test env vars
    importlib.reload(sec)
    importlib.reload(api_app)

    api_app.retriever = mock_retriever
    api_app.generator = mock_generator
    return TestClient(api_app.app, raise_server_exceptions=False)


# --- Health endpoint (no auth) ---

class TestHealthEndpoint:
    def test_health_no_auth_required(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["collection_size"] == 95


# --- API Key Auth ---

class TestAPIKeyAuth:
    def test_query_with_valid_api_key(self, client):
        resp = client.post(
            "/query",
            json={"question": "What are aged care standards?"},
            headers={"X-API-Key": "test-key-12345"},
        )
        assert resp.status_code == 200

    def test_query_with_invalid_api_key(self, client):
        resp = client.post(
            "/query",
            json={"question": "What are aged care standards?"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_query_without_auth(self, client):
        resp = client.post(
            "/query",
            json={"question": "What are aged care standards?"},
        )
        assert resp.status_code == 401


# --- JWT Auth ---

class TestJWTAuth:
    def test_get_token_with_valid_api_key(self, client):
        resp = client.post(
            "/auth/token",
            json={"api_key": "test-key-12345"},
            headers={"X-API-Key": "test-key-12345"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_get_token_without_api_key(self, client):
        resp = client.post(
            "/auth/token",
            json={"api_key": "test-key-12345"},
        )
        assert resp.status_code == 401

    def test_query_with_jwt(self, client):
        # Get token
        token_resp = client.post(
            "/auth/token",
            json={"api_key": "test-key-12345"},
            headers={"X-API-Key": "test-key-12345"},
        )
        token = token_resp.json()["access_token"]

        # Use token for query
        resp = client.post(
            "/query",
            json={"question": "What are aged care standards?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_query_with_expired_token(self, client):
        resp = client.post(
            "/query",
            json={"question": "What are aged care standards?"},
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert resp.status_code == 401


# --- Input Validation ---

class TestInputValidation:
    def test_question_too_short(self, client):
        resp = client.post(
            "/query",
            json={"question": "Hi"},
            headers={"X-API-Key": "test-key-12345"},
        )
        assert resp.status_code == 422

    def test_question_too_long(self, client):
        resp = client.post(
            "/query",
            json={"question": "x" * 1001},
            headers={"X-API-Key": "test-key-12345"},
        )
        assert resp.status_code == 422

    def test_top_k_out_of_range(self, client):
        resp = client.post(
            "/query",
            json={"question": "What are standards?", "top_k": 50},
            headers={"X-API-Key": "test-key-12345"},
        )
        assert resp.status_code == 422

    def test_missing_question(self, client):
        resp = client.post(
            "/query",
            json={},
            headers={"X-API-Key": "test-key-12345"},
        )
        assert resp.status_code == 422


# --- CORS ---

class TestCORS:
    def test_cors_allowed_origin(self, client):
        resp = client.options(
            "/query",
            headers={
                "Origin": "http://localhost:8504",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:8504"

    def test_cors_disallowed_origin(self, client):
        resp = client.options(
            "/query",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "http://evil.com"
