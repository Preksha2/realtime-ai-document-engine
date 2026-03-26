"""
Unit tests for the FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from src.api.server import app


client = TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "index_loaded" in data
        assert "total_chunks" in data


class TestQueryEndpoint:

    def test_query_without_index_returns_503(self):
        response = client.post("/query", json={
            "question": "What is the revenue?",
            "top_k": 5
        })
        assert response.status_code == 503

    def test_query_empty_question_returns_422(self):
        response = client.post("/query", json={
            "question": "",
            "top_k": 5
        })
        assert response.status_code == 422

    def test_query_invalid_top_k_returns_422(self):
        response = client.post("/query", json={
            "question": "Valid question here",
            "top_k": 0
        })
        assert response.status_code == 422

    def test_query_exceeds_max_top_k_returns_422(self):
        response = client.post("/query", json={
            "question": "Valid question here",
            "top_k": 25
        })
        assert response.status_code == 422


class TestIndexEndpoint:

    def test_index_invalid_directory_returns_500(self):
        response = client.post("/index", json={
            "source_dir": "/nonexistent/directory/path"
        })
        assert response.status_code == 500
