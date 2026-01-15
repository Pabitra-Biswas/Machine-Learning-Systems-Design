import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_hello_world():
    """Basic API health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()