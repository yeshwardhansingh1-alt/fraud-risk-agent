import pytest
from fastapi.testclient import TestClient
from agent.api import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_missing_features(client):
    # Sending empty features should trigger 400 Bad Request
    response = client.post("/predict", headers={"X-API-Key": "default-dev-key"}, json={
        "features": {},
        "transaction_amount": 100.0,
        "transaction_id": "TEST-123"
    })
    assert response.status_code == 400
    assert "Missing" in response.json()["detail"]

def test_predict_endpoint_unauthorized(client):
    response = client.post("/predict", json={
        "features": {},
        "transaction_amount": 100.0,
        "transaction_id": "TEST-123"
    })
    assert response.status_code == 401
