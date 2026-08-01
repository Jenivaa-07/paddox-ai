import pytest
from fastapi.testclient import TestClient
from main import app, HAS_TORCH, HAS_TRANSFORMERS, HAS_FASTF1

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "paddox-ai"}

def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "dependencies" in data

def test_unimplemented_predict_race():
    response = client.post("/predict-race")
    # In Phase 3, this is now implemented and expects a JSON body, so it returns 422 Unprocessable Entity
    assert response.status_code == 422



if __name__ == "__main__":
    pytest.main(["-q", __file__])
