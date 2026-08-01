import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_race_endpoint():
    req = {
        "recent_laps": [90000, 89000],
        "features": {
            "grid_position": 1,
            "rolling_avg_finish": 1.5
        }
    }
    
    response = client.post("/predict-race", json=req)
    assert response.status_code == 200
    data = response.json()
    assert "inference_latency_ms" in data
    assert "request_id" in data
    assert "expected_finish_position" in data

def test_predict_fantasy_endpoint():
    req = {
        "drivers": [
            {
                "driver_id": "VER",
                "constructor_id": "Red Bull",
                "qualifying_position": 1,
                "features": {"rolling_avg_finish": 1.0}
            }
        ]
    }
    
    response = client.post("/predict-fantasy", json=req)
    assert response.status_code == 200
    data = response.json()
    assert "inference_latency_ms" in data
    assert "request_id" in data
    assert data["field_size"] == 1
    assert data["predictions"][0]["predicted_rank"] is None # single driver rule
