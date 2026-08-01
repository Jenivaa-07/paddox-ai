from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_recommend_endpoint():
    response = client.post("/recommend", json={
        "user_id": "test_user",
        "context": {"page": "shop"},
        "k": 5
    })
    # Since model is trained on synthetic data, it should return 200
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "strategy" in data
    assert "inference_latency_ms" in data

def test_rank_highlights_endpoint():
    response = client.post("/rank-highlights", json={
        "user_id": "test_user",
        "race_id": "2023_1",
        "k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "authorized_inventory_unavailable"
    assert data["highlights"] == []
    assert data["grounded"] == False
