import pytest
from fastapi.testclient import TestClient
from main import app, models

client = TestClient(app)

def test_sentiment_no_model():
    """Verify that if model is not loaded, it returns 503."""
    # Temporarily remove model if it exists
    original = models.pop("sentiment", None)
    
    response = client.post("/analyze-sentiment", json={"text": "This is a test"})
    assert response.status_code == 503
    assert response.json() == {"status": "model_not_ready"}
    
    if original is not None:
        models["sentiment"] = original

def test_sentiment_classification():
    """Verify classification output format by forcing the load of a smoke test model."""
    import glob
    import os
    from transformers import pipeline
    
    # Load the latest smoke model manually for testing
    artifacts_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sentiment")
    runs = glob.glob(os.path.join(artifacts_path, "run_*"))
    if not runs:
        pytest.skip("No smoke model trained yet, skipping integration test.")
    
    valid_runs = [r for r in runs if os.path.exists(os.path.join(r, "config.json"))]
    if not valid_runs:
        pytest.skip("No valid smoke model with config.json trained yet, skipping integration test.")
    
    latest_run = max(valid_runs, key=os.path.getmtime)
    
    # Force into models dict
    models["sentiment"] = pipeline("text-classification", model=latest_run, tokenizer=latest_run, device=-1)
        
    response = client.post("/analyze-sentiment", json={"text": "Max Verstappen is an incredible driver."})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["sentiment"] in ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    assert "score" in data
    
    # Clean up so we don't leak into other tests
    models.pop("sentiment", None)
    
def test_sentiment_invalid_input():
    """Verify invalid input is handled properly (Pydantic validation)."""
    response = client.post("/analyze-sentiment", json={"wrong_key": "test"})
    assert response.status_code == 422 # Unprocessable Entity
