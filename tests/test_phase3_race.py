import pytest
import torch
import json
from models.lstm_race_model import LSTMRacePredictor
from services.race_predictor import RacePredictorService

def test_lstm_model_shapes():
    model = LSTMRacePredictor(input_dim=3, hidden_dim=64, num_layers=2)
    # Batch=2, Seq=10, Feat=3
    x = torch.randn(2, 10, 3)
    reg, cls = model(x)
    assert reg.shape == (2, 1)
    assert cls.shape == (2, 1)

def test_race_predictor_service_loads_and_predicts():
    svc = RacePredictorService()
    # It should be ready since we trained it
    assert svc.is_ready()
    
    req = {
        "recent_laps": [90000, 89000, 88000],
        "features": {
            "grid_position": 1,
            "rolling_avg_finish": 1.5
        }
    }
    
    res = svc.predict(req)
    assert "expected_finish_position" in res
    assert "top10_probability" in res
    assert 1.0 <= res["expected_finish_position"] <= 22.0
    assert 0.0 <= res["top10_probability"] <= 1.0
