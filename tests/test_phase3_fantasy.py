import pytest
import yaml
from services.fantasy_predictor import FantasyPredictorService

def test_fantasy_scoring_config():
    with open("config/fantasy_scoring_v1.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert config["name"] == "PADDOX_FANTASY_V1"
    assert config["race_points"][1] == 25
    assert config["modifiers"]["dnf"] == -10

def test_fantasy_predictor_service():
    svc = FantasyPredictorService()
    assert svc.is_ready()
    
    req = {
        "drivers": [
            {
                "driver_id": "VER",
                "constructor_id": "Red Bull",
                "qualifying_position": 1,
                "features": {"rolling_avg_finish": 1.0}
            },
            {
                "driver_id": "HAM",
                "constructor_id": "Mercedes",
                "qualifying_position": 5,
                "features": {"rolling_avg_finish": 4.5}
            }
        ]
    }
    
    res = svc.predict_batch(req)
    assert res["field_size"] == 2
    assert "PADDOX_FANTASY_V1" in res["scoring_version"]
    assert len(res["predictions"]) == 2
    
    preds = res["predictions"]
    # Check that predicted rank is assigned correctly (lower number is better rank, so higher points = better rank)
    points_0 = preds[0]["predicted_fantasy_points"]
    points_1 = preds[1]["predicted_fantasy_points"]
    
    rank_0 = preds[0]["predicted_rank"]
    rank_1 = preds[1]["predicted_rank"]
    
    if points_0 > points_1:
        assert rank_0 < rank_1
    else:
        assert rank_1 < rank_0
