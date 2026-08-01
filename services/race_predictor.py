import os
import json
import torch
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any
from models.lstm_race_model import LSTMRacePredictor

class RacePredictorService:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.load_model()
        
    def load_model(self):
        try:
            with open("artifacts/predictive/lstm_race/current_model.json", "r") as f:
                config = json.load(f)
                self.model_version = config["run_id"]
                self.threshold = config.get("threshold", 0.5)
                
            model_path = f"artifacts/predictive/lstm_race/{self.model_version}/model.pt"
            if os.path.exists(model_path):
                self.model = LSTMRacePredictor(input_dim=3, hidden_dim=64, num_layers=2)
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                self.model.eval()
                print(f"Loaded LSTM run_id {self.model_version} with threshold {self.threshold}")
        except Exception as e:
            print(f"Failed to load LSTM model: {e}")
            self.model = None
            
    def is_ready(self):
        return self.model is not None

    def predict(self, req_data: Dict[str, Any]):
        if not self.is_ready():
            raise ValueError("model_not_ready")
            
        if 'features' not in req_data:
            raise ValueError("Malformed input: 'features' is required")
            
        # Basic parsing & formatting
        recent_laps = req_data.get('recent_laps', [])
        # We need to pad to 10
        features = req_data.get('features', {})
        grid_pos = float(features.get('grid_position', 20.0))
        rolling_avg = float(features.get('rolling_avg_finish', 20.0))
        field_size = float(features.get('field_size', 20.0))
        
        from utils.scaling import scale_lap_time
        # Sequence build
        seq = []
        for lap_time in recent_laps[-10:]:
            seq.append([scale_lap_time(lap_time), grid_pos, rolling_avg])
            
        while len(seq) < 10:
            seq.insert(0, [0.0, grid_pos, rolling_avg])
            
        x_tensor = torch.tensor([seq], dtype=torch.float32)
        
        with torch.no_grad():
            pred_reg, pred_cls = self.model(x_tensor)
            
        # normalized_position = (position - 1) / (field_size - 1)
        normalized_pred = float(torch.clamp(pred_reg, 0.0, 1.0).item())
        predicted_position = 1.0 + normalized_pred * (field_size - 1.0)
        
        top10_prob = float(torch.sigmoid(pred_cls).item())
        is_top10 = top10_prob >= getattr(self, "threshold", 0.30)
        
        return {
            "expected_finish_position": round(predicted_position, 2),
            "top10_probability": round(top10_prob, 4),
            "is_top10": is_top10,
            "model_version": self.model_version
        }
