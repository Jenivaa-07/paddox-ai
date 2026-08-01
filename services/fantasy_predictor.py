import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, List

class FantasyPredictorService:
    def __init__(self):
        self.pipeline = None
        self.model_version = None
        self.load_model()
        
    def load_model(self):
        try:
            with open("artifacts/predictive/rf_fantasy/current_model.json", "r") as f:
                self.model_version = json.load(f)["run_id"]
                
            model_path = f"artifacts/predictive/rf_fantasy/{self.model_version}/model.joblib"
            if os.path.exists(model_path):
                self.pipeline = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load RF model: {e}")
            self.pipeline = None
            
    def is_ready(self):
        return self.pipeline is not None

    def predict_batch(self, req_data: Dict[str, Any]):
        if not self.is_ready():
            raise ValueError("model_not_ready")
            
        drivers = req_data.get('drivers', [])
        if not drivers:
            raise ValueError("missing drivers list")
            
        # Build DataFrame
        rows = []
        for d in drivers:
            features = d.get('features', {})
            rows.append({
                'driver_id': d.get('driver_id'),
                'qualifying_position': float(d.get('qualifying_position', 20.0)),
                'rolling_avg_finish': float(features.get('rolling_avg_finish', 20.0)),
                'constructor_id': d.get('constructor_id', 'Unknown')
            })
            
        df = pd.DataFrame(rows)
        
        # Predict
        preds = self.pipeline.predict(df[['qualifying_position', 'rolling_avg_finish', 'constructor_id']])
        df['predicted_fantasy_points'] = preds
        
        # Rank calculation requires batch.
        is_single = len(drivers) == 1
        
        df['predicted_rank'] = df['predicted_fantasy_points'].rank(method='min', ascending=False)
        
        results = []
        for i, row in df.iterrows():
            results.append({
                "driver_id": row['driver_id'],
                "predicted_fantasy_points": round(float(row['predicted_fantasy_points']), 2),
                "predicted_rank": None if is_single else int(row['predicted_rank'])
            })
            
        return {
            "predictions": results,
            "field_size": len(drivers),
            "scoring_version": "PADDOX_FANTASY_V1",
            "model_version": self.model_version
        }
