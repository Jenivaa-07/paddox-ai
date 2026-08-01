import json
import os
import torch
import time
from models.hybrid_recommender_model import TwoTowerRecommender

class RecommendationService:
    def __init__(self):
        self.model = None
        self.config = None
        self.user_map = {}
        self.item_map = {}
        self.is_ready = False
        self.load_latest_model()
        
    def load_latest_model(self):
        try:
            base_dir = "artifacts/recommendations"
            if not os.path.exists(base_dir): return
            runs = sorted(os.listdir(base_dir), reverse=True)
            if not runs: return
            
            latest_run = runs[0]
            run_dir = os.path.join(base_dir, latest_run)
            
            with open(os.path.join(run_dir, "model_config.json"), "r") as f:
                self.config = json.load(f)
                
            with open("data/processed/recommendations/synthetic_smoke/user_map.json", "r") as f:
                self.user_map = json.load(f)
                
            with open("data/processed/recommendations/synthetic_smoke/item_map.json", "r") as f:
                self.item_map = json.load(f)
                
            self.model = TwoTowerRecommender(len(self.user_map), len(self.item_map), embed_dim=self.config.get("embedding_dim", 64))
            self.model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt")))
            self.model.eval()
            self.is_ready = True
            self.model_version = latest_run
        except Exception as e:
            print(f"Error loading recommendation model: {e}")
            self.is_ready = False

    def get_recommendations(self, user_id, context, k, exclude_item_ids):
        start = time.time()
        
        if not self.is_ready:
            strategy = "catalog_featured_fallback"
            recs = [{"item_id": "fallback_item_1", "item_type": "product", "score": 0.5, "reason": "Featured in catalog"}]
            latency = (time.time() - start) * 1000
            return recs, strategy, "fallback", latency
            
        u_idx = self.user_map.get(user_id, 0) # 0 is padding/unknown (cold-start)
        strategy = "hybrid" if u_idx != 0 else "catalog_featured_fallback"
        
        # We will mock inference across a subset of items to simulate real deployment
        # because the full catalog mapping is not cleanly stored right now.
        # But for engineering tests this is sufficient.
        
        recs = []
        for i_id, i_idx in list(self.item_map.items())[:k+5]:
            if i_id in exclude_item_ids: continue
            
            with torch.no_grad():
                score = self.model(
                    torch.tensor([u_idx]), torch.tensor([0]), torch.tensor([0]),
                    torch.tensor([i_idx]), torch.tensor([0]), torch.tensor([0])
                ).item()
                
            recs.append({
                "item_id": i_id,
                "item_type": "product",
                "score": round(score, 4),
                "reason": "Matches your preference profile"
            })
            
        recs = sorted(recs, key=lambda x: x['score'], reverse=True)[:k]
        latency = (time.time() - start) * 1000
        
        return recs, strategy, self.model_version, latency

recommendation_service = RecommendationService()
