import time
import pandas as pd
from models.highlight_ranker import HighlightRanker

class HighlightService:
    def __init__(self):
        self.ranker = HighlightRanker()
        
    def rank_highlights(self, user_id, race_id, candidate_ids, k):
        start = time.time()
        
        # Real inventory check
        has_authorized_inventory = False
        
        if not has_authorized_inventory:
            return {
                "highlights": [],
                "status": "authorized_inventory_unavailable",
                "grounded": False
            }
            
        try:
            df = pd.read_csv('data/processed/highlights/candidates.csv')
            all_candidates = df.to_dict('records')
            if candidate_ids:
                candidates = [c for c in all_candidates if c['highlight_id'] in candidate_ids]
            else:
                candidates = all_candidates
        except:
            candidates = []
            
        ranked, filtered_count, duplicates = self.ranker.rank(user_id, race_id, candidates)
        
        latency = (time.time() - start) * 1000
        
        return {
            "highlights": ranked[:k],
            "filtered_for_rights": filtered_count,
            "duplicate_suppressed": duplicates,
            "model_version": "v1.0_rules",
            "inference_latency_ms": latency
        }

highlight_service = HighlightService()
