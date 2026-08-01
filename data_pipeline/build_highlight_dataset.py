import os
import json
import pandas as pd
from datetime import datetime, timedelta
import random
import hashlib

def build_highlight_dataset():
    print("Building Highlight dataset (Synthetic for tests)...")
    
    highlights = []
    for i in range(1, 51):
        rights = random.choice(["approved", "pending", "expired", "unknown"])
        source = random.choice(["paddox_internal", "licensed_partner", "youtube_rip", "social_media_rip"])
        # ensure correlation: paddox_internal is approved, youtube_rip is unknown
        if source == "paddox_internal": rights = "approved"
        elif source == "youtube_rip": rights = "unknown"
        
        highlights.append({
            "highlight_id": f"hl_{i}",
            "race_id": f"2023_{random.randint(1, 20)}",
            "title": f"Amazing overtake on lap {random.randint(10, 50)}",
            "description": "A brilliant move down the inside.",
            "driver_ids": [random.randint(1, 20)],
            "constructor_ids": [random.randint(1, 10)],
            "event_type": "overtake",
            "lap": random.randint(10, 50),
            "timestamp": (datetime.now() - timedelta(days=random.randint(1, 100))).isoformat(),
            "source": source,
            "rights_status": rights,
            "expiry_date": (datetime.now() + timedelta(days=random.randint(-10, 30))).isoformat() if rights == "approved" else None
        })
        
    # Inject duplicates for testing
    duplicate_1 = highlights[0].copy()
    duplicate_2 = highlights[1].copy()
    highlights.append(duplicate_1)
    highlights.append(duplicate_2)
    
    df = pd.DataFrame(highlights)
    os.makedirs('data/processed/highlights', exist_ok=True)
    df.to_csv('data/processed/highlights/candidates.csv', index=False)
    
    manifest = {
        "source_collections": ["synthetic_mock_highlights"],
        "extraction_timestamp": datetime.now().isoformat(),
        "total_candidates": len(df),
        "synthetic_mode": True,
        "eligible_for_paper": False,
        "eligible_for_production": False
    }
    
    os.makedirs('data/manifests', exist_ok=True)
    with open('data/manifests/highlight_dataset_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Exported {len(df)} synthetic highlight candidates.")

if __name__ == "__main__":
    build_highlight_dataset()
