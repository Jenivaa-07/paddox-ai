import pandas as pd
import json
import hashlib
from datetime import datetime
import os

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def validate_and_manifest():
    print("Validating datasets and building manifest...")
    
    # Paths
    splits = ['train', 'val', 'test']
    
    manifest = {
        "source": "FastF1 API (Ergast/Jolpica via FOM data)",
        "source_url": "https://github.com/theOehrly/Fast-F1",
        "collection_timestamp": datetime.now().isoformat(),
        "seasons_and_races": "Chronological split from 2022-2024 (subset limits apply)",
        "preprocessing_steps": ["Dropped null laps", "Shifted rolling windows to prevent leakage", "Padding up to 10 timesteps"],
        "random_seed": 42,
        "files": {}
    }
    
    total_leakages = 0
    
    for split in splits:
        race_seq = pd.read_csv(f"data/processed/predictive/{split}_race_seq.csv")
        fantasy = pd.read_csv(f"data/processed/predictive/{split}_fantasy_final.csv")
        
        # Leakage check: The target should never be used as input.
        # Ensure 'final_position' is not in fantasy features.
        if 'final_position' in fantasy.columns or 'Position' in fantasy.columns:
            print(f"LEAKAGE DETECTED in fantasy {split}!")
            total_leakages += 1
            
        manifest["files"][split] = {
            "race_seq_rows": len(race_seq),
            "fantasy_rows": len(fantasy),
            "race_seq_hash": hash_file(f"data/processed/predictive/{split}_race_seq.csv"),
            "fantasy_hash": hash_file(f"data/processed/predictive/{split}_fantasy_final.csv")
        }
    
    manifest["leakage_checks"] = "PASSED" if total_leakages == 0 else "FAILED"
    
    with open("data/manifests/predictive_dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Manifest saved to data/manifests/predictive_dataset_manifest.json")
    if total_leakages > 0:
        raise ValueError("Data leakage validation failed!")

if __name__ == "__main__":
    validate_and_manifest()
