import pandas as pd
import yaml
import hashlib
import json
from pathlib import Path

def load_scoring_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_fantasy_targets(df_path, out_path, scoring_config_path='config/fantasy_scoring_v1.yaml'):
    print(f"Building fantasy targets for {df_path}")
    df = pd.read_csv(df_path)
    config = load_scoring_config(scoring_config_path)
    
    race_points = config['race_points']
    modifiers = config['modifiers']
    
    points = []
    
    # Calculate target (PADDOX fantasy points)
    for _, row in df.iterrows():
        pos = row.get('final_position', 20.0)
        grid = row.get('qualifying_position', 20.0)
        
        # Base points for finishing position
        p = race_points.get(int(pos), 0) if pd.notnull(pos) else 0
        
        # Positions gained modifier
        gained = max(0, grid - pos) if pd.notnull(grid) and pd.notnull(pos) else 0
        p += gained * modifiers['position_gained']
        
        # DNF modifier
        status = str(row.get('status', ''))
        if 'Finished' not in status and '+1 Lap' not in status and '+2 Laps' not in status:
            if 'DSQ' in status:
                p += modifiers['dsq']
            else:
                p += modifiers['dnf']
                
        points.append(p)
        
    df['fantasy_points_target'] = points
    
    df['rolling_avg_finish'] = df['rolling_avg_finish'].fillna(20.0)
    
    # Pre-race features only + target + metadata
    # Removed Position, Time, status
    feature_cols = ['season', 'round', 'DriverNumber', 'constructor_id', 'qualifying_position', 'rolling_avg_finish', 'field_size', 'fantasy_points_target']
    
    df_clean = df[feature_cols].copy()
    df_clean.to_csv(out_path, index=False)
    print(f"Saved fantasy dataset to {out_path}")

def write_fantasy_split_manifest(output_dir="data/processed/predictive"):
    """Persist exact fantasy split IDs, counts, and hashes after target generation."""
    root = Path(output_dir)
    manifest = {
        "source": "FastF1-derived public race/session data",
        "target_definition_file": "config/fantasy_scoring_v1.yaml",
        "split_unit": "race (season, round)",
        "splits": {}
    }
    race_sets = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}_fantasy_final.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated fantasy split: {path}")
        frame = pd.read_csv(path)
        race_ids = sorted(
            frame[["season", "round"]].drop_duplicates()
            .astype(int).itertuples(index=False, name=None)
        )
        race_sets[split] = set(race_ids)
        manifest["splits"][split] = {
            "rows": int(len(frame)),
            "unique_races": len(race_ids),
            "race_ids": [f"{season}-{round_no}" for season, round_no in race_ids],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    manifest["no_race_overlap"] = not (
        (race_sets["train"] & race_sets["val"])
        or (race_sets["train"] & race_sets["test"])
        or (race_sets["val"] & race_sets["test"])
    )
    if not manifest["no_race_overlap"]:
        raise RuntimeError("Fantasy race-level split leakage detected.")
    destination = root / "fantasy_split_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        build_fantasy_targets(
            f"data/processed/predictive/{split}_fantasy_raw.csv",
            f"data/processed/predictive/{split}_fantasy_final.csv"
        )
    write_fantasy_split_manifest()
