import os
import json
import pandas as pd
import pytest

def test_dataset_manifest_exists():
    manifest_path = "data/manifests/predictive_dataset_manifest.json"
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert manifest["leakage_checks"] == "PASSED"
    assert "train" in manifest["files"]
    assert "test" in manifest["files"]
    
def test_race_split_isolation():
    # Verify no race in train split exists in test split
    train_seq = pd.read_csv("data/processed/predictive/train_race_seq.csv")
    test_seq = pd.read_csv("data/processed/predictive/test_race_seq.csv")
    
    train_races = set(train_seq['season'].astype(str) + train_seq['round'].astype(str))
    test_races = set(test_seq['season'].astype(str) + test_seq['round'].astype(str))
    
    assert len(train_races.intersection(test_races)) == 0, "Leakage detected: Races overlap between splits!"

def test_feature_leakage_prevention():
    # Final position must not be in the fantasy input features
    train_fantasy = pd.read_csv("data/processed/predictive/train_fantasy_final.csv")
    assert 'Position' not in train_fantasy.columns
    assert 'final_position' not in train_fantasy.columns
    # Target must be present
    assert 'fantasy_points_target' in train_fantasy.columns
