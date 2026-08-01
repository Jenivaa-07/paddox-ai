import os
from fetch_predictive_data import fetch_race_data
from build_race_sequences import build_features
from build_fantasy_features import build_fantasy_targets
from validate_predictive_data import validate_and_manifest

def clean_and_process():
    # In a real environment we would clean erroneous telemetries.
    # Since we are using FastF1 we skip complex cleaning.
    # We call the sequence builders directly.
    pass

def main():
    print("Starting PADDOX Predictive Data Pipeline...")
    
    # Check if raw files exist
    if not os.path.exists("data/raw/predictive/laps.csv"):
        print("Raw laps not found. Running fetch...")
        # Since fetch_race_data is slow, it should already be run.
        fetch_race_data([2022, 2023, 2024], max_rounds_per_year=4)
        
    print("Building sequences...")
    build_features('data/raw/predictive/laps.csv', 'data/raw/predictive/results.csv', 'data/processed/predictive')
    
    print("Building fantasy targets...")
    for split in ['train', 'val', 'test']:
        build_fantasy_targets(f"data/processed/predictive/{split}_fantasy.csv", f"data/processed/predictive/{split}_fantasy_final.csv")
        
    print("Validating datasets...")
    validate_and_manifest()
    
    print("Data Pipeline Complete!")

if __name__ == "__main__":
    main()
