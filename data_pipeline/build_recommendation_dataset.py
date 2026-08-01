import os
import json
import yaml
import pandas as pd
import numpy as np

def load_weights(path='config/recommendation_event_weights.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_dataset():
    print("Building recommendation dataset...")
    
    # Load raw data
    try:
        interactions = pd.read_csv('data/raw/recommendations/synthetic_interactions.csv')
        users = pd.read_csv('data/raw/recommendations/synthetic_users.csv')
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return
        
    if interactions.empty:
        print("No interactions found.")
        return
        
    # Apply weights
    weights = load_weights()
    interactions['implicit_score'] = interactions['event_type'].map(weights).fillna(1.0)
    
    # Sort chronologically to prevent future leakage
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    interactions = interactions.sort_values('timestamp').reset_index(drop=True)
    
    # Aggregate interactions per user-item pair (sum implicit scores)
    # But wait, to keep chronological splits, we should split FIRST, then aggregate inside the trainer
    # Actually, standard practice for sequential evaluation is to split chronologically by user or globally.
    # The prompt says: "chronological interaction splitting: - training: earlier interactions - validation: later interactions - test: latest untouched interactions"
    
    # Global chronological split (80/10/10)
    n = len(interactions)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_df = interactions.iloc[:train_end].copy()
    val_df = interactions.iloc[train_end:val_end].copy()
    test_df = interactions.iloc[val_end:].copy()
    
    # Create User and Item mappings based ON TRAINING SET to simulate real cold starts
    user_vocab = {u: idx for idx, u in enumerate(train_df['user_id'].unique())}
    item_vocab = {i: idx for idx, i in enumerate(train_df['item_id'].unique())}
    
    # Map IDs
    for df in [train_df, val_df, test_df]:
        df['user_idx'] = df['user_id'].map(user_vocab).fillna(-1).astype(int)
        df['item_idx'] = df['item_id'].map(item_vocab).fillna(-1).astype(int)
        
    # Save mappings
    os.makedirs('data/processed/recommendations/synthetic_smoke', exist_ok=True)
    
    with open('data/processed/recommendations/synthetic_smoke/user_map.json', 'w') as f:
        json.dump(user_vocab, f, indent=4)
    with open('data/processed/recommendations/synthetic_smoke/item_map.json', 'w') as f:
        json.dump(item_vocab, f, indent=4)
        
    # Save splits
    train_df.to_csv('data/processed/recommendations/synthetic_smoke/train.csv', index=False)
    val_df.to_csv('data/processed/recommendations/synthetic_smoke/val.csv', index=False)
    test_df.to_csv('data/processed/recommendations/synthetic_smoke/test.csv', index=False)
    
    # Save user features for cold start
    users.to_csv('data/processed/recommendations/user_features.csv', index=False)
    
    print(f"Dataset built. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    build_dataset()
