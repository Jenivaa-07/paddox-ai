import os
import pandas as pd
import json

def test_recommendation_dataset_splits():
    train_df = pd.read_csv('data/processed/recommendations/synthetic_smoke/train.csv')
    val_df = pd.read_csv('data/processed/recommendations/synthetic_smoke/val.csv')
    test_df = pd.read_csv('data/processed/recommendations/synthetic_smoke/test.csv')
    
    # Assert temporal split (train max time <= val min time <= test min time)
    assert pd.to_datetime(train_df['timestamp']).max() <= pd.to_datetime(val_df['timestamp']).min()
    assert pd.to_datetime(val_df['timestamp']).max() <= pd.to_datetime(test_df['timestamp']).min()
    
def test_mock_highlight_dataset():
    assert os.path.exists('data/processed/highlights/candidates.csv')
    df = pd.read_csv('data/processed/highlights/candidates.csv')
    assert 'rights_status' in df.columns
    
def test_no_synthetic_leakage():
    # if real data exists, assert no synth_user_ or synth_product_ appears
    import os
    if os.path.exists('data/raw/recommendations/real_interactions.csv'):
        real_df = pd.read_csv('data/raw/recommendations/real_interactions.csv')
        assert not real_df['user_id'].astype(str).str.contains('synth').any()
        assert not real_df['item_id'].astype(str).str.contains('synth').any()
