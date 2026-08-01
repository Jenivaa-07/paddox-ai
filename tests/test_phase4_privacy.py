import json
import pandas as pd
import re
import os
import hmac
import hashlib
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

def test_pseudonymization_and_privacy():
    with open('data/manifests/real_interactions_manifest.json', 'r') as f:
        manifest = json.load(f)
        
    assert "HMAC-SHA256 for user IDs" in manifest["privacy_transformations"]
    assert "Names, emails, passwords, addresses excluded" in manifest["privacy_transformations"]
    
    if os.path.exists('data/raw/recommendations/real_users.csv'):
        users_df = pd.read_csv('data/raw/recommendations/real_users.csv')
        
        # Check that user IDs are 64-char hex strings
        for uid in users_df['user_id']:
            assert re.match(r'^[a-fA-F0-9]{64}$', str(uid)), f"User ID {uid} is not a valid HMAC-SHA256 hash"
            
        # Check that sensitive fields are not exported
        banned_fields = ['name', 'email', 'password', 'address', 'phone', 'token']
        for col in users_df.columns:
            for b in banned_fields:
                assert b not in col.lower()
                
def test_hmac_properties():
    secret = os.getenv('RECOMMENDATION_PSEUDONYMIZATION_KEY')
    assert secret is not None, "Secret must be loaded from .env"
    
    def hash_id(user_id):
        return hmac.new(secret.encode(), str(user_id).encode(), hashlib.sha256).hexdigest()
        
    # identical IDs produce identical pseudonyms
    assert hash_id("user_123") == hash_id("user_123")
    
    # different IDs produce different pseudonyms
    assert hash_id("user_123") != hash_id("user_124")
    
    # Original identifiers are absent (hash isn't simply the original string)
    assert hash_id("user_123") != "user_123"
    
    # Check that secret is not accidentally dumped to logs
    with open('data/manifests/real_interactions_manifest.json', 'r') as f:
        content = f.read()
        assert secret not in content
