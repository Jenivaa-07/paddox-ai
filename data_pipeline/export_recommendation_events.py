import os
import json
import hmac
import hashlib
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load env
load_dotenv(os.path.join(os.path.dirname(__file__), '../../paddox-backend/.env'))
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

MONGO_URI = os.getenv('MONGO_URI')
HMAC_SECRET = os.getenv('RECOMMENDATION_PSEUDONYMIZATION_KEY')

if not HMAC_SECRET:
    raise ValueError("RECOMMENDATION_PSEUDONYMIZATION_KEY is not set in .env")

def hash_id(user_id):
    if not user_id: return None
    # HMAC-SHA256
    return hmac.new(HMAC_SECRET.encode(), str(user_id).encode(), hashlib.sha256).hexdigest()

def export_data():
    print("Connecting to MongoDB...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("Connected to MongoDB successfully.")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        client = None

    real_interactions = []
    real_users_meta = []
    
    if client:
        db = None
        for d in client.list_database_names():
            if 'paddox' in d.lower():
                db = client[d]
                break
        if db is None:
            db = client.get_default_database(default='test')
            
        print(f"Using database: {db.name}")
        
        # Extract Users
        user_coll = db['users']
        if user_coll is not None:
            for u in user_coll.find({}):
                if u.get('role') == 'admin': continue
                if u.get('email', '').endswith('@paddox.com'): continue
                
                uid = hash_id(u['_id'])
                prefs = u.get('preferences', {})
                real_users_meta.append({
                    'user_id': uid,
                    'favourite_team': prefs.get('favouriteTeam'),
                    'favourite_driver': prefs.get('favouriteDriver'),
                    'fan_tier': u.get('fanTier'),
                    'ai_credits': u.get('aiCredits')
                })
        
        # Extract Orders
        order_coll = db['orders']
        if order_coll is not None:
            for o in order_coll.find({}):
                if o.get('status') in ['Cancelled', 'Refunded', 'Failed']: continue
                uid = hash_id(o.get('user'))
                if not uid: continue
                for item in o.get('orderItems', []):
                    real_interactions.append({
                        'user_id': uid,
                        'item_id': str(item.get('product', '')),
                        'item_type': 'product',
                        'event_type': 'purchase',
                        'timestamp': o.get('createdAt', o.get('updatedAt', datetime.now())).isoformat()
                    })
                    
        # Extract Wishlist
        wishlist_coll = db['wishlists']
        if wishlist_coll is not None:
            for w in wishlist_coll.find({}):
                uid = hash_id(w.get('user'))
                if not uid: continue
                for item in w.get('products', []):
                    real_interactions.append({
                        'user_id': uid,
                        'item_id': str(item),
                        'item_type': 'product',
                        'event_type': 'wishlist',
                        'timestamp': datetime.now().isoformat()
                    })
                    
        # Extract Reviews
        review_coll = db['reviews']
        if review_coll is not None:
            for r in review_coll.find({}):
                uid = hash_id(r.get('user'))
                if not uid: continue
                real_interactions.append({
                    'user_id': uid,
                    'item_id': str(r.get('product', '')),
                    'item_type': 'product',
                    'event_type': 'review',
                    'timestamp': r.get('createdAt', datetime.now()).isoformat()
                })
                
        # Extract FanPoints
        fp_coll = db['fanpoints']
        if fp_coll is not None:
            for fp in fp_coll.find({}):
                action = fp.get('action')
                if action in ['download', 'review']:
                    uid = hash_id(fp.get('user'))
                    if not uid: continue
                    real_interactions.append({
                        'user_id': uid,
                        'item_id': str(fp.get('meta', {}).get('itemId', 'unknown')),
                        'item_type': 'wallpaper' if action == 'download' else 'unknown',
                        'event_type': 'wallpaper_download' if action == 'download' else action,
                        'timestamp': fp.get('createdAt', datetime.now()).isoformat()
                    })

    os.makedirs('data/raw/recommendations', exist_ok=True)
    os.makedirs('data/manifests', exist_ok=True)

    # Process REAL dataset
    df_real = pd.DataFrame(real_interactions)
    df_real_users = pd.DataFrame(real_users_meta).drop_duplicates(subset=['user_id'])
    
    real_manifest = {
        "dataset_type": "real_observed_interactions",
        "synthetic_mode": False,
        "eligible_for_pilot_analysis": True,
        "eligible_for_final_deep_model_evaluation": False,
        "source_collections": ["users", "orders", "wishlists", "reviews", "fanpoints"],
        "extraction_timestamp": datetime.now().isoformat(),
        "privacy_transformations": "HMAC-SHA256 for user IDs. Names, emails, passwords, addresses excluded."
    }
    
    if not df_real.empty and not df_real_users.empty:
        df_real.to_csv('data/raw/recommendations/real_interactions.csv', index=False)
        df_real_users.to_csv('data/raw/recommendations/real_users.csv', index=False)
        
        user_counts = df_real['user_id'].value_counts()
        item_counts = df_real['item_id'].value_counts()
        
        total_users = len(df_real_users)
        total_items = len(item_counts)
        unique_pairs = len(df_real[['user_id', 'item_id']].drop_duplicates())
        density = unique_pairs / (total_users * total_items) if total_users * total_items > 0 else 0
        sparsity = 1.0 - density
        
        real_manifest.update({
            "genuine_event_count": len(df_real),
            "pseudonymous_user_count": total_users,
            "item_count": total_items,
            "unique_user_item_pairs": unique_pairs,
            "event_counts_by_type": df_real['event_type'].value_counts().to_dict(),
            "date_range": {
                "start": df_real['timestamp'].min(),
                "end": df_real['timestamp'].max()
            },
            "users_with_fewer_than_2_interactions": int((user_counts < 2).sum()) + (total_users - len(user_counts)),
            "users_with_fewer_than_5_interactions": int((user_counts < 5).sum()) + (total_users - len(user_counts)),
            "users_with_fewer_than_10_interactions": int((user_counts < 10).sum()) + (total_users - len(user_counts)),
            "items_with_zero_interaction": 0, # Cannot know total possible items unless extracted, so just counting observed
            "items_with_one_interaction": int((item_counts == 1).sum()),
            "matrix_density": round(density, 6),
            "sparsity": round(sparsity, 6),
            "sha256_hashes": {
                "real_interactions.csv": hashlib.sha256(df_real.to_csv(index=False).encode()).hexdigest(),
                "real_users.csv": hashlib.sha256(df_real_users.to_csv(index=False).encode()).hexdigest()
            }
        })
    else:
        real_manifest["genuine_event_count"] = 0
        real_manifest["sparsity"] = 1.0

    with open('data/manifests/real_interactions_manifest.json', 'w') as f:
        json.dump(real_manifest, f, indent=4)
        
    print(f"Real data stats: {real_manifest.get('genuine_event_count', 0)} events. Sparsity: {real_manifest.get('sparsity', 1.0)}")

    # Process SYNTHETIC dataset for engineering tests
    import random
    synth_interactions = []
    synth_users = []
    for i in range(100):
        uid = hash_id(f"synth_user_{i%20}")
        synth_users.append({
            'user_id': uid,
            'favourite_team': random.choice(['Red Bull', 'Ferrari', 'Mercedes', 'McLaren']),
            'favourite_driver': random.choice(['Max Verstappen', 'Charles Leclerc', 'Lewis Hamilton']),
            'fan_tier': 'Regular',
            'ai_credits': 50
        })
        for _ in range(random.randint(1, 5)):
            item_type = random.choice(['product', 'content', 'highlight'])
            synth_interactions.append({
                'user_id': uid,
                'item_id': f"synth_{item_type}_{random.randint(1,50)}",
                'item_type': item_type,
                'event_type': random.choice(['product_view', 'content_view', 'purchase', 'wishlist', 'highlight_view']),
                'timestamp': datetime.now().isoformat()
            })
            
    df_synth = pd.DataFrame(synth_interactions)
    df_synth_users = pd.DataFrame(synth_users).drop_duplicates(subset=['user_id'])
    
    df_synth.to_csv('data/raw/recommendations/synthetic_interactions.csv', index=False)
    df_synth_users.to_csv('data/raw/recommendations/synthetic_users.csv', index=False)
    
    synth_manifest = {
        "dataset_type": "synthetic_test_fixture",
        "synthetic_mode": True,
        "eligible_for_paper": False,
        "eligible_for_production": False,
        "sha256_hashes": {
            "synthetic_interactions.csv": hashlib.sha256(df_synth.to_csv(index=False).encode()).hexdigest(),
            "synthetic_users.csv": hashlib.sha256(df_synth_users.to_csv(index=False).encode()).hexdigest()
        }
    }
    
    with open('data/manifests/synthetic_recommendation_smoke_manifest.json', 'w') as f:
        json.dump(synth_manifest, f, indent=4)

if __name__ == "__main__":
    export_data()
