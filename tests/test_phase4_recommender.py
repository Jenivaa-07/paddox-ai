from services.recommendation_service import RecommendationService

def test_recommender_cold_start():
    svc = RecommendationService()
    recs, strategy, model_version, latency = svc.get_recommendations("unknown_cold_user", {}, 10, [])
    # even if model loaded, unknown user defaults to cold start or popularity fallback
    assert strategy in ["catalog_featured_fallback"]
    
def test_recommender_exclude_items():
    svc = RecommendationService()
    if svc.is_ready:
        recs, strat, ver, lat = svc.get_recommendations(list(svc.user_map.keys())[0], {}, 10, ["fallback_item_1"])
        for r in recs:
            assert r["item_id"] != "fallback_item_1"
