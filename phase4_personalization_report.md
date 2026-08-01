# Phase 4: Hybrid Personalization & Highlight Ranking Report

This document confirms the implementation of PADDOX Phase 4 (Recommendation System and Highlight Ranking).

Phase 4 Engineering Status: COMPLETE
Phase 4 Real Interaction Dataset Status: UNAVAILABLE – ZERO EVENTS
Phase 4 Deep Model Status: SMOKE TEST ONLY
Phase 4 Preference/Catalogue Fallback Status: LOCALLY VALIDATED
Phase 4 Highlight Rights Filter Status: LOCALLY VALIDATED
Phase 4 Node Gateway Status: LOCALLY VALIDATED
Phase 4 IEEE Result Eligibility: PENDING

## 1. Directory Tree
```text
data/raw/recommendations/
  real_interactions.csv
  real_users.csv
  synthetic_interactions.csv
  synthetic_users.csv
data/processed/recommendations/
  real/
  synthetic_smoke/
    train.csv
    val.csv
    test.csv
    user_map.json
    item_map.json
data/processed/highlights/
  candidates.csv
models/
  hybrid_recommender_model.py
  hybrid_recommender_trainer.py
  highlight_ranker.py
services/
  recommendation_service.py
  highlight_service.py
config/
  recommendation_event_weights.yaml
  recommendation_model_config.yaml
  highlight_rights_policy.yaml
```

## 2. Real Recommendation Data Extraction

An automated extraction process from the MongoDB backend (`export_recommendation_events.py`) was executed.

### Reconciliation of Previous Report
The previous report mentioned "327 events across 20 configurations". That extraction observed zero real interactions and reverted to a synthetic generation routine to produce the 327 events. The actual genuine behavioral extraction is precisely zero.

| Record Type | Count |
| :--- | :--- |
| Records initially discovered | 0 |
| Genuine behavioural events | 0 |
| Genuine preference profiles | 1 |
| Genuine interacting users | 0 |
| Synthetic fixture events | 327 |
| Rejected development/test records | 0 |
| **Final genuine interaction count** | **0** |

### Data Statistics
- **genuine event count:** 0
- **pseudonymous user count (interacting):** 0
- **genuine preference profiles:** 1
- **item count:** 0
- **unique user-item pairs:** 0
- **event counts by type:** {}
- **date range:** null
- **interacting users with fewer than 2 interactions:** 0
- **interacting users with fewer than 5 interactions:** 0
- **interacting users with fewer than 10 interactions:** 0
- **preference profiles without behavioural events:** 1
- **items with zero interaction:** 0
- **items with one interaction:** 0
- **matrix density:** null
- **sparsity:** null
- **matrix_status:** unavailable_no_real_interactions

Density and sparsity are not mathematically defined because no genuine user-item interaction matrix could be formed.

### Privacy Processing
User IDs were pseudonymized using an HMAC-SHA256 signature with the key securely loaded from `RECOMMENDATION_PSEUDONYMIZATION_KEY` in `.env`. Names, emails, passwords, addresses, phone numbers, payment details, and authentication tokens were strictly excluded. 

## 3. Hybrid Recommender System
- **Model Architecture:** PyTorch Two-Tower (User Tower and Item Tower) using BPR (Bayesian Personalized Ranking) loss.
- **Cold-Start Cohorts:** Users without sufficient interaction history are mapped to index `0` and routed through the `catalog_featured_fallback` path via `RecommendationService`.

### Synthetic Code-Path Smoke-Test Metrics – Not Research or IEEE Results.
Due to zero genuine behavioural-interaction events, no real interaction matrix could be constructed. Engineering validation was therefore performed using a separately stored synthetic test fixture (`synthetic_recommendation_smoke_manifest.json`) strictly separated from genuine behavioural-interaction and preference data.
- Precision@5: 0.12
- Recall@10: 0.25
- NDCG@10: 0.17

### Measured Benchmark (Latency)
- **warm-up count:** 50
- **measured-call count:** 500
- **p50:** 2.3501 ms
- **p95:** 3.9446 ms
- **mean:** 2.5532 ms
- **standard deviation:** 0.6332 ms
- **raw sample path:** `reports/recommendations/latency_samples.csv`
- **environment details:** OS: Windows, Machine: AMD64, CPU Count: 16

## 4. Personalized Highlight Ranker
A rules-based rights-filtering and ranking service was established. When no authorized inventory exists, `/rank-highlights` securely returns:
```json
{
  "highlights": [],
  "status": "authorized_inventory_unavailable",
  "grounded": false
}
```
It does not return synthetic or unauthorized candidates to real API requests.

### 100% rights-policy accuracy on the labelled synthetic smoke-test fixture
- **authorized accepted:** 9
- **prohibited rejected:** 13
- **expired rejected:** 4
- **missing-rights rejected:** 13
- **duplicate suppressed:** 1
- **false acceptance count:** 0
- **false rejection count:** 0

## 5. Endpoints & Verification
FastAPI endpoints `POST /recommend` and `POST /rank-highlights` were integrated and validated in the local development environment. The Node gateway `aiClient.service.js` has been updated with `getRecommendations` and `rankHighlights`. 

### Exact Verification Evidence

**Phase 4 pytest command:**
`.\.venv\Scripts\pytest.exe tests/test_phase4_data.py tests/test_phase4_recommender.py tests/test_phase4_highlights.py tests/test_phase4_endpoints.py tests/test_phase4_privacy.py`
- passed counts: 9
- failed counts: 0
- skipped counts: 0
- warning counts: 1 
  - *Warning Class:* `StarletteDeprecationWarning`
  - *Source:* `fastapi.testclient` ("Using `httpx` with `starlette.testclient` is deprecated")

**Node gateway command:**
`node test_aiClient.js` (Run in paddox-backend)
- passed assertions: 4
- failed assertions: 0
- exit code: 0

**privacy-test result:** PASSED (Identical IDs match, different differ, identifiers/secrets absent)
**rights-policy-test result:** PASSED (authorized accepted, prohibited/expired/missing rejected, duplicates suppressed)

## 6. Phase 4 Conclusion

Phase 4 engineering is complete. The Two-Tower recommendation pipeline, catalogue-featured fallback, privacy-preserving extraction, highlight-rights filter, FastAPI endpoints and Node.js gateway were implemented and locally validated. Because PADDOX currently contains zero genuine behavioural-interaction events, the deep collaborative model remains a synthetic code-path smoke test and is not eligible for production deployment or IEEE research metrics. The catalogue and explicit-preference fallback remains the active safe strategy until sufficient genuine interactions are collected.

**Note:** No fabricated user interactions, relevance labels, engagement improvements, or fake IEEE metrics are claimed in this report. No synthetic recommendation fixture is mixed with genuine behavioural-interaction or preference data.
