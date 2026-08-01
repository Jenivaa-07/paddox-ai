from models.highlight_ranker import HighlightRanker

def test_highlight_rights_filtering():
    ranker = HighlightRanker()
    
    # 1. Test approved
    is_valid, _ = ranker.check_rights({"rights_status": "approved", "source": "paddox_internal"})
    assert is_valid == True
    
    # 2. Test unknown/pending
    is_valid, reason = ranker.check_rights({"rights_status": "unknown"})
    assert is_valid == False
    assert reason == "Not approved"
    
    # 3. Test expired
    is_valid, reason = ranker.check_rights({"rights_status": "approved", "source": "paddox_internal", "expiry_date": "2000-01-01T00:00:00Z"})
    assert is_valid == False
    assert reason == "Rights expired"
    
    # 4. Test prohibited source
    is_valid, reason = ranker.check_rights({"rights_status": "approved", "source": "youtube_rip"})
    assert is_valid == False
    assert reason == "Prohibited source"

def test_highlight_duplicate_suppression():
    ranker = HighlightRanker()
    candidates = [
        {"highlight_id": "hl_1", "rights_status": "approved", "source": "paddox_internal"},
        {"highlight_id": "hl_2", "rights_status": "approved", "source": "paddox_internal"},
        {"highlight_id": "hl_1", "rights_status": "approved", "source": "paddox_internal"} # Duplicate
    ]
    ranked, filtered, duplicates = ranker.rank("user_123", "2023_1", candidates)
    
    assert len(ranked) == 2
    assert duplicates == 1
    assert filtered == 1 # 1 duplicate filtered
