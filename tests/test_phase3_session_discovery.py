"""
Tests for the corrected Phase 3 session-discovery pipeline.
Covers normal weekends (conventional), sprint weekends (2023 Sprint Shootout),
and sprint weekends (2024+ Sprint Qualifying).
"""
import pytest
from unittest.mock import MagicMock, patch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_pipeline.fetch_predictive_data import (
    discover_sessions_for_event,
    SESSION_NAME_MAP,
    REQUIRED_SESSIONS,
    OPTIONAL_SESSIONS,
)


def make_event_row(sessions):
    """Helper: create a dict simulating a FastF1 schedule row with Session1–5."""
    row = {}
    for i, s in enumerate(sessions[:5], start=1):
        row[f"Session{i}"] = s
    for j in range(len(sessions) + 1, 6):
        row[f"Session{j}"] = ""
    return row


# ── Normal weekend ──────────────────────────────────────────────────────────

def test_conventional_weekend_discovers_race_and_qualifying():
    event = make_event_row(["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"])
    req, opt, all_s = discover_sessions_for_event(event)
    assert "Race" in req
    assert "Qualifying" in req
    assert opt == set()


def test_conventional_weekend_no_sprint_sessions():
    event = make_event_row(["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"])
    req, opt, all_s = discover_sessions_for_event(event)
    for sprint_name in OPTIONAL_SESSIONS:
        assert sprint_name not in opt


# ── 2023 Sprint Shootout weekends ────────────────────────────────────────────

def test_2023_sprint_shootout_discovered():
    # 2023 format: Practice 1, Sprint Shootout, Sprint, Qualifying, Race
    event = make_event_row(["Practice 1", "Sprint Shootout", "Sprint", "Qualifying", "Race"])
    req, opt, all_s = discover_sessions_for_event(event)
    assert "Race" in req
    assert "Qualifying" in req
    assert "Sprint Shootout" in opt
    assert "Sprint" in opt


def test_2023_sprint_shootout_maps_to_SS():
    assert SESSION_NAME_MAP["Sprint Shootout"] == "SS"


def test_2023_sprint_shootout_not_classified_as_SQ():
    # Sprint Shootout must NOT be mapped to SQ
    assert SESSION_NAME_MAP.get("Sprint Shootout") != "SQ"


def test_2023_sprint_weekend_required_sessions_still_present():
    event = make_event_row(["Practice 1", "Sprint Shootout", "Sprint", "Qualifying", "Race"])
    req, opt, _ = discover_sessions_for_event(event)
    # Even without Sprint Shootout data, the race weekend is not excluded
    # because Sprint Shootout is optional
    assert "Race" in req
    assert "Qualifying" in req
    assert "Sprint Shootout" in opt  # present in schedule, so it is attempted


# ── 2024+ Sprint Qualifying weekends ─────────────────────────────────────────

def test_2024_sprint_qualifying_discovered():
    # 2024 format: Practice 1, Sprint Qualifying, Sprint, Qualifying, Race
    event = make_event_row(["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"])
    req, opt, all_s = discover_sessions_for_event(event)
    assert "Race" in req
    assert "Qualifying" in req
    assert "Sprint Qualifying" in opt
    assert "Sprint" in opt


def test_2024_sprint_qualifying_maps_to_SQ():
    assert SESSION_NAME_MAP["Sprint Qualifying"] == "SQ"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_event_row_returns_empty_sets():
    event = make_event_row([])
    req, opt, all_s = discover_sessions_for_event(event)
    assert req == set()
    assert opt == set()


def test_event_with_only_race_returns_required_set():
    event = make_event_row(["Race"])
    req, opt, _ = discover_sessions_for_event(event)
    assert "Race" in req
    assert "Qualifying" not in req


def test_unknown_session_names_not_classified():
    event = make_event_row(["Unknown Session", "Race", "Qualifying"])
    req, opt, all_s = discover_sessions_for_event(event)
    assert "Race" in req
    assert "Qualifying" in req
    assert "Unknown Session" not in req
    assert "Unknown Session" not in opt


def test_required_sessions_constant():
    """Required sessions must always include Race and Qualifying."""
    assert "Race" in REQUIRED_SESSIONS
    assert "Qualifying" in REQUIRED_SESSIONS


def test_optional_sessions_constant():
    """Optional sessions must include all sprint format variants."""
    assert "Sprint" in OPTIONAL_SESSIONS
    assert "Sprint Shootout" in OPTIONAL_SESSIONS
    assert "Sprint Qualifying" in OPTIONAL_SESSIONS


def test_sprint_shootout_is_optional_not_required():
    """Sprint Shootout must never be classified as required."""
    assert "Sprint Shootout" not in REQUIRED_SESSIONS
    assert "Sprint Shootout" in OPTIONAL_SESSIONS


def test_sprint_qualifying_is_optional_not_required():
    """Sprint Qualifying must never be classified as required."""
    assert "Sprint Qualifying" not in REQUIRED_SESSIONS
    assert "Sprint Qualifying" in OPTIONAL_SESSIONS


# ── is_manifest_valid rejects SUPERSEDED_INVALID ─────────────────────────────

def test_manifest_valid_rejects_superseded(tmp_path):
    from data_pipeline.fetch_predictive_data import is_manifest_valid
    p = tmp_path / "completion_manifest.json"
    p.write_text('{"status": "SUPERSEDED_INVALID"}')
    assert is_manifest_valid(str(p)) is False


def test_manifest_valid_accepts_valid(tmp_path):
    from data_pipeline.fetch_predictive_data import is_manifest_valid
    import json
    p = tmp_path / "completion_manifest.json"
    manifest = {
        "session": "Q",
        "row_counts": {"results": 20},
        "source_hashes": {"results": "abc123"}
    }
    p.write_text(json.dumps(manifest))
    assert is_manifest_valid(str(p)) is True
