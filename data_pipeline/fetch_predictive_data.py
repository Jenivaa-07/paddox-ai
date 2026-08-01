import os
import sys
import json
import time
import hashlib
import fastf1
import pandas as pd
from datetime import datetime
import psutil
from fastf1.exceptions import RateLimitExceededError

fastf1.Cache.enable_cache("data/cache")

LOCK_FILE = "data/manifests/predictive_fetch.lock"
PROGRESS_FILE = "data/manifests/full_fetch_progress.json"
COMPLETE_FILE = "data/manifests/full_fetch_complete.json"
LOG_DIR = "reports/predictive/fetch_logs"
os.makedirs("data/manifests", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# SESSION DISCOVERY
# FastF1 session naming differs by season and format:
#   2023 sprint weekends: "Sprint Shootout" (SQ equivalent)
#   2024+ sprint weekends: "Sprint Qualifying" (SQ)
#   All seasons: "Race" (R), "Qualifying" (Q), "Sprint" (S)
#
# Strategy: inspect the FastF1 event schedule Session1-Session5
# columns and request only sessions that genuinely appear.
# Never request a session identifier that is not in the schedule.
# ──────────────────────────────────────────────────────────────

# Map from FastF1 schedule session name → our internal identifier (for manifest)
SESSION_NAME_MAP = {
    "Race":              "R",
    "Qualifying":        "Q",
    "Sprint":            "S",
    "Sprint Shootout":   "SS",   # 2023 naming
    "Sprint Qualifying": "SQ",   # 2024+ naming
}

# Sessions required for LSTM race predictor and RF fantasy predictor
REQUIRED_SESSIONS = {"Race", "Qualifying"}
# Optional sessions (sprint data; enriches model but absence doesn't exclude a race weekend)
OPTIONAL_SESSIONS = {"Sprint", "Sprint Shootout", "Sprint Qualifying"}


def discover_sessions_for_event(event_row):
    """
    Inspect Session1-Session5 columns in the FastF1 schedule row.
    Return (required_sessions, optional_sessions) as sets of FastF1 session names
    that genuinely appear in this event's schedule.
    """
    session_columns = [f"Session{i}" for i in range(1, 6)]
    schedule_sessions = set()
    for col in session_columns:
        val = event_row.get(col, "")
        if isinstance(val, str) and val.strip():
            schedule_sessions.add(val.strip())

    found_required = schedule_sessions & REQUIRED_SESSIONS
    found_optional = schedule_sessions & OPTIONAL_SESSIONS
    return found_required, found_optional, schedule_sessions


def get_hash(df):
    if df is None or df.empty:
        return "empty"
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def is_manifest_valid(manifest_path):
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        # Reject the invalid completion manifest
        if manifest.get("status") in ("SUPERSEDED_INVALID",):
            return False
        if 'results' not in manifest.get('row_counts', {}): return False
        if manifest['row_counts']['results'] <= 0: return False
        for file_name, file_hash in manifest.get('source_hashes', {}).items():
            if file_hash == "empty": return False
        sess = manifest.get('session', '')
        if sess in ('R', 'S', 'SS', 'SQ'):
            if 'laps' not in manifest.get('row_counts', {}): return False
            if manifest['row_counts']['laps'] <= 0: return False
        return True
    except Exception:
        return False


class FetchLogger:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"{LOG_DIR}/{self.timestamp}.log"
        self.logs = []
        self.status = {
            "starting_target": None,
            "completed_sessions": 0,
            "skipped_validated_sessions": 0,
            "invalid_sessions_queued": 0,
            "rate_limit_occurrence": False,
            "next_incomplete_session": None,
            "total_completed_races": 0,
            "remaining_races": 0,
            "exit_status": "RUNNING"
        }

    def log(self, msg):
        print(msg)
        self.logs.append(f"{datetime.now().isoformat()} - {msg}")

    def save(self):
        with open(self.log_path, 'w') as f:
            for line in self.logs:
                f.write(line + "\n")
            f.write("\n--- SUMMARY ---\n")
            json.dump(self.status, f, indent=4)


def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                lock_data = json.load(f)
            pid = lock_data.get("pid")
            if pid and psutil.pid_exists(pid):
                return False, lock_data
        except Exception:
            pass

    lock_data = {
        "pid": os.getpid(),
        "start_timestamp": datetime.now().isoformat(),
        "host": "windows",
        "command": " ".join(sys.argv),
        "current_target": None
    }
    with open(LOCK_FILE, 'w') as f:
        json.dump(lock_data, f, indent=4)
    return True, lock_data


def update_lock(lock_data, target):
    lock_data["current_target"] = target
    with open(LOCK_FILE, 'w') as f:
        json.dump(lock_data, f, indent=4)


def release_lock():
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass


def fetch_session_with_retries(year, rnd, session_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            session = fastf1.get_session(year, rnd, session_name)
            session.load(telemetry=False, weather=True)
            return session, None
        except RateLimitExceededError as e:
            raise e
        except Exception as e:
            if attempt == max_retries - 1:
                return None, str(e)
            time.sleep(2 ** attempt)
    return None, "Unknown error"


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "requested_seasons": [2022, 2023, 2024, 2025, 2026],
        "completed_events": [],
        "failed_events": [],
        "rate_limited_events": [],
        "remaining_events": [],
        "permanently_unavailable_events": [],
        "last_successful_timestamp": None,
        "next_resume_target": None
    }


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=4)


def process_event(year, event, schedule_sessions, lock_data, fetch_logger):
    """
    Fetch sessions for one race weekend.

    schedule_sessions: set of FastF1 session names actually in this event's schedule.
    Returns (event_success, sessions_fetched, sessions_unavailable)
    """
    rnd = event['RoundNumber']
    event_name = event['EventName']
    event_id = f"{year}_{rnd}"

    required_fetched = set()
    optional_fetched = set()
    sessions_unavailable = []

    for schedule_name in schedule_sessions:
        internal_id = SESSION_NAME_MAP.get(schedule_name, schedule_name)
        target_name = f"{year} Round {rnd} ({event_name}) - {schedule_name} [{internal_id}]"

        if lock_data:
            update_lock(lock_data, target_name)

        session_dir = f"data/raw/predictive/{year}/{rnd}/{internal_id}"
        manifest_path = f"{session_dir}/completion_manifest.json"

        if os.path.exists(manifest_path):
            if is_manifest_valid(manifest_path):
                fetch_logger.status["skipped_validated_sessions"] += 1
                if schedule_name in REQUIRED_SESSIONS:
                    required_fetched.add(schedule_name)
                else:
                    optional_fetched.add(schedule_name)
                continue
            else:
                fetch_logger.log(f"Invalid manifest for {target_name}. Refetching.")
                fetch_logger.status["invalid_sessions_queued"] += 1

        fetch_logger.log(f"Fetching {target_name}")
        if fetch_logger.status["starting_target"] is None:
            fetch_logger.status["starting_target"] = target_name

        os.makedirs(session_dir, exist_ok=True)
        temp_dir = f"{session_dir}/temp"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            session, error = fetch_session_with_retries(year, rnd, schedule_name)
        except RateLimitExceededError:
            fetch_logger.log(f"RateLimitExceededError hit at {target_name}")
            fetch_logger.status["rate_limit_occurrence"] = True
            fetch_logger.status["next_incomplete_session"] = target_name
            raise RateLimitExceededError()

        if error:
            fetch_logger.log(f"Error fetching {target_name}: {error[:200]}")
            sessions_unavailable.append({
                "event": event_id,
                "schedule_name": schedule_name,
                "internal_id": internal_id,
                "reason": "fetch_error",
                "detail": error[:200]
            })
            continue

        # Extract data
        files_saved = {}
        row_counts = {}
        missing_values = {}
        hashes = {}
        driver_counts = {}

        def save_df(name, df):
            if df is not None and not df.empty:
                tmp_path = f"{temp_dir}/{name}.csv"
                df.to_csv(tmp_path, index=False)
                row_counts[name] = len(df)
                missing_values[name] = int(df.isna().sum().sum())
                hashes[name] = get_hash(df)
                if 'DriverNumber' in df.columns:
                    driver_counts[name] = int(df['DriverNumber'].nunique())
                return True
            return False

        def get_data_safe(sess, attr):
            try:
                return getattr(sess, attr, None)
            except Exception:
                return None

        save_df("laps", get_data_safe(session, 'laps'))
        save_df("results", get_data_safe(session, 'results'))
        save_df("weather", get_data_safe(session, 'weather_data'))

        for name in list(row_counts.keys()):
            tmp_path = f"{temp_dir}/{name}.csv"
            final_path = f"{session_dir}/{name}.csv"
            if os.path.exists(tmp_path):
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(tmp_path, final_path)

        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

        manifest = {
            "season": year,
            "round": rnd,
            "event_name": event_name,
            "session_schedule_name": schedule_name,
            "session": internal_id,
            "expected_files": list(row_counts.keys()),
            "row_counts": row_counts,
            "driver_counts": driver_counts,
            "missing_values": missing_values,
            "source_hashes": hashes,
            "validation_status": "VALID",
            "completed_timestamp": datetime.now().isoformat()
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

        fetch_logger.status["completed_sessions"] += 1

        if schedule_name in REQUIRED_SESSIONS:
            required_fetched.add(schedule_name)
        else:
            optional_fetched.add(schedule_name)

    # Event succeeds if all required sessions are present
    required_in_schedule = schedule_sessions & REQUIRED_SESSIONS
    event_success = required_fetched >= required_in_schedule

    return event_success, required_fetched, optional_fetched, sessions_unavailable


def fetch_all_data():
    fetch_logger = FetchLogger()

    if os.path.exists(COMPLETE_FILE):
        try:
            with open(COMPLETE_FILE, 'r') as f:
                manifest = json.load(f)
            if manifest.get("status") == "SUPERSEDED_INVALID":
                fetch_logger.log("Invalid completion manifest detected. Proceeding with fetch.")
            else:
                fetch_logger.log("Fetch already completed (valid manifest found).")
                fetch_logger.status["exit_status"] = "ALREADY_COMPLETE"
                fetch_logger.save()
                return
        except Exception:
            pass

    acquired, lock_data = acquire_lock()
    if not acquired:
        fetch_logger.log(f"Another fetcher is running (PID: {lock_data.get('pid')}). Exiting.")
        fetch_logger.status["exit_status"] = "LOCKED"
        fetch_logger.save()
        return

    # Counters for the new manifest
    total_race_weekends_requested = 0
    total_race_weekends_successfully_fetched = 0
    total_sessions_required_fetched = 0
    total_sessions_optional_fetched = 0
    all_unavailable_sessions = []

    try:
        progress = load_progress()

        all_past_events = []
        for year in progress["requested_seasons"]:
            try:
                schedule = fastf1.get_event_schedule(year, include_testing=False)
                past = schedule[schedule['EventDate'] < pd.Timestamp.now()]
                for _, event in past.iterrows():
                    all_past_events.append((year, event))
                    total_race_weekends_requested += 1
            except RateLimitExceededError:
                fetch_logger.status["rate_limit_occurrence"] = True
                raise
            except Exception as e:
                fetch_logger.log(f"Failed to fetch schedule for {year}: {e}")

        progress["remaining_events"] = []
        for year, event in all_past_events:
            event_id = f"{year}_{event['RoundNumber']}"
            if event_id not in progress["completed_events"]:
                progress["remaining_events"].append(event_id)

        fetch_logger.status["remaining_races"] = len(progress["remaining_events"])
        fetch_logger.status["total_completed_races"] = len(progress["completed_events"])

        for year, event in all_past_events:
            event_id = f"{year}_{event['RoundNumber']}"

            # Use schedule-driven session discovery
            required_sessions, optional_sessions, all_schedule_sessions = discover_sessions_for_event(event)

            if not required_sessions:
                fetch_logger.log(f"Skipping {event_id}: no required sessions found in schedule. Available: {all_schedule_sessions}")
                continue

            sessions_to_fetch = required_sessions | optional_sessions

            success, req_fetched, opt_fetched, unavailable = process_event(
                year, event, sessions_to_fetch, lock_data, fetch_logger
            )

            all_unavailable_sessions.extend(unavailable)
            total_sessions_required_fetched += len(req_fetched)
            total_sessions_optional_fetched += len(opt_fetched)

            if success and event_id not in progress["completed_events"]:
                progress["completed_events"].append(event_id)
                if event_id in progress.get("remaining_events", []):
                    progress["remaining_events"].remove(event_id)
                progress["last_successful_timestamp"] = datetime.now().isoformat()
                fetch_logger.status["total_completed_races"] += 1
                fetch_logger.status["remaining_races"] = len(progress["remaining_events"])
                total_race_weekends_successfully_fetched += 1
                save_progress(progress)

        if len(progress["remaining_events"]) == 0:
            fetch_logger.log("All race weekends fetched successfully!")
            fetch_logger.status["exit_status"] = "COMPLETE"

            completion_manifest = {
                "schema_version": "2.0",
                "status": "fetch_exhausted_all_available",
                "completed_timestamp": datetime.now().isoformat(),
                "race_weekends": {
                    "requested": total_race_weekends_requested,
                    "successfully_fetched": total_race_weekends_successfully_fetched,
                },
                "sessions": {
                    "required_successfully_fetched": total_sessions_required_fetched,
                    "optional_successfully_fetched": total_sessions_optional_fetched,
                    "unavailable": len(all_unavailable_sessions),
                },
                "unavailable_session_details": all_unavailable_sessions,
                "model_training_gate": "OPEN",
                "integrity_validation_required_before_training": True,
                "note": (
                    "Session discovery used actual FastF1 event schedule (Session1-Session5) "
                    "rather than hard-coded identifiers. 2023 Sprint Shootout sessions are "
                    "requested using the FastF1 schedule name 'Sprint Shootout'. "
                    "A race weekend is marked complete only when all required sessions "
                    "(Race, Qualifying) are present."
                )
            }
            with open(COMPLETE_FILE, 'w') as f:
                json.dump(completion_manifest, f, indent=4)
        else:
            fetch_logger.status["exit_status"] = "PARTIAL_COMPLETE"

    except RateLimitExceededError:
        fetch_logger.log("Stopped cleanly due to RateLimitExceededError.")
        fetch_logger.status["exit_status"] = "RATE_LIMITED"
        progress = load_progress()
        if fetch_logger.status["next_incomplete_session"]:
            progress["next_resume_target"] = fetch_logger.status["next_incomplete_session"]
        save_progress(progress)
    except Exception as e:
        fetch_logger.log(f"Unexpected error: {e}")
        fetch_logger.status["exit_status"] = "ERROR"
    finally:
        fetch_logger.save()
        release_lock()


if __name__ == "__main__":
    print("Starting scheduled robust incremental FastF1 fetcher (v2 – schedule-driven session discovery)...")
    fetch_all_data()
