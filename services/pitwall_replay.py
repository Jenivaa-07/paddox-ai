from __future__ import annotations

import bisect
import hmac
import math
import os
import tempfile
import threading
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

try:
    import fastf1
except ImportError:  # pragma: no cover
    fastf1 = None

router = APIRouter(prefix="/pitwall/replay", tags=["pitwall-replay"])

_CACHE_DIR = os.getenv("FASTF1_CACHE_DIR", os.path.join(tempfile.gettempdir(), "paddox-fastf1-cache"))
_cache_ready = False
_cache_lock = threading.Lock()
_build_lock = threading.Lock()
_replay_cache: dict[str, dict[str, Any]] = {}

_SESSION_MAP = {
    "R": "R", "RACE": "R",
    "Q": "Q", "QUALIFYING": "Q",
    "S": "S", "SPRINT": "S", "SPRINT RACE": "S",
    "SQ": "SQ", "SPRINT QUALIFYING": "SQ", "SPRINT SHOOTOUT": "SQ",
    "FP1": "FP1", "PRACTICE 1": "FP1",
    "FP2": "FP2", "PRACTICE 2": "FP2",
    "FP3": "FP3", "PRACTICE 3": "FP3",
}


def _verify_service_key(x_paddox_ai_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("AI_SERVICE_KEY", "").strip()
    if expected and (not x_paddox_ai_key or not hmac.compare_digest(x_paddox_ai_key, expected)):
        raise HTTPException(status_code=401, detail="invalid_service_key")


def _ensure_cache() -> None:
    global _cache_ready
    if _cache_ready or fastf1 is None:
        return
    with _cache_lock:
        if _cache_ready:
            return
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fastf1.Cache.enable_cache(_CACHE_DIR)
        _cache_ready = True


def _session_code(value: str) -> str:
    code = _SESSION_MAP.get(str(value or "R").strip().upper())
    if not code:
        raise ValueError(f"unsupported_session:{value}")
    return code


def _cache_key(year: int, round_num: int, session_code: str) -> str:
    return f"{year}:{round_num}:{session_code}"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else default


def _td_seconds(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value.total_seconds())
    except Exception:
        return _safe_float(value)


def _fmt_lap_time(value: Any) -> str:
    seconds = _td_seconds(value)
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:06.3f}" if minutes else f"{remainder:.3f}"


def _hex_color(value: Any) -> str:
    text = str(value or "").strip().replace("#", "")
    if len(text) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in text):
        return f"#{text}"
    return "#e8002d"


def _driver_metadata(session) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    drivers: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    results = getattr(session, "results", None)
    if results is None:
        return drivers, lookup

    for _, row in results.iterrows():
        code = str(row.get("Abbreviation", "") or "").strip().upper()
        number = str(row.get("DriverNumber", "") or "").strip()
        if not code:
            continue
        item = {
            "code": code,
            "driverNumber": number,
            "name": str(row.get("FullName", "") or code),
            "team": str(row.get("TeamName", "") or "Formula 1"),
            "teamColor": _hex_color(row.get("TeamColor")),
            "gridPosition": _safe_int(row.get("GridPosition")),
            "resultPosition": _safe_int(row.get("Position")),
        }
        drivers.append(item)
        lookup[code] = item
        if number:
            lookup[number] = item
    return drivers, lookup


def _driver_laps(session, code: str):
    laps = session.laps
    try:
        return laps.pick_drivers(code)
    except Exception:
        try:
            return laps.pick_driver(code)
        except Exception:
            return laps[laps["Driver"].astype(str).str.upper() == code]


def _build_track(session) -> dict[str, Any]:
    fastest = session.laps.pick_fastest()
    if fastest is None:
        raise ValueError("track_telemetry_unavailable")
    telemetry = fastest.get_telemetry()
    if telemetry is None or telemetry.empty or "X" not in telemetry.columns or "Y" not in telemetry.columns:
        raise ValueError("track_telemetry_unavailable")

    coords = telemetry[["X", "Y"]].dropna()
    if coords.empty:
        raise ValueError("track_telemetry_unavailable")

    step = max(1, len(coords) // 420)
    sampled = coords.iloc[::step]
    if len(sampled) > 1 and not sampled.iloc[0].equals(sampled.iloc[-1]):
        sampled = pd.concat([sampled, sampled.iloc[[0]]], ignore_index=True)

    x_min = float(coords["X"].min())
    x_max = float(coords["X"].max())
    y_min = float(coords["Y"].min())
    y_max = float(coords["Y"].max())
    x_span = max(1.0, x_max - x_min)
    y_span = max(1.0, y_max - y_min)

    points = [
        {"x": round(float(row.X), 2), "y": round(float(row.Y), 2)}
        for row in sampled.itertuples(index=False)
    ]
    return {
        "points": points,
        "bounds": {"xMin": x_min, "xMax": x_max, "yMin": y_min, "yMax": y_max, "xSpan": x_span, "ySpan": y_span},
    }


def _build_driver_series(session, code: str) -> dict[str, np.ndarray] | None:
    laps = _driver_laps(session, code)
    if laps is None or len(laps) == 0:
        return None
    try:
        telemetry = laps.get_telemetry()
    except Exception:
        return None
    if telemetry is None or telemetry.empty or "SessionTime" not in telemetry.columns:
        return None

    data = telemetry.copy()
    data["_t"] = data["SessionTime"].dt.total_seconds()
    data = data.dropna(subset=["_t"]).sort_values("_t")
    if data.empty:
        return None

    data["_bucket"] = np.floor(data["_t"].to_numpy(dtype=float) * 2.0).astype(np.int64)
    data = data.groupby("_bucket", sort=True).tail(1)

    def numeric(column: str, default: float = np.nan) -> np.ndarray:
        if column not in data.columns:
            return np.full(len(data), default, dtype=float)
        return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)

    return {
        "t": data["_t"].to_numpy(dtype=float),
        "x": numeric("X"),
        "y": numeric("Y"),
        "speed": numeric("Speed"),
        "throttle": numeric("Throttle"),
        "brake": numeric("Brake", 0.0),
        "gear": numeric("nGear"),
        "drs": numeric("DRS"),
        "rpm": numeric("RPM"),
    }


def _build_laps(session, code: str) -> list[dict[str, Any]]:
    laps = _driver_laps(session, code)
    output: list[dict[str, Any]] = []
    if laps is None:
        return output

    for _, lap in laps.iterrows():
        lap_number = _safe_int(lap.get("LapNumber"))
        if lap_number is None:
            continue
        start = _td_seconds(lap.get("LapStartTime"))
        end = _td_seconds(lap.get("Time"))
        output.append({
            "lap": lap_number,
            "start": start,
            "end": end,
            "position": _safe_int(lap.get("Position")),
            "lapTime": _fmt_lap_time(lap.get("LapTime")),
            "lapSec": _td_seconds(lap.get("LapTime")),
            "s1": _fmt_lap_time(lap.get("Sector1Time")),
            "s2": _fmt_lap_time(lap.get("Sector2Time")),
            "s3": _fmt_lap_time(lap.get("Sector3Time")),
            "compound": str(lap.get("Compound", "") or "").upper() if pd.notna(lap.get("Compound")) else None,
            "tyreLife": _safe_int(lap.get("TyreLife")),
        })
    return output


def _build_weather(session) -> list[dict[str, Any]]:
    weather = getattr(session, "weather_data", None)
    if weather is None or weather.empty:
        return []
    result: list[dict[str, Any]] = []
    for _, row in weather.iterrows():
        t = _td_seconds(row.get("Time"))
        if t is None:
            t = _td_seconds(row.get("SessionTime"))
        if t is None:
            continue
        result.append({
            "t": t,
            "airTemp": _safe_float(row.get("AirTemp")),
            "trackTemp": _safe_float(row.get("TrackTemp")),
            "humidity": _safe_float(row.get("Humidity")),
            "pressure": _safe_float(row.get("Pressure")),
            "windSpeed": _safe_float(row.get("WindSpeed")),
            "rainfall": bool(row.get("Rainfall")) if pd.notna(row.get("Rainfall")) else False,
        })
    return result


def _build_messages(session) -> list[dict[str, Any]]:
    messages = getattr(session, "race_control_messages", None)
    if messages is None or messages.empty:
        return []
    result: list[dict[str, Any]] = []
    for _, row in messages.iterrows():
        t = _td_seconds(row.get("Time"))
        if t is None:
            continue
        text = str(row.get("Message", "") or row.get("Category", "") or "Race control update")
        result.append({
            "t": t,
            "message": text,
            "flag": str(row.get("Flag", "") or row.get("Category", "") or ""),
            "lap": _safe_int(row.get("Lap")),
        })
    return result


def _build_replay(year: int, round_num: int, session_name: str) -> dict[str, Any]:
    if fastf1 is None:
        raise RuntimeError("fastf1_not_installed")
    _ensure_cache()
    session_code = _session_code(session_name)
    key = _cache_key(year, round_num, session_code)
    if key in _replay_cache:
        return _replay_cache[key]

    with _build_lock:
        if key in _replay_cache:
            return _replay_cache[key]

        session = fastf1.get_session(year, round_num, session_code)
        session.load(telemetry=True, laps=True, weather=True, messages=True)
        if session.laps is None or len(session.laps) == 0:
            raise ValueError("fastf1_session_has_no_laps")

        drivers, driver_lookup = _driver_metadata(session)
        series: dict[str, dict[str, np.ndarray]] = {}
        laps: dict[str, list[dict[str, Any]]] = {}
        all_start_times: list[float] = []
        all_end_times: list[float] = []

        for driver in drivers:
            code = driver["code"]
            driver_series = _build_driver_series(session, code)
            if driver_series is not None and len(driver_series["t"]):
                series[code] = driver_series
                all_start_times.append(float(driver_series["t"][0]))
                all_end_times.append(float(driver_series["t"][-1]))
            laps[code] = _build_laps(session, code)

        if not series:
            raise ValueError("fastf1_telemetry_unavailable")

        start_t = min(all_start_times)
        end_t = max(all_end_times)
        duration = max(1.0, end_t - start_t)

        for driver_series in series.values():
            driver_series["t"] = driver_series["t"] - start_t
        for driver_laps in laps.values():
            for lap in driver_laps:
                if lap["start"] is not None:
                    lap["start"] -= start_t
                if lap["end"] is not None:
                    lap["end"] -= start_t

        weather = _build_weather(session)
        for item in weather:
            item["t"] -= start_t
        messages = _build_messages(session)
        for item in messages:
            item["t"] -= start_t

        track = _build_track(session)
        total_laps = max((lap["lap"] for values in laps.values() for lap in values), default=0)
        event = getattr(session, "event", {})
        event_name = str(event.get("EventName", "") if hasattr(event, "get") else "")
        location = str(event.get("Location", "") if hasattr(event, "get") else "")

        replay = {
            "sessionKey": key,
            "year": year,
            "round": round_num,
            "session": session_name,
            "sessionCode": session_code,
            "eventName": event_name,
            "circuit": location,
            "durationSeconds": round(duration, 3),
            "totalLaps": total_laps,
            "drivers": drivers,
            "driverLookup": driver_lookup,
            "series": series,
            "laps": laps,
            "weather": sorted(weather, key=lambda x: x["t"]),
            "messages": sorted(messages, key=lambda x: x["t"]),
            "track": track,
        }
        _replay_cache[key] = replay
        return replay


def _sample_index(times: np.ndarray, t: float) -> int | None:
    if times is None or len(times) == 0:
        return None
    index = int(np.searchsorted(times, t, side="right") - 1)
    if index < 0:
        return 0
    return min(index, len(times) - 1)


def _latest_completed_lap(driver_laps: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    completed = [lap for lap in driver_laps if lap["end"] is not None and lap["end"] <= t]
    return completed[-1] if completed else None


def _current_lap(driver_laps: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    current = None
    for lap in driver_laps:
        start = lap.get("start")
        end = lap.get("end")
        if start is not None and start <= t and (end is None or t <= end):
            return lap
        if start is not None and start <= t:
            current = lap
    return current


def _best_lap(driver_laps: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    candidates = [lap for lap in driver_laps if lap["end"] is not None and lap["end"] <= t and lap.get("lapSec")]
    return min(candidates, key=lambda lap: lap["lapSec"]) if candidates else None


def _nearest_channel(items: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    if not items:
        return None
    times = [item["t"] for item in items]
    idx = bisect.bisect_right(times, t) - 1
    if idx < 0:
        return items[0]
    return items[min(idx, len(items) - 1)]


def _position_to_screen(x: float | None, y: float | None, bounds: dict[str, float]) -> tuple[float | None, float | None]:
    if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
        return None, None
    sx = 70.0 + ((x - bounds["xMin"]) / bounds["xSpan"]) * 860.0
    sy = 70.0 + ((y - bounds["yMin"]) / bounds["ySpan"]) * 480.0
    return round(sx, 2), round(sy, 2)


def _frame(replay: dict[str, Any], at_seconds: float) -> dict[str, Any]:
    duration = float(replay["durationSeconds"])
    t = max(0.0, min(float(at_seconds), duration))
    telemetry: list[dict[str, Any]] = []
    locations: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    lap_states: dict[str, dict[str, Any]] = {}

    bounds = replay["track"]["bounds"]
    driver_meta = {d["code"]: d for d in replay["drivers"]}

    for code, driver_series in replay["series"].items():
        idx = _sample_index(driver_series["t"], t)
        if idx is None:
            continue
        meta = driver_meta.get(code, {"code": code, "teamColor": "#e8002d"})
        x = _safe_float(driver_series["x"][idx])
        y = _safe_float(driver_series["y"][idx])
        sx, sy = _position_to_screen(x, y, bounds)
        telemetry.append({
            "code": code,
            "speed": _safe_float(driver_series["speed"][idx]),
            "throttle": _safe_float(driver_series["throttle"][idx]),
            "brake": _safe_float(driver_series["brake"][idx], 0.0),
            "gear": _safe_int(driver_series["gear"][idx]),
            "drs": _safe_int(driver_series["drs"][idx]),
            "rpm": _safe_int(driver_series["rpm"][idx]),
        })
        if sx is not None and sy is not None:
            locations.append({"code": code, "sx": sx, "sy": sy, "teamColor": meta.get("teamColor", "#e8002d")})

        driver_laps = replay["laps"].get(code, [])
        current = _current_lap(driver_laps, t)
        completed = _latest_completed_lap(driver_laps, t)
        best = _best_lap(driver_laps, t)
        lap_states[code] = {"current": current, "completed": completed, "best": best}

    completed_states = [(code, state["completed"]) for code, state in lap_states.items() if state["completed"] is not None]
    leader_code = None
    leader_lap = 0
    leader_end = None
    if completed_states:
        leader_code, leader_state = max(completed_states, key=lambda pair: (pair[1]["lap"], -(pair[1]["end"] or float("inf"))))
        leader_lap = leader_state["lap"]
        same_lap = [pair for pair in completed_states if pair[1]["lap"] == leader_lap and pair[1]["end"] is not None]
        if same_lap:
            leader_code, leader_state = min(same_lap, key=lambda pair: pair[1]["end"])
            leader_end = leader_state["end"]

    for driver in replay["drivers"]:
        code = driver["code"]
        state = lap_states.get(code, {})
        current = state.get("current")
        completed = state.get("completed")
        best = state.get("best")
        fallback_position = driver.get("gridPosition") or driver.get("resultPosition")
        position = (completed or current or {}).get("position") or fallback_position

        gap = "LEADER" if code == leader_code else "—"
        if completed and leader_lap:
            if completed["lap"] < leader_lap:
                diff = leader_lap - completed["lap"]
                gap = f"+{diff} LAP" if diff == 1 else f"+{diff} LAPS"
            elif leader_end is not None and completed.get("end") is not None:
                delta = max(0.0, float(completed["end"]) - float(leader_end))
                gap = f"+{delta:.3f}"

        timing_rows.append({
            **driver,
            "position": position,
            "gap": gap,
            "bestLap": best.get("lapTime") if best else "—",
            "bestSec": best.get("lapSec") if best else None,
            "lastLap": completed.get("lapTime") if completed else "—",
            "lastSec": completed.get("lapSec") if completed else None,
            "s1": completed.get("s1") if completed else "—",
            "s2": completed.get("s2") if completed else "—",
            "s3": completed.get("s3") if completed else "—",
            "tyre": (current or completed or {}).get("compound") or "—",
            "tyreAge": (current or completed or {}).get("tyreLife"),
            "laps": (current or completed or {}).get("lap") or 0,
            "noTiming": completed is None,
        })

    timing_rows.sort(key=lambda row: (row.get("position") is None, row.get("position") or 999, row.get("code", "")))
    for index, row in enumerate(timing_rows, start=1):
        if row.get("position") is None:
            row["position"] = index

    current_lap = max((row.get("laps", 0) for row in timing_rows), default=0)
    weather = _nearest_channel(replay["weather"], t)
    recent_messages = [item for item in replay["messages"] if item["t"] <= t][-8:]

    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    return {
        "sessionKey": replay["sessionKey"],
        "at": round(t, 3),
        "progress": round(t / duration, 6) if duration else 0.0,
        "elapsedText": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        "lap": current_lap,
        "totalLaps": replay["totalLaps"],
        "locations": locations,
        "telemetry": telemetry,
        "timingRows": timing_rows,
        "weather": weather,
        "raceControl": recent_messages,
    }


def _manifest(replay: dict[str, Any]) -> dict[str, Any]:
    initial = _frame(replay, 0.0)
    return {
        "sessionKey": replay["sessionKey"],
        "source": "FastF1",
        "year": replay["year"],
        "round": replay["round"],
        "session": replay["session"],
        "sessionCode": replay["sessionCode"],
        "eventName": replay["eventName"],
        "circuit": replay["circuit"],
        "durationSeconds": replay["durationSeconds"],
        "totalLaps": replay["totalLaps"],
        "drivers": replay["drivers"],
        "trackPoints": replay["track"]["points"],
        "initialFrame": initial,
    }


@router.get("/manifest", dependencies=[Depends(_verify_service_key)])
async def replay_manifest(
    year: int = Query(..., ge=2018, le=2030),
    round_num: int = Query(..., alias="round", ge=1, le=30),
    session: str = Query("Race", min_length=1, max_length=32),
):
    try:
        replay = await run_in_threadpool(_build_replay, year, round_num, session)
        return _manifest(replay)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"fastf1_replay_unavailable:{exc}") from exc


@router.get("/frame", dependencies=[Depends(_verify_service_key)])
async def replay_frame(
    year: int = Query(..., ge=2018, le=2030),
    round_num: int = Query(..., alias="round", ge=1, le=30),
    session: str = Query("Race", min_length=1, max_length=32),
    at: float = Query(0.0, ge=0.0),
):
    try:
        replay = await run_in_threadpool(_build_replay, year, round_num, session)
        return _frame(replay, at)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"fastf1_replay_unavailable:{exc}") from exc
