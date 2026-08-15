import hashlib
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("JOLPICA_BASE_URL", "https://api.jolpi.ca/ergast/f1").rstrip("/")
DEFAULT_SEASONS = "2022,2023,2024,2025,2026"
USER_AGENT = os.getenv("JOLPICA_USER_AGENT", "PADDOX-AI/1.0 fantasy-bootstrap")
PAGE_LIMIT = 100

RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
POSITION_GAINED = 1
DNF_PENALTY = -10
DSQ_PENALTY = -15


def _configured_seasons():
    raw = os.getenv("FANTASY_BOOTSTRAP_SEASONS", DEFAULT_SEASONS)
    seasons = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            seasons.append(int(token))
    seasons = sorted(set(seasons))
    if len(seasons) < 2:
        raise RuntimeError("FANTASY_BOOTSTRAP_SEASONS must contain at least two seasons")
    return seasons


def _request_json(url, attempts=3, timeout=25):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(min(4, attempt * 1.5))
    raise RuntimeError(f"Jolpica request failed after {attempts} attempts: {url}: {last_error}")


def _season_results(season):
    rows = []
    offset = 0
    total = None

    while total is None or offset < total:
        query = urllib.parse.urlencode({"limit": PAGE_LIMIT, "offset": offset})
        url = f"{BASE_URL}/{season}/results/?{query}"
        payload = _request_json(url)
        mrdata = payload.get("MRData", {})
        total = int(mrdata.get("total") or 0)
        races = mrdata.get("RaceTable", {}).get("Races", []) or []

        page_rows = 0
        for race in races:
            round_no = int(race.get("round") or 0)
            race_name = race.get("raceName") or ""
            for result in race.get("Results", []) or []:
                driver = result.get("Driver") or {}
                constructor = result.get("Constructor") or {}
                driver_id = driver.get("driverId") or str(result.get("number") or "")
                if not driver_id:
                    continue
                try:
                    final_position = int(result.get("position") or 0)
                except (TypeError, ValueError):
                    continue
                try:
                    grid = int(result.get("grid") or 0)
                except (TypeError, ValueError):
                    grid = 0

                rows.append({
                    "season": int(season),
                    "round": round_no,
                    "race_name": race_name,
                    "driver_id": driver_id,
                    "constructor_id": constructor.get("constructorId") or constructor.get("name") or "Unknown",
                    "grid": grid,
                    "final_position": final_position,
                    "status": str(result.get("status") or "Unknown"),
                })
                page_rows += 1

        offset += int(mrdata.get("limit") or PAGE_LIMIT)
        if total == 0 or page_rows == 0:
            break
        time.sleep(0.12)

    return rows


def _build_dataset(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No Jolpica race results were returned")

    df = df.sort_values(["season", "round", "final_position", "driver_id"]).reset_index(drop=True)
    field_sizes = df.groupby(["season", "round"])["driver_id"].transform("count").clip(lower=1)
    df["field_size"] = field_sizes.astype(int)

    grid = pd.to_numeric(df["grid"], errors="coerce")
    df["qualifying_position"] = grid.where(grid > 0, df["field_size"]).astype(float)
    df["final_position"] = pd.to_numeric(df["final_position"], errors="coerce").fillna(df["field_size"]).astype(float)

    df["rolling_avg_finish"] = (
        df.groupby("driver_id", group_keys=False)["final_position"]
          .apply(lambda series: series.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_avg_finish"] = df["rolling_avg_finish"].fillna(10.0)

    def fantasy_target(row):
        pos = int(row["final_position"])
        grid_pos = float(row["qualifying_position"])
        points = RACE_POINTS.get(pos, 0)
        points += max(0.0, grid_pos - float(pos)) * POSITION_GAINED

        status = str(row.get("status") or "")
        finished = (
            "Finished" in status
            or "+1 Lap" in status
            or "+2 Laps" in status
            or "+3 Laps" in status
        )
        if not finished:
            if "DSQ" in status.upper() or "DISQUAL" in status.upper():
                points += DSQ_PENALTY
            else:
                points += DNF_PENALTY
        return float(points)

    df["fantasy_points_target"] = df.apply(fantasy_target, axis=1)
    return df


def _ndcg_at_k(y_true, y_pred, k):
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(y_pred)[::-1][:k]
    ideal = np.argsort(y_true)[::-1][:k]
    dcg = sum(float(y_true[i]) / np.log2(rank + 2) for rank, i in enumerate(order))
    idcg = sum(float(y_true[i]) / np.log2(rank + 2) for rank, i in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _build_pipeline():
    numeric_features = ["qualifying_position", "rolling_avg_finish"]
    categorical_features = ["constructor_id"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_runtime_fantasy_model():
    seasons = _configured_seasons()
    logger.info("Bootstrapping fantasy RF from Jolpica seasons: %s", seasons)

    rows = []
    for season in seasons:
        season_rows = _season_results(season)
        logger.info("Fetched %s fantasy training rows for %s", len(season_rows), season)
        rows.extend(season_rows)

    df = _build_dataset(rows)
    latest_season = max(seasons)
    test_df = df[df["season"] == latest_season].copy()
    train_df = df[df["season"] < latest_season].copy()

    if len(test_df) < 40 or len(train_df) < 200:
        races = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
        split_idx = max(1, int(len(races) * 0.85))
        train_races = races.iloc[:split_idx]
        test_races = races.iloc[split_idx:]
        train_df = df.merge(train_races, on=["season", "round"], how="inner")
        test_df = df.merge(test_races, on=["season", "round"], how="inner")

    if len(train_df) < 100 or len(test_df) < 20:
        raise RuntimeError(
            f"Insufficient real F1 data to bootstrap fantasy model: train={len(train_df)}, test={len(test_df)}"
        )

    features = ["qualifying_position", "rolling_avg_finish", "constructor_id"]
    target = "fantasy_points_target"

    pipeline = _build_pipeline()
    pipeline.fit(train_df[features], train_df[target].values)

    y_true = test_df[target].to_numpy(dtype=float)
    y_pred = pipeline.predict(test_df[features])
    baseline = 25 - test_df["qualifying_position"].fillna(20.0).to_numpy(dtype=float)
    spearman = spearmanr(y_true, y_pred).correlation

    dataset_fingerprint = hashlib.sha256(
        df[["season", "round", "driver_id", "constructor_id", "qualifying_position", "rolling_avg_finish", target]]
        .to_csv(index=False)
        .encode("utf-8")
    ).hexdigest()[:10]
    run_id = f"run_live_{dataset_fingerprint}"
    out_dir = os.path.join("artifacts", "predictive", "rf_fantasy", run_id)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearman) if np.isfinite(spearman) else 0.0,
        "NDCG_5": _ndcg_at_k(y_true, y_pred, 5),
        "NDCG_10": _ndcg_at_k(y_true, y_pred, 10),
        "baseline_MAE": float(mean_absolute_error(y_true, baseline)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_seasons": sorted(int(v) for v in train_df["season"].unique()),
        "test_seasons": sorted(int(v) for v in test_df["season"].unique()),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with open(os.path.join(out_dir, "feature_schema.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "numeric_features": ["qualifying_position", "rolling_avg_finish"],
            "categorical_features": ["constructor_id"],
            "description": "Runtime bootstrap of the PADDOX Random Forest fantasy predictor from real Jolpica race results.",
        }, handle, indent=2)

    model_hash = _sha256(model_path)
    provenance = {
        "run_id": run_id,
        "source": "Jolpica/Ergast-compatible F1 results",
        "source_url": BASE_URL,
        "seasons": seasons,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sha256": model_hash,
        "algorithm": "RandomForestRegressor",
        "runtime_bootstrap": True,
        "scoring_version": "PADDOX_FANTASY_V1",
    }
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)

    current_model_path = os.path.join("artifacts", "predictive", "rf_fantasy", "current_model.json")
    with open(current_model_path, "w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id}, handle, indent=2)

    logger.info(
        "Runtime fantasy RF ready: %s (sha256=%s, train=%s, test=%s, MAE=%.3f)",
        run_id, model_hash, len(train_df), len(test_df), metrics["MAE"]
    )
    return {"run_id": run_id, "model_path": model_path, "sha256": model_hash, "metrics": metrics}


def ensure_runtime_fantasy_model():
    return train_runtime_fantasy_model()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = ensure_runtime_fantasy_model()
    print(json.dumps(result, indent=2))
