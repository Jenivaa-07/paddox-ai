#!/usr/bin/env python3
"""Create a reproducibility and metric audit for PADDOX predictive experiments.

Run from the repository root after generating the processed splits and model
held-out prediction files. This script does not train models or modify existing
metrics; it recomputes checks from the saved data and predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
RACE_KEYS = ("season", "round")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def race_ids(frame: pd.DataFrame) -> set[tuple[int, int]]:
    missing = set(RACE_KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing race-key columns: {sorted(missing)}")
    return set(
        frame.loc[:, RACE_KEYS]
        .dropna()
        .astype(int)
        .itertuples(index=False, name=None)
    )


def split_summary(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    races = sorted(race_ids(frame))
    summary: dict[str, Any] = {
        "file": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "unique_races": len(races),
        "race_ids": [f"{season}-{round_no}" for season, round_no in races],
    }
    if races:
        summary["first_race"] = f"{races[0][0]}-{races[0][1]}"
        summary["last_race"] = f"{races[-1][0]}-{races[-1][1]}"
    return summary


def load_splits(kind: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    for split in SPLITS:
        path = ROOT / "data" / "processed" / "predictive" / f"{split}_{kind}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Required split not found: {path}. Build the predictive dataset first."
            )
        frame = pd.read_csv(path)
        frames[split] = frame
        summaries[split] = split_summary(path, frame)

    ids = {name: race_ids(frame) for name, frame in frames.items()}
    overlaps = {
        "train_val": sorted(ids["train"] & ids["val"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "val_test": sorted(ids["val"] & ids["test"]),
    }
    all_ids = {name: sorted(values) for name, values in ids.items()}
    order_check = None
    if all(all_ids[name] for name in SPLITS):
        order_check = (
            max(all_ids["train"]) < min(all_ids["val"])
            and max(all_ids["val"]) < min(all_ids["test"])
        )
    return frames, {
        "splits": summaries,
        "race_overlap": {key: [f"{s}-{r}" for s, r in value] for key, value in overlaps.items()},
        "no_race_overlap": all(not value for value in overlaps.values()),
        "strict_chronological_order": order_check,
    }


def clustered_bootstrap_delta(
    frame: pd.DataFrame,
    truth_col: str,
    model_col: str,
    baseline_col: str,
    iterations: int = 2000,
    seed: int = 42,
) -> list[float] | None:
    """Bootstrap paired (model MAE - baseline MAE) by whole race clusters."""
    grouped = list(frame.groupby(list(RACE_KEYS), sort=False))
    if len(grouped) < 2:
        return None
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(grouped), size=len(grouped))
        sample = pd.concat([grouped[index][1] for index in indices], ignore_index=True)
        delta = mean_absolute_error(sample[truth_col], sample[model_col]) - mean_absolute_error(
            sample[truth_col], sample[baseline_col]
        )
        deltas.append(float(delta))
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(relevance) == 0:
        return 0.0
    k = min(k, len(relevance))
    predicted_order = np.argsort(scores)[::-1][:k]
    ideal_order = np.argsort(relevance)[::-1][:k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[predicted_order] / discounts))
    idcg = float(np.sum(relevance[ideal_order] / discounts))
    return dcg / idcg if idcg > 0 else 0.0


def mean_race_ndcg(frame: pd.DataFrame, truth_col: str, score_col: str, k: int) -> float:
    values = [
        ndcg_at_k(group[truth_col].to_numpy(dtype=float), group[score_col].to_numpy(dtype=float), k)
        for _, group in frame.groupby(list(RACE_KEYS), sort=False)
    ]
    return float(np.mean(values)) if values else float("nan")


def audit_lstm(run_id: str, split_info: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / "artifacts" / "predictive" / "lstm_race" / run_id / "heldout_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"LSTM held-out predictions not found: {path}")
    frame = pd.read_csv(path)
    required = {"final_position", "pred_position", "grid_position", "rolling_avg_finish", *RACE_KEYS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"LSTM predictions missing columns: {sorted(missing)}")
    baselines = {
        "grid_position": frame["grid_position"].fillna(20.0),
        "rolling_average_finish": frame["rolling_avg_finish"].fillna(20.0),
    }
    result: dict[str, Any] = {
        "run_id": run_id,
        "predictions_file": str(path.relative_to(ROOT)),
        "predictions_sha256": sha256_file(path),
        "rows": int(len(frame)),
        "unique_races": int(frame.groupby(list(RACE_KEYS)).ngroups),
        "model": {
            "MAE": float(mean_absolute_error(frame.final_position, frame.pred_position)),
            "RMSE": float(np.sqrt(mean_squared_error(frame.final_position, frame.pred_position))),
            "Spearman": float(spearmanr(frame.final_position, frame.pred_position).statistic),
        },
        "baselines": {},
    }
    for name, predictions in baselines.items():
        temp = frame.assign(_baseline=predictions)
        result["baselines"][name] = {
            "MAE": float(mean_absolute_error(frame.final_position, predictions)),
            "paired_race_clustered_delta_model_minus_baseline_95pct_CI": clustered_bootstrap_delta(
                temp, "final_position", "pred_position", "_baseline"
            ),
        }
    return result


def audit_fantasy(run_id: str) -> dict[str, Any]:
    base = ROOT / "artifacts" / "predictive" / "rf_fantasy" / run_id
    path = base / "heldout_predictions.csv"
    train_path = ROOT / "data" / "processed" / "predictive" / "train_fantasy_final.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Fantasy held-out predictions not found: {path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Fantasy training split not found: {train_path}")

    frame = pd.read_csv(path)
    train = pd.read_csv(train_path)
    target = "fantasy_points_target"
    required = {target, "pred_points", "qualifying_position", *RACE_KEYS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fantasy predictions missing columns: {sorted(missing)}")
    if target not in train:
        raise ValueError(f"Training split is missing target column: {target}")

    # This is the exact proxy currently used by models/rf_fantasy_trainer.py.
    grid_proxy = 25 - frame["qualifying_position"].fillna(20.0)
    train_mean = float(train[target].mean())
    train_median = float(train[target].median())
    frame = frame.assign(
        _grid_proxy=grid_proxy,
        _train_mean=train_mean,
        _train_median=train_median,
    )
    result: dict[str, Any] = {
        "run_id": run_id,
        "predictions_file": str(path.relative_to(ROOT)),
        "predictions_sha256": sha256_file(path),
        "training_split_file": str(train_path.relative_to(ROOT)),
        "training_split_sha256": sha256_file(train_path),
        "rows": int(len(frame)),
        "unique_races": int(frame.groupby(list(RACE_KEYS)).ngroups),
        "model": {
            "MAE": float(mean_absolute_error(frame[target], frame.pred_points)),
            "RMSE": float(np.sqrt(mean_squared_error(frame[target], frame.pred_points))),
            "R2": float(r2_score(frame[target], frame.pred_points)),
            "Spearman": float(spearmanr(frame[target], frame.pred_points).statistic),
            "NDCG_global_5": ndcg_at_k(frame[target].to_numpy(), frame.pred_points.to_numpy(), 5),
            "NDCG_global_10": ndcg_at_k(frame[target].to_numpy(), frame.pred_points.to_numpy(), 10),
            "mean_race_NDCG_5": mean_race_ndcg(frame, target, "pred_points", 5),
            "mean_race_NDCG_10": mean_race_ndcg(frame, target, "pred_points", 10),
        },
        "baselines": {},
    }
    for name, column, definition in (
        ("grid_position_proxy", "_grid_proxy", "25 - qualifying_position; proxy baseline, not a training-mean baseline"),
        ("training_target_mean", "_train_mean", f"constant mean of training targets ({train_mean:.10g})"),
        ("training_target_median", "_train_median", f"constant median of training targets ({train_median:.10g})"),
    ):
        result["baselines"][name] = {
            "definition": definition,
            "MAE": float(mean_absolute_error(frame[target], frame[column])),
            "paired_race_clustered_delta_model_minus_baseline_95pct_CI": clustered_bootstrap_delta(
                frame, target, "pred_points", column
            ),
        }
    result["reported_grid_proxy_reduction_percent"] = float(
        100
        * (
            result["baselines"]["grid_position_proxy"]["MAE"]
            - result["model"]["MAE"]
        )
        / result["baselines"]["grid_position_proxy"]["MAE"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lstm-run", help="LSTM run directory ID, e.g. run_22f2c2a7")
    parser.add_argument("--fantasy-run", help="Fantasy run directory ID, e.g. run_6a942dcf")
    parser.add_argument("--output", default="reports/predictive/reproducibility_audit.json")
    args = parser.parse_args()

    race_splits, race_info = load_splits("race_seq")
    fantasy_splits, fantasy_info = load_splits("fantasy_final")
    report: dict[str, Any] = {
        "audit_version": 1,
        "note": "Recomputed from local split files and held-out predictions; no metrics are inferred or fabricated.",
        "race_dataset": race_info,
        "fantasy_dataset": fantasy_info,
        "checks": {
            "race_splits_have_no_overlap": race_info["no_race_overlap"],
            "race_splits_are_strictly_chronological": race_info["strict_chronological_order"],
            "fantasy_splits_have_no_overlap": fantasy_info["no_race_overlap"],
            "fantasy_splits_are_strictly_chronological": fantasy_info["strict_chronological_order"],
        },
    }
    if args.lstm_run:
        report["lstm"] = audit_lstm(args.lstm_run, race_info)
    if args.fantasy_run:
        report["fantasy"] = audit_fantasy(args.fantasy_run)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nAudit written to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
