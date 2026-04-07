import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from train_team_model import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    FORECAST_POINTS,
    FUTURE_TARGET_COLS,
    MPLCONFIG_DIR,
    TARGET_COL,
    build_feature_frame,
    build_temporal_priors,
    prepare_inputs,
    prepare_step_frame,
    score_wape_rbias,
)


TRAIN_PATH = Path("train_team_track.parquet")
SEARCH_RESULTS_PATH = ARTIFACTS_DIR / "metrics" / "team_track_fast_search.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast leaderboard-oriented search for team_track.")
    parser.add_argument("--train-days", type=int, nargs="+", default=[10, 14], help="Candidate train windows in days.")
    parser.add_argument("--halflife-days", type=float, nargs="+", default=[2.0, 3.5], help="Candidate recency half-lives.")
    parser.add_argument("--post-scale", type=float, nargs="+", default=[1.0, 1.005, 1.01], help="Candidate global post-scales.")
    parser.add_argument("--n-estimators", type=int, nargs="+", default=[180, 220], help="Candidate number of trees.")
    parser.add_argument("--learning-rate", type=float, nargs="+", default=[0.04], help="Candidate learning rates.")
    parser.add_argument("--num-leaves", type=int, nargs="+", default=[127], help="Candidate num_leaves.")
    parser.add_argument("--min-child-samples", type=int, nargs="+", default=[40], help="Candidate min_child_samples.")
    parser.add_argument("--cutoff-count", type=int, default=3, help="How many recent same-weekday cutoffs to evaluate.")
    parser.add_argument("--limit", type=int, default=12, help="Limit number of searched configs after cartesian product.")
    parser.add_argument("--output", type=Path, default=SEARCH_RESULTS_PATH, help="Where to save search results JSON.")
    return parser.parse_args()


def get_leaderboard_like_cutoffs(train_df: pd.DataFrame, count: int) -> list[pd.Timestamp]:
    max_timestamp = pd.to_datetime(train_df["timestamp"]).max()
    same_weekday = train_df.loc[
        (pd.to_datetime(train_df["timestamp"]).dt.dayofweek == max_timestamp.dayofweek)
        & (pd.to_datetime(train_df["timestamp"]).dt.hour == max_timestamp.hour)
        & (pd.to_datetime(train_df["timestamp"]).dt.minute == max_timestamp.minute),
        "timestamp",
    ]
    cutoff_values = sorted(pd.to_datetime(same_weekday).unique())
    cutoff_values = [timestamp for timestamp in cutoff_values if timestamp < max_timestamp]
    return cutoff_values[-count:]


def build_sample_weights(timestamps: pd.Series, halflife_days: float) -> np.ndarray:
    age_in_days = (timestamps.max() - timestamps).dt.total_seconds() / 86400.0
    return np.exp(-age_in_days / halflife_days).astype("float32")


def build_model_params(config: dict[str, float | int]) -> dict[str, float | int | str]:
    return {
        "objective": "mae",
        "n_estimators": int(config["n_estimators"]),
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "min_child_samples": int(config["min_child_samples"]),
        "subsample": 0.8,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }


def evaluate_config(
    feature_frame: pd.DataFrame,
    train_df: pd.DataFrame,
    cutoff_timestamps: list[pd.Timestamp],
    config: dict[str, float | int],
) -> dict[str, object]:
    model_params = build_model_params(config)
    train_days = int(config["train_days"])
    halflife_days = float(config["halflife_days"])
    post_scale = float(config["post_scale"])

    cutoff_results: list[dict[str, object]] = []
    all_actual: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []

    for cutoff_timestamp in cutoff_timestamps:
        history_mask = feature_frame["timestamp"] < cutoff_timestamp
        train_window_mask = history_mask & (feature_frame["timestamp"] >= cutoff_timestamp - pd.Timedelta(days=train_days))
        train_window_frame = feature_frame.loc[train_window_mask].copy()
        cutoff_rows = feature_frame.loc[feature_frame["timestamp"] == cutoff_timestamp].copy()

        route_slot_mean, office_slot_mean, global_slot_mean = build_temporal_priors(
            train_df.loc[train_df["timestamp"] < cutoff_timestamp].copy()
        )

        prediction_matrix = np.zeros((len(cutoff_rows), FORECAST_POINTS), dtype="float32")

        for step in range(1, FORECAST_POINTS + 1):
            target_col = f"target_step_{step}"
            step_frame = prepare_step_frame(train_window_frame, step, route_slot_mean, office_slot_mean, global_slot_mean)
            required_columns = ["current_target", "target_2h_lag_1", "target_2h_lag_2", "target_2h_lag_4", "target_2h_lag_48", target_col]
            step_frame = step_frame.loc[step_frame[required_columns].notna().all(axis=1)].copy()

            x_train = prepare_inputs(step_frame)
            y_train = step_frame[target_col].astype("float32")
            sample_weight = build_sample_weights(step_frame["timestamp"], halflife_days)

            model = LGBMRegressor(**model_params)
            model.fit(
                x_train,
                y_train,
                sample_weight=sample_weight,
                categorical_feature=CATEGORICAL_FEATURES,
            )

            fit_predictions = np.maximum(0.0, model.predict(x_train)).astype("float32")
            step_scale = float(np.sum(y_train * sample_weight) / np.sum(fit_predictions * sample_weight))

            inference_step_frame = prepare_step_frame(cutoff_rows, step, route_slot_mean, office_slot_mean, global_slot_mean)
            x_inference = prepare_inputs(inference_step_frame)
            inference_predictions = np.maximum(0.0, model.predict(x_inference)).astype("float32")
            prediction_matrix[:, step - 1] = inference_predictions * np.float32(step_scale * post_scale)

        actual_matrix = cutoff_rows[FUTURE_TARGET_COLS].to_numpy(dtype="float32")
        metrics = score_wape_rbias(actual_matrix, prediction_matrix)

        cutoff_results.append(
            {
                "cutoff_timestamp": cutoff_timestamp.isoformat(),
                "metrics": metrics,
                "rows": int(len(cutoff_rows)),
            }
        )
        all_actual.append(actual_matrix)
        all_predicted.append(prediction_matrix)

    overall_metrics = score_wape_rbias(np.concatenate(all_actual), np.concatenate(all_predicted))
    return {
        "config": {
            "train_days": train_days,
            "halflife_days": halflife_days,
            "post_scale": post_scale,
            **model_params,
        },
        "overall_metrics": overall_metrics,
        "cutoffs": cutoff_results,
    }


def main() -> None:
    args = parse_args()

    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(TRAIN_PATH)
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    feature_frame = build_feature_frame(train_df)
    cutoff_timestamps = get_leaderboard_like_cutoffs(train_df, args.cutoff_count)

    candidate_grid = list(
        itertools.product(
            args.train_days,
            args.halflife_days,
            args.post_scale,
            args.n_estimators,
            args.learning_rate,
            args.num_leaves,
            args.min_child_samples,
        )
    )[: args.limit]

    results: list[dict[str, object]] = []
    for index, values in enumerate(candidate_grid, start=1):
        config = {
            "train_days": values[0],
            "halflife_days": values[1],
            "post_scale": values[2],
            "n_estimators": values[3],
            "learning_rate": values[4],
            "num_leaves": values[5],
            "min_child_samples": values[6],
        }
        print(f"[{index}/{len(candidate_grid)}] {config}")
        result = evaluate_config(feature_frame, train_df, cutoff_timestamps, config)
        print(" ->", result["overall_metrics"])
        results.append(result)

    ranked_results = sorted(results, key=lambda item: item["overall_metrics"]["score"])
    payload = {
        "cutoff_timestamps": [timestamp.isoformat() for timestamp in cutoff_timestamps],
        "searched_configs": len(candidate_grid),
        "results": ranked_results,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if ranked_results:
        print("Best config:", ranked_results[0]["config"])
        print("Best metrics:", ranked_results[0]["overall_metrics"])
    print("Saved search results:", args.output.resolve())


if __name__ == "__main__":
    main()
