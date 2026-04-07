import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


TRAIN_PATH = Path("train_team_track.parquet")
TEST_PATH = Path("test_team_track.parquet")

TARGET_COL = "target_2h"
FORECAST_POINTS = 10
STATUS_COLS = [f"status_{index}" for index in range(1, 9)]
FUTURE_TARGET_COLS = [f"target_step_{step}" for step in range(1, FORECAST_POINTS + 1)]
MODEL_TAG = "team_track_lgbm_competition_stable"

TRAIN_DAYS = 14
RECENCY_HALFLIFE_DAYS = 3.5
FINAL_POST_SCALE = 1.005
RANDOM_STATE = 42

MODEL_PARAMS = {
    "objective": "mae",
    "n_estimators": 260,
    "learning_rate": 0.035,
    "num_leaves": 127,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": -1,
}

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MPLCONFIG_DIR = ARTIFACTS_DIR / "mplconfig"

CATEGORICAL_FEATURES = ["route_id", "office_from_id"]
FEATURE_COLS = [
    "route_id",
    "office_from_id",
    "current_target",
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
    "slot",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "status_sum",
    "status_mean",
    "status_std",
    "status_min",
    "status_max",
    "status_range",
    "status_last_first",
    *STATUS_COLS,
    *[f"status_diff_{index}" for index in range(1, 8)],
    *[f"status_sum_lag_{lag}" for lag in [1, 2, 4, 8, 12, 16, 24, 48, 96]],
    *[f"{column}_lag_{lag}" for column in STATUS_COLS for lag in [1, 2, 4, 8]],
    *[f"{TARGET_COL}_lag_{lag}" for lag in [1, 2, 4, 8, 12, 16, 24, 48, 96]],
    *[f"{TARGET_COL}_roll_mean_{window}" for window in [2, 4, 8, 16, 48, 96]],
    *[f"{TARGET_COL}_roll_std_{window}" for window in [2, 4, 8, 16, 48, 96]],
    *[f"target_diff_{left}_{right}" for left, right in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 48), (48, 96)]],
    "future_slot",
    "future_dayofweek",
    "route_slot_mean",
    "office_slot_mean",
    "global_slot_mean",
    "baseline",
]


@dataclass
class StepModelArtifact:
    step: int
    scale: float
    model: LGBMRegressor


def score_wape_rbias(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_flat = y_true.ravel().astype("float64")
    y_pred_flat = y_pred.ravel().astype("float64")
    wape = float(np.abs(y_pred_flat - y_true_flat).sum() / y_true_flat.sum())
    relative_bias = float(abs(y_pred_flat.sum() / y_true_flat.sum() - 1))
    return {
        "score": round(wape + relative_bias, 6),
        "wape": round(wape, 6),
        "relative_bias": round(relative_bias, 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train competition model for team_track.")
    parser.add_argument("--model-tag", default=MODEL_TAG, help="Artifact name prefix.")
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS, help="Rolling train window in days.")
    parser.add_argument("--halflife-days", type=float, default=RECENCY_HALFLIFE_DAYS, help="Recency weighting half-life.")
    parser.add_argument("--post-scale", type=float, default=FINAL_POST_SCALE, help="Global post-prediction scale.")
    parser.add_argument("--n-estimators", type=int, default=MODEL_PARAMS["n_estimators"], help="LightGBM trees.")
    parser.add_argument("--learning-rate", type=float, default=MODEL_PARAMS["learning_rate"], help="LightGBM learning rate.")
    parser.add_argument("--num-leaves", type=int, default=MODEL_PARAMS["num_leaves"], help="LightGBM num_leaves.")
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=MODEL_PARAMS["min_child_samples"],
        help="LightGBM min_child_samples.",
    )
    return parser.parse_args()


def build_feature_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True).copy()
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])

    for column in ["office_from_id", "route_id", *STATUS_COLS]:
        train_df[column] = pd.to_numeric(train_df[column], downcast="integer")
    train_df[TARGET_COL] = train_df[TARGET_COL].astype("float32")

    features = pd.DataFrame(
        {
            "route_id": train_df["route_id"],
            "office_from_id": train_df["office_from_id"],
            "timestamp": train_df["timestamp"],
            "current_target": train_df[TARGET_COL],
        }
    )

    features["hour"] = features["timestamp"].dt.hour.astype("int8")
    features["minute"] = features["timestamp"].dt.minute.astype("int8")
    features["dayofweek"] = features["timestamp"].dt.dayofweek.astype("int8")
    features["is_weekend"] = (features["dayofweek"] >= 5).astype("int8")
    features["slot"] = ((features["hour"].astype("int16") * 60 + features["minute"].astype("int16")) // 30).astype("int16")

    minutes_of_day = features["hour"].astype("int16") * 60 + features["minute"].astype("int16")
    features["tod_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440).astype("float32")
    features["tod_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440).astype("float32")
    features["dow_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7).astype("float32")
    features["dow_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7).astype("float32")

    status_frame = train_df[STATUS_COLS].astype("float32")
    for column in STATUS_COLS:
        features[column] = status_frame[column]

    features["status_sum"] = status_frame.sum(axis=1).astype("float32")
    features["status_mean"] = status_frame.mean(axis=1).astype("float32")
    features["status_std"] = status_frame.std(axis=1).fillna(0).astype("float32")
    features["status_min"] = status_frame.min(axis=1).astype("float32")
    features["status_max"] = status_frame.max(axis=1).astype("float32")
    features["status_range"] = (features["status_max"] - features["status_min"]).astype("float32")
    features["status_last_first"] = (train_df["status_8"] - train_df["status_1"]).astype("float32")

    for index in range(1, 8):
        features[f"status_diff_{index}"] = (train_df[f"status_{index + 1}"] - train_df[f"status_{index}"]).astype("float32")

    route_group_target = train_df.groupby("route_id", sort=False)
    route_group_features = features.groupby("route_id", sort=False)

    for lag in [1, 2, 4, 8, 12, 16, 24, 48, 96]:
        features[f"{TARGET_COL}_lag_{lag}"] = route_group_target[TARGET_COL].shift(lag).astype("float32")
        features[f"status_sum_lag_{lag}"] = route_group_features["status_sum"].shift(lag).astype("float32")

    for column in STATUS_COLS:
        for lag in [1, 2, 4, 8]:
            features[f"{column}_lag_{lag}"] = route_group_target[column].shift(lag).astype("float32")

    shifted_target = route_group_target[TARGET_COL].shift(1)
    for window in [2, 4, 8, 16, 48, 96]:
        features[f"{TARGET_COL}_roll_mean_{window}"] = shifted_target.rolling(window, min_periods=1).mean().astype("float32")
        features[f"{TARGET_COL}_roll_std_{window}"] = shifted_target.rolling(window, min_periods=2).std().fillna(0).astype("float32")

    for left, right in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 48), (48, 96)]:
        features[f"target_diff_{left}_{right}"] = (features[f"{TARGET_COL}_lag_{left}"] - features[f"{TARGET_COL}_lag_{right}"]).astype("float32")

    for step in range(1, FORECAST_POINTS + 1):
        features[f"target_step_{step}"] = route_group_target[TARGET_COL].shift(-step).astype("float32")

    return features


def build_temporal_priors(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    history = train_df[["route_id", "office_from_id", "timestamp", TARGET_COL]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history["slot"] = ((history["timestamp"].dt.hour * 60 + history["timestamp"].dt.minute) // 30).astype("int16")

    route_slot_mean = history.groupby(["route_id", "slot"])[TARGET_COL].mean().rename("route_slot_mean")
    office_slot_mean = history.groupby(["office_from_id", "slot"])[TARGET_COL].mean().rename("office_slot_mean")
    global_slot_mean = history.groupby("slot")[TARGET_COL].mean().rename("global_slot_mean")
    return route_slot_mean, office_slot_mean, global_slot_mean


def prepare_step_frame(
    base_frame: pd.DataFrame,
    step: int,
    route_slot_mean: pd.Series,
    office_slot_mean: pd.Series,
    global_slot_mean: pd.Series,
) -> pd.DataFrame:
    frame = base_frame.copy()
    future_timestamp = frame["timestamp"] + pd.to_timedelta(step * 30, unit="m")
    frame["future_slot"] = ((future_timestamp.dt.hour * 60 + future_timestamp.dt.minute) // 30).astype("int16")
    frame["future_dayofweek"] = future_timestamp.dt.dayofweek.astype("int8")

    frame = frame.join(route_slot_mean, on=["route_id", "future_slot"])
    frame = frame.join(office_slot_mean, on=["office_from_id", "future_slot"])
    frame = frame.join(global_slot_mean, on="future_slot")

    global_fill_value = float(global_slot_mean.mean())
    frame["route_slot_mean"] = frame["route_slot_mean"].fillna(frame["global_slot_mean"]).fillna(global_fill_value).astype("float32")
    frame["office_slot_mean"] = frame["office_slot_mean"].fillna(frame["global_slot_mean"]).fillna(global_fill_value).astype("float32")
    frame["global_slot_mean"] = frame["global_slot_mean"].fillna(global_fill_value).astype("float32")
    frame["baseline"] = (
        0.7 * frame["route_slot_mean"] + 0.2 * frame["office_slot_mean"] + 0.1 * frame["global_slot_mean"]
    ).astype("float32")
    return frame


def prepare_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[FEATURE_COLS].copy()
    for column in CATEGORICAL_FEATURES:
        x[column] = x[column].astype("category")
    return x


def build_sample_weights(timestamps: pd.Series) -> np.ndarray:
    age_in_days = (timestamps.max() - timestamps).dt.total_seconds() / 86400.0
    return np.exp(-age_in_days / RECENCY_HALFLIFE_DAYS).astype("float32")


def train_models(
    train_frame: pd.DataFrame,
    route_slot_mean: pd.Series,
    office_slot_mean: pd.Series,
    global_slot_mean: pd.Series,
) -> list[StepModelArtifact]:
    artifacts: list[StepModelArtifact] = []

    for step in range(1, FORECAST_POINTS + 1):
        target_col = f"target_step_{step}"
        step_frame = prepare_step_frame(train_frame, step, route_slot_mean, office_slot_mean, global_slot_mean)
        required_columns = ["current_target", "target_2h_lag_1", "target_2h_lag_2", "target_2h_lag_4", "target_2h_lag_48", target_col]
        step_frame = step_frame.loc[step_frame[required_columns].notna().all(axis=1)].copy()

        x_train = prepare_inputs(step_frame)
        y_train = step_frame[target_col].astype("float32")
        sample_weight = build_sample_weights(step_frame["timestamp"])

        model = LGBMRegressor(**MODEL_PARAMS)
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            categorical_feature=CATEGORICAL_FEATURES,
        )

        fit_predictions = np.maximum(0.0, model.predict(x_train)).astype("float32")
        scale = float(np.sum(y_train * sample_weight) / np.sum(fit_predictions * sample_weight))
        artifacts.append(StepModelArtifact(step=step, scale=scale, model=model))

    return artifacts


def predict_matrix(
    inference_frame: pd.DataFrame,
    artifacts: list[StepModelArtifact],
    route_slot_mean: pd.Series,
    office_slot_mean: pd.Series,
    global_slot_mean: pd.Series,
    post_scale: float,
) -> np.ndarray:
    prediction_matrix = np.zeros((len(inference_frame), FORECAST_POINTS), dtype="float32")

    for artifact in artifacts:
        step_frame = prepare_step_frame(inference_frame, artifact.step, route_slot_mean, office_slot_mean, global_slot_mean)
        x_inference = prepare_inputs(step_frame)
        predictions = np.maximum(0.0, artifact.model.predict(x_inference)).astype("float32")
        prediction_matrix[:, artifact.step - 1] = predictions * np.float32(artifact.scale * post_scale)

    return prediction_matrix


def build_submission(
    test_df: pd.DataFrame,
    inference_rows: pd.DataFrame,
    prediction_matrix: np.ndarray,
    inference_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    wide_predictions = pd.DataFrame(prediction_matrix, columns=FUTURE_TARGET_COLS, index=inference_rows.index)
    wide_predictions["route_id"] = inference_rows["route_id"].astype("int64").values

    forecast = wide_predictions.melt(
        id_vars="route_id",
        value_vars=FUTURE_TARGET_COLS,
        var_name="step",
        value_name="y_pred",
    )
    forecast["step_num"] = forecast["step"].str.extract(r"(\d+)").astype("int64")
    forecast["timestamp"] = inference_timestamp + pd.to_timedelta(forecast["step_num"] * 30, unit="m")
    forecast = forecast[["route_id", "timestamp", "y_pred"]]

    submission = test_df.copy()
    submission["timestamp"] = pd.to_datetime(submission["timestamp"])
    submission = submission.merge(forecast, on=["route_id", "timestamp"], how="left")
    submission["y_pred"] = submission["y_pred"].clip(lower=0).fillna(0).astype("float32")
    return submission[["id", "y_pred"]].sort_values("id").reset_index(drop=True)


def evaluate_cutoff(
    feature_frame: pd.DataFrame,
    train_df: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp,
) -> dict[str, object]:
    history_mask = feature_frame["timestamp"] < cutoff_timestamp
    train_window_mask = history_mask & (feature_frame["timestamp"] >= cutoff_timestamp - pd.Timedelta(days=TRAIN_DAYS))
    cutoff_rows = feature_frame.loc[feature_frame["timestamp"] == cutoff_timestamp].copy()

    route_slot_mean, office_slot_mean, global_slot_mean = build_temporal_priors(train_df.loc[train_df["timestamp"] < cutoff_timestamp].copy())
    artifacts = train_models(feature_frame.loc[train_window_mask].copy(), route_slot_mean, office_slot_mean, global_slot_mean)
    prediction_matrix = predict_matrix(cutoff_rows, artifacts, route_slot_mean, office_slot_mean, global_slot_mean, FINAL_POST_SCALE)
    actual_matrix = cutoff_rows[FUTURE_TARGET_COLS].to_numpy(dtype="float32")

    return {
        "cutoff_timestamp": cutoff_timestamp.isoformat(),
        "metrics": score_wape_rbias(actual_matrix, prediction_matrix),
        "rows": int(len(cutoff_rows)),
    }


def main():
    global MODEL_TAG, TRAIN_DAYS, RECENCY_HALFLIFE_DAYS, FINAL_POST_SCALE, MODEL_PARAMS

    args = parse_args()
    MODEL_TAG = args.model_tag
    TRAIN_DAYS = args.train_days
    RECENCY_HALFLIFE_DAYS = args.halflife_days
    FINAL_POST_SCALE = args.post_scale
    MODEL_PARAMS = {
        **MODEL_PARAMS,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
    }

    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for directory in [MODELS_DIR, SUBMISSIONS_DIR, METRICS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    feature_frame = build_feature_frame(train_df)

    inference_timestamp = train_df["timestamp"].max()
    train_window_start = inference_timestamp - pd.Timedelta(days=TRAIN_DAYS)
    train_window_frame = feature_frame.loc[feature_frame["timestamp"] >= train_window_start].copy()
    inference_rows = feature_frame.loc[feature_frame["timestamp"] == inference_timestamp].copy()

    route_slot_mean, office_slot_mean, global_slot_mean = build_temporal_priors(train_df)
    model_artifacts = train_models(train_window_frame, route_slot_mean, office_slot_mean, global_slot_mean)
    prediction_matrix = predict_matrix(inference_rows, model_artifacts, route_slot_mean, office_slot_mean, global_slot_mean, FINAL_POST_SCALE)
    submission = build_submission(test_df, inference_rows, prediction_matrix, inference_timestamp)

    evaluation_cutoffs = [
        inference_timestamp - pd.Timedelta(days=2),
        inference_timestamp - pd.Timedelta(days=1),
    ]
    validation_results = [evaluate_cutoff(feature_frame, train_df, cutoff) for cutoff in evaluation_cutoffs]

    model_artifact_path = MODELS_DIR / f"{MODEL_TAG}.joblib"
    submission_path = SUBMISSIONS_DIR / f"{MODEL_TAG}_submission.csv"
    metrics_path = METRICS_DIR / f"{MODEL_TAG}_validation.json"

    joblib.dump(
        {
            "models": [artifact.model for artifact in model_artifacts],
            "step_scales": [artifact.scale for artifact in model_artifacts],
            "feature_cols": FEATURE_COLS,
            "categorical_features": CATEGORICAL_FEATURES,
            "forecast_points": FORECAST_POINTS,
            "target_col": TARGET_COL,
            "train_days": TRAIN_DAYS,
            "final_post_scale": FINAL_POST_SCALE,
            "model_params": MODEL_PARAMS,
        },
        model_artifact_path,
    )
    submission.to_csv(submission_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "validation": validation_results,
                "train_days": TRAIN_DAYS,
                "recency_halflife_days": RECENCY_HALFLIFE_DAYS,
                "final_post_scale": FINAL_POST_SCALE,
                "model_params": MODEL_PARAMS,
                "step_scales": [round(artifact.scale, 8) for artifact in model_artifacts],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Saved model:", model_artifact_path.resolve())
    print("Saved submission:", submission_path.resolve())
    print("Saved metrics:", metrics_path.resolve())
    for result in validation_results:
        print(result["cutoff_timestamp"], result["metrics"])


if __name__ == "__main__":
    main()
