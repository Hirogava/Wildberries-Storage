from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


@dataclass(frozen=True)
class TrackConfig:
    target_col: str
    forecast_points: int
    default_train_path: str
    default_test_path: str


TRACK_CONFIG: dict[str, TrackConfig] = {
    "solo": TrackConfig(
        target_col="target_1h",
        forecast_points=8,
        default_train_path="raw/train_solo_track.parquet",
        default_test_path="raw/test_solo_track.parquet",
    ),
    "team": TrackConfig(
        target_col="target_2h",
        forecast_points=10,
        default_train_path="raw/train_team_track.parquet",
        default_test_path="raw/test_team_track.parquet",
    ),
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Циклическое кодирование времени суток
    minutes_of_day = df["hour"] * 60 + df["minute"]
    df["tod_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)

    # Циклическое кодирование дня недели
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def build_features_for_predict(train_df: pd.DataFrame, *, target_col: str, forecast_points: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    train_df = train_df.copy()
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    train_df = train_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    train_df = add_calendar_features(train_df)
    route_group = train_df.groupby("route_id", sort=False)

    # Лаги по таргету
    for lag in [1, 2, 4, 8, 48]:
        train_df[f"{target_col}_lag_{lag}"] = route_group[target_col].shift(lag)

    # rolling-статистики только по прошлому
    for window in [2, 4, 8, 48]:
        train_df[f"{target_col}_roll_mean_{window}"] = route_group[target_col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        train_df[f"{target_col}_roll_std_{window}"] = route_group[target_col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std()
        )

    # Лаги по status_*
    status_cols = [c for c in train_df.columns if c.startswith("status_")]
    train_df["status_sum"] = train_df[status_cols].sum(axis=1)

    for lag in [1, 2, 4]:
        train_df[f"status_sum_lag_{lag}"] = train_df.groupby("route_id")["status_sum"].shift(lag)

    for col in status_cols:
        for lag in [1, 2]:
            train_df[f"{col}_lag_{lag}"] = route_group[col].shift(lag)

    future_target_cols = [f"target_step_{step}" for step in range(1, forecast_points + 1)]
    for step in range(1, forecast_points + 1):
        train_df[f"target_step_{step}"] = route_group[target_col].shift(-step)

    feature_cols = [
        col
        for col in train_df.columns
        if col not in {target_col, "timestamp", "id", *future_target_cols}
    ]

    return train_df, feature_cols, future_target_cols


def predict_submission(
    *,
    track: str,
    train_parquet: str | Path,
    test_parquet: str | Path,
    model_path: str | Path = "baseline_ridge",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    if track not in TRACK_CONFIG:
        raise ValueError(f"Unknown track: {track}. Expected one of: {sorted(TRACK_CONFIG.keys())}")

    cfg = TRACK_CONFIG[track]
    train_parquet = Path(train_parquet)
    test_parquet = Path(test_parquet)
    model_path = Path(model_path)

    if output_path is None:
        output_path = Path(f"submission_{track}.csv")
    else:
        output_path = Path(output_path)

    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)

    train_df, feature_cols, future_target_cols = build_features_for_predict(
        train_df, target_col=cfg.target_col, forecast_points=cfg.forecast_points
    )
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    test_df = test_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    inference_ts = train_df["timestamp"].max()
    test_model_df = train_df[train_df["timestamp"] == inference_ts].copy()
    X_test = test_model_df[feature_cols].copy()

    model = joblib.load(model_path)
    test_pred_df = pd.DataFrame(model.predict(X_test), columns=future_target_cols, index=test_model_df.index)

    # добавляем к прогнозу маршруты
    test_pred_df["route_id"] = X_test["route_id"].values

    # разворачиваем target_step_* в строки
    forecast_df = test_pred_df.melt(
        id_vars="route_id",
        value_vars=[c for c in test_pred_df.columns if c.startswith("target_step_")],
        var_name="step",
        value_name="forecast",
    )

    # достаем номер шага из target_step_1, target_step_2, ...
    forecast_df["step_num"] = forecast_df["step"].str.extract(r"(\d+)").astype(int)

    # строим timestamp: каждый шаг = +30 минут от времени прогноза
    forecast_df["timestamp"] = inference_ts + pd.to_timedelta(forecast_df["step_num"] * 30, unit="m")

    forecast_df = forecast_df[["route_id", "timestamp", "forecast"]].sort_values(
        ["route_id", "timestamp"]
    ).reset_index(drop=True)

    forecast_df = test_df.merge(forecast_df, "outer")[["id", "forecast"]]
    forecast_df = forecast_df.rename(columns={"forecast": "y_pred"})

    forecast_df.to_csv(output_path, index=False)
    return forecast_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=sorted(TRACK_CONFIG.keys()), default="team")
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--model-path", default="baseline_ridge")
    parser.add_argument("--output-path", default=None)

    args = parser.parse_args(argv)
    cfg = TRACK_CONFIG[args.track]

    train_parquet = args.train_parquet or cfg.default_train_path
    test_parquet = args.test_parquet or cfg.default_test_path
    output_path = args.output_path or f"submission_{args.track}.csv"

    predict_submission(
        track=args.track,
        train_parquet=train_parquet,
        test_parquet=test_parquet,
        model_path=args.model_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

