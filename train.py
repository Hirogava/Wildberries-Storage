from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
import joblib



class WapePlusRbias:
    """Calculates as WAPE + Relative Bias."""

    @property
    def name(self) -> str:
        """Возвращает имя метрики."""
        return "wape_plus_rbias"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Рассчитывает значение метрики."""
        wape = (np.abs(y_pred - y_true)).sum() / y_true.sum()
        rbias = np.abs(y_pred.sum() / y_true.sum() - 1)
        return wape + rbias


metric = WapePlusRbias()

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 5)

# Модифицируйте в соответствии со своей задачей
TRACK = "team"  
TRAIN_DAYS = 14
MAX_TRAIN_ROWS = 1_500_000
RIDGE_ALPHA = 4.0
RANDOM_STATE = 42

script_dir = os.path.dirname(os.path.abspath(__file__))

# Меняйте конфигурацию при необходимости
TRACK_CONFIG = {
    "solo": {
        "train_path": os.path.join(script_dir, "raw", "train_solo_track.parquet"),
        "test_path": os.path.join(script_dir, "raw", "test_solo_track.parquet"),
        "target_col": "target_1h",
        "forecast_points": 8,
    },
    "team": {
        "train_path": os.path.join(script_dir, "raw", "train_team_track.parquet"),
        "test_path": os.path.join(script_dir, "raw", "test_team_track.parquet"),
        "target_col": "target_2h",
        "forecast_points": 10,
    },
}

CONFIG = TRACK_CONFIG[TRACK]
TARGET_COL = CONFIG["target_col"]
FORECAST_POINTS = CONFIG["forecast_points"]
FUTURE_TARGET_COLS = [f"target_step_{step}" for step in range(1, FORECAST_POINTS + 1)]

train_df = pd.read_parquet(CONFIG["train_path"])
test_df = pd.read_parquet(CONFIG["test_path"])

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

train_df = train_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
test_df = test_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)


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


train_df = add_calendar_features(train_df)
test_df = add_calendar_features(test_df)

route_group = train_df.groupby("route_id", sort=False)

# лаги по таргету
for lag in [1, 2, 4, 8, 48]:
    train_df[f"{TARGET_COL}_lag_{lag}"] = route_group[TARGET_COL].shift(lag)

# rolling-статистики только по прошлому
for window in [2, 4, 8, 48]:
    train_df[f"{TARGET_COL}_roll_mean_{window}"] = route_group[TARGET_COL].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    train_df[f"{TARGET_COL}_roll_std_{window}"] = route_group[TARGET_COL].transform(
        lambda s: s.shift(1).rolling(window, min_periods=2).std()
    )

# лаги по status_*
status_cols = [c for c in train_df.columns if c.startswith("status_")]

train_df["status_sum"] = train_df[status_cols].sum(axis=1)
for lag in [1, 2, 4]:
    train_df[f"status_sum_lag_{lag}"] = train_df.groupby("route_id")["status_sum"].shift(lag)

for col in status_cols:
    for lag in [1, 2]:
        train_df[f"{col}_lag_{lag}"] = route_group[col].shift(lag)

# разница между текущей суммой status и значением lag=1
train_df["status_sum_delta_1"] = train_df["status_sum"] - train_df["status_sum_lag_1"]

for step in range(1, FORECAST_POINTS + 1):
    train_df[f"target_step_{step}"] = route_group[TARGET_COL].shift(-step)

train_df[["route_id", "timestamp", TARGET_COL] + FUTURE_TARGET_COLS].head(10)
supervised_df = train_df.dropna(subset=FUTURE_TARGET_COLS).copy()

# Формируем 4 блока признаков для лёгкой абляции.
# Важно: никаких экзотических/ratio/interaction признаков - только то, что уже создаётся в feature engineering.

def uniq(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

# Базовые группы
status_cols = [f"status_{i}" for i in range(1, 9) if f"status_{i}" in train_df.columns]
calendar_cols = [
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
]
calendar_cols = [c for c in calendar_cols if c in train_df.columns]

# Лаги target
lag_steps = [1, 2, 4, 8, 48]
target_lag_cols = [f"{TARGET_COL}_lag_{lag}" for lag in lag_steps if f"{TARGET_COL}_lag_{lag}" in train_df.columns]

# Rolling по target
roll_windows = [2, 4, 8, 48]
target_roll_cols = []
for w in roll_windows:
    c1 = f"{TARGET_COL}_roll_mean_{w}"
    c2 = f"{TARGET_COL}_roll_std_{w}"
    if c1 in train_df.columns:
        target_roll_cols.append(c1)
    if c2 in train_df.columns:
        target_roll_cols.append(c2)

# История status
status_sum_cols = [
    "status_sum",
    "status_sum_lag_1",
    "status_sum_lag_2",
    "status_sum_lag_4",
]
status_sum_cols = [c for c in status_sum_cols if c in train_df.columns]

status_i_lags_cols = []
for i in range(1, 9):
    for lag in [1, 2]:
        col = f"status_{i}_lag_{lag}"
        if col in train_df.columns:
            status_i_lags_cols.append(col)

status_sum_delta_cols = [c for c in ["status_sum_delta_1"] if c in train_df.columns]

# Сами блоки абляции
block_A = uniq([c for c in ["route_id", "office_from_id", *status_cols, *calendar_cols] if c in train_df.columns])
block_B = uniq(block_A + target_lag_cols)
block_C = uniq(block_B + target_roll_cols)
block_D = uniq(block_C + status_sum_cols + status_i_lags_cols + status_sum_delta_cols)

feature_blocks = {
    "baseline_calendar_status": block_A,
    "+target_lags": block_B,
    "+target_rolling": block_C,
    "+status_history": block_D,
}

# Финальная модель для submission: используем самый полный блок D.
feature_cols = feature_blocks["+status_history"]

train_model_df = supervised_df[feature_cols + ["timestamp"] + FUTURE_TARGET_COLS].copy()
train_model_df = train_model_df.rename(columns={"timestamp": "source_timestamp"})

train_ts_max = train_model_df["source_timestamp"].max()
train_window_start = train_ts_max - pd.Timedelta(days=TRAIN_DAYS)
train_model_df = train_model_df[train_model_df["source_timestamp"] >= train_window_start].copy()

inference_ts = train_df["timestamp"].max()
test_model_df = train_df[train_df["timestamp"] == inference_ts]

train_model_df = train_model_df.sort_values("source_timestamp").copy()

# Более честный time-based split:
# - valid берём как последние 2 дня относительно max(source_timestamp)
# - fit берём всё, что раньше
# - если 2 дня дают слишком мало данных, расширяем valid до последних 3 дней
max_ts = train_model_df["source_timestamp"].max()

VALID_DAYS = 2
MIN_VALID_ROWS = 10_000  # если нужно более жёстко/мягко, поменяйте порог

valid_start = max_ts - pd.Timedelta(days=VALID_DAYS)

valid_df = train_model_df[train_model_df["source_timestamp"] >= valid_start].copy()
fit_df = train_model_df[train_model_df["source_timestamp"] < valid_start].copy()

if len(valid_df) < MIN_VALID_ROWS:
    VALID_DAYS = 3
    valid_start = max_ts - pd.Timedelta(days=VALID_DAYS)
    valid_df = train_model_df[train_model_df["source_timestamp"] >= valid_start].copy()
    fit_df = train_model_df[train_model_df["source_timestamp"] < valid_start].copy()

# Никакого случайного sample: ограничиваем fit только по времени.
# Мы уже отсортировали train_model_df по source_timestamp, поэтому tail берёт самые свежие строки fit.
if len(fit_df) > MAX_TRAIN_ROWS:
    fit_df = fit_df.tail(MAX_TRAIN_ROWS).copy()

print("Fit rows:", fit_df.shape)
print("Valid rows:", valid_df.shape)

X_fit = fit_df[feature_cols].copy()
y_fit = fit_df[FUTURE_TARGET_COLS].copy()

X_valid = valid_df[feature_cols].copy()
y_valid = valid_df[FUTURE_TARGET_COLS].copy()

X_test = test_model_df[feature_cols].copy()

categorical_features = ["route_id", "office_from_id"]

# LGBM ожидает категориальные признаки как str/category.
X_fit_lgbm = X_fit.copy()
X_valid_lgbm = X_valid.copy()
X_test_lgbm = X_test.copy()

for c in categorical_features:
    if c in X_fit_lgbm.columns:
        X_fit_lgbm[c] = X_fit_lgbm[c].astype("category")
    if c in X_valid_lgbm.columns:
        X_valid_lgbm[c] = X_valid_lgbm[c].astype("category")
    if c in X_test_lgbm.columns:
        X_test_lgbm[c] = X_test_lgbm[c].astype("category")

fit_pred_df = pd.DataFrame(index=X_fit_lgbm.index, columns=FUTURE_TARGET_COLS, dtype=float)
valid_pred_df = pd.DataFrame(index=X_valid_lgbm.index, columns=FUTURE_TARGET_COLS, dtype=float)
test_pred_df = pd.DataFrame(index=X_test_lgbm.index, columns=FUTURE_TARGET_COLS, dtype=float)

models = {}

for col in FUTURE_TARGET_COLS:
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        objective="mae",
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    model.fit(
        X_fit_lgbm,
        y_fit[col],
        categorical_feature=categorical_features,
    )

    models[col] = model

    fit_pred_df[col] = model.predict(X_fit_lgbm)
    valid_pred_df[col] = model.predict(X_valid_lgbm)
    test_pred_df[col] = model.predict(X_test_lgbm)

fit_pred_df = fit_pred_df.clip(lower=0)
valid_pred_df = valid_pred_df.clip(lower=0)
test_pred_df = test_pred_df.clip(lower=0)

print('Общая метрика на тесте:')
print(f'{metric.calculate(y_fit.to_numpy().flatten(), fit_pred_df.to_numpy().flatten()):.2f}')

print('Общая метрика на валидации:')
print(f'{metric.calculate(y_valid.to_numpy().flatten(), valid_pred_df.to_numpy().flatten()):.2f}')

# добавляем к прогнозу маршруты
test_pred_df['route_id'] = X_test['route_id']

# разворачиваем target_step_* в строки
forecast_df = test_pred_df.melt(
    id_vars="route_id",
    value_vars=[c for c in test_pred_df.columns if c.startswith("target_step_")],
    var_name="step",
    value_name="forecast"
)

# достаем номер шага из target_step_1, target_step_2, ...
forecast_df["step_num"] = forecast_df["step"].str.extract(r"(\d+)").astype(int)

# строим timestamp: каждый шаг = +30 минут от времени прогноза
forecast_df["timestamp"] = inference_ts + pd.to_timedelta(forecast_df["step_num"] * 30, unit="m")

# оставляем нужные столбцы
forecast_df = forecast_df[["route_id", "timestamp", "forecast"]].sort_values(
    ["route_id", "timestamp"]
).reset_index(drop=True)

forecast_df = test_df.merge(forecast_df, 'outer')[["id", "forecast"]]
forecast_df = forecast_df.rename(columns={"forecast": "y_pred"})

submission_path =  f"submission_lgbm_{TRACK}.csv"
joined_path =  f"test_with_forecast_{TRACK}.csv"

forecast_df.to_csv(submission_path, index=False)

print("submission saved to:", submission_path)
joblib.dump(models, filename='baseline_lgbm',)