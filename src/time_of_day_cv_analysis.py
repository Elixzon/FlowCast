from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/m5_splits/horizon_30.csv"
REPORT_DIR = ROOT / "reports"

TARGET = "traffic_volume_t_plus_30"
GAP = pd.Timedelta(minutes=30)

CATEGORICAL = [
    "road_id", "weather_condition", "temperature_band"
]

NUMERICAL = [
    "latitude", "longitude", "road_capacity",
    "temperature", "rainfall", "visibility",
    "public_holiday", "event_flag", "roadwork_flag",
    "hour", "day_of_week", "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "is_weekend", "is_peak_period", "rain_flag",
    "low_visibility_flag", "holiday_peak_interaction",
    "event_proximity_flag",
    "traffic_volume_lag_1", "traffic_volume_lag_2",
    "traffic_volume_lag_48",
    "traffic_volume_rolling_mean_4",
    "traffic_volume_rolling_std_4",
    "traffic_volume_rolling_mean_8",
    "traffic_volume_rolling_std_8",
    "avg_speed_lag_1", "avg_speed_lag_2",
    "avg_speed_lag_48",
    "avg_speed_rolling_mean_4",
    "avg_speed_rolling_std_4",
    "avg_speed_rolling_mean_8",
    "avg_speed_rolling_std_8",
]

FEATURES = CATEGORICAL + NUMERICAL


def assign_period(hour):
    if 7 <= hour < 10:
        return "Morning peak"
    if 10 <= hour < 16:
        return "Afternoon"
    if 16 <= hour < 20:
        return "Evening peak"
    return "Night"


def main():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train = df[df["split"] == "train"].dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    timestamps = np.sort(train["timestamp"].unique())
    blocks = np.array_split(timestamps, 6)

    results = []

    for fold in range(1, 4):
        val_times = blocks[fold + 2]
        val_start = pd.Timestamp(val_times[0])
        cutoff = val_start - GAP

        fold_train = train[train["timestamp"] < cutoff]
        fold_val = train[train["timestamp"].isin(val_times)].copy()

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERICAL),
        ])

        X_train = preprocessor.fit_transform(fold_train[FEATURES])
        X_val = preprocessor.transform(fold_val[FEATURES])

        model = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            objective="reg:squarederror",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            tree_method="hist",
        )

        print(f"Training fold {fold}...", flush=True)
        model.fit(X_train, fold_train[TARGET])

        predictions = np.maximum(model.predict(X_val), 0)

        fold_val["prediction"] = predictions
        fold_val["absolute_error"] = abs(
            fold_val[TARGET] - predictions
        )
        fold_val["percentage_error"] = (
            fold_val["absolute_error"] / fold_val[TARGET] * 100
        )

        fold_val["period"] = fold_val["timestamp"].dt.hour.map(
            assign_period
        )

        grouped = fold_val.groupby("period").agg(
            samples=(TARGET, "count"),
            MAE=("absolute_error", "mean"),
            MAPE=("percentage_error", "mean"),
        ).reset_index()

        grouped["fold"] = fold
        results.append(grouped)

    result = pd.concat(results, ignore_index=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        REPORT_DIR / "m5_time_of_day_cv_30m.csv",
        index=False
    )

    summary = result.groupby("period").agg(
        total_samples=("samples", "sum"),
        mean_MAE=("MAE", "mean"),
        mean_MAPE=("MAPE", "mean"),
    ).sort_values("mean_MAPE", ascending=False)

    print("\nTIME-OF-DAY ERROR ANALYSIS")
    print(summary.round(3).to_string())
    print("\nSaved: reports/m5_time_of_day_cv_30m.csv")


if __name__ == "__main__":
    main()
