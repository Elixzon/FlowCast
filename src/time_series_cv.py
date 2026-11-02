from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/m5_splits/horizon_30.csv"
REPORT_FILE = ROOT / "reports/m5_time_series_cv_30m.json"

TARGET = "traffic_volume_t_plus_30"
GAP = pd.Timedelta(minutes=30)

CATEGORICAL = [
    "road_id",
    "weather_condition",
    "temperature_band",
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


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL,
            ),
            ("numerical", StandardScaler(), NUMERICAL),
        ],
        remainder="drop",
    )


def evaluate(y_true, predictions):
    predictions = np.maximum(predictions, 0)

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, predictions))
        ),
        "mape": float(
            np.mean(np.abs((y_true - predictions) / y_true)) * 100
        ),
    }


def make_model(model_name):
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

    return XGBRegressor(
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


def main():
    print("Loading chronological split data...", flush=True)

    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df = df.loc[df["split"] == "train"].copy()
    train_df = train_df.dropna(subset=[TARGET])

    missing_columns = [
        col for col in FEATURES + [TARGET, "timestamp"]
        if col not in train_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    timestamps = np.sort(train_df["timestamp"].unique())

    # Six chronological blocks:
    # first three form initial history; remaining three are validation folds.
    blocks = np.array_split(timestamps, 6)

    models = ["Random Forest", "XGBoost"]
    all_results = []

    for fold_number in range(1, 4):
        validation_times = blocks[fold_number + 2]

        validation_start = pd.Timestamp(validation_times[0])
        validation_end = pd.Timestamp(validation_times[-1])

        # Purge the final 30 minutes before validation begins.
        training_cutoff = validation_start - GAP

        fold_train = train_df[
            train_df["timestamp"] < training_cutoff
        ].copy()

        fold_val = train_df[
            train_df["timestamp"].isin(validation_times)
        ].copy()

        # Keep only complete feature rows; do not impute warm-up lags.
        fold_train = fold_train.dropna(subset=FEATURES + [TARGET])
        fold_val = fold_val.dropna(subset=FEATURES + [TARGET])

        if fold_train.empty or fold_val.empty:
            raise ValueError(f"Fold {fold_number} has no usable rows.")

        X_train = fold_train[FEATURES]
        y_train = fold_train[TARGET].to_numpy(dtype=float)

        X_val = fold_val[FEATURES]
        y_val = fold_val[TARGET].to_numpy(dtype=float)

        # Fit preprocessing on this fold's training data only.
        preprocessor = make_preprocessor()
        X_train_ready = preprocessor.fit_transform(X_train)
        X_val_ready = preprocessor.transform(X_val)

        print(
            f"\nFOLD {fold_number}"
            f"\nTrain: {fold_train['timestamp'].min()} "
            f"to {fold_train['timestamp'].max()}"
            f"\nValidation: {validation_start} to {validation_end}"
            f"\nRows: train={len(fold_train)}, "
            f"validation={len(fold_val)}",
            flush=True,
        )

        for model_name in models:
            print(f"Training {model_name}...", flush=True)

            model = make_model(model_name)
            model.fit(X_train_ready, y_train)

            predictions = model.predict(X_val_ready)
            metrics = evaluate(y_val, predictions)

            result = {
                "fold": fold_number,
                "model": model_name,
                "train_rows": int(len(fold_train)),
                "validation_rows": int(len(fold_val)),
                "training_cutoff": str(training_cutoff),
                "validation_start": str(validation_start),
                "validation_end": str(validation_end),
                **metrics,
            }

            all_results.append(result)

            print(
                f"{model_name} | "
                f"MAE={metrics['mae']:.3f} | "
                f"RMSE={metrics['rmse']:.3f} | "
                f"MAPE={metrics['mape']:.3f}%",
                flush=True,
            )

    results_df = pd.DataFrame(all_results)

    summary = (
        results_df.groupby("model")
        .agg(
            folds=("fold", "count"),
            mean_MAE=("mae", "mean"),
            mean_RMSE=("rmse", "mean"),
            mean_MAPE=("mape", "mean"),
            std_RMSE=("rmse", "std"),
        )
        .reset_index()
        .sort_values("mean_RMSE")
    )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "Expanding-window time-series cross-validation",
        "target": TARGET,
        "horizon_minutes": 30,
        "fold_count": 3,
        "gap_minutes": 30,
        "preprocessing_fit_per_fold": True,
        "test_data_used": False,
        "fold_results": all_results,
        "summary": summary.to_dict(orient="records"),
    }

    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\nCROSS-VALIDATION SUMMARY")
    print(summary.to_string(index=False))
    print("\nReport saved:", REPORT_FILE)


if __name__ == "__main__":
    main()
