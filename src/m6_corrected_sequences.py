"""Leakage-controlled sequence preparation for corrected M6 experiments."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/processed/m5_splits_corrected_v2"
OUTPUT_DIR = ROOT / "data/processed/m6_sequences_corrected_v2"
REPORT_DIR = ROOT / "reports/corrected_v2"
SEQUENCE_LENGTH = 48
HORIZONS = [30, 60, 90, 120]
CATEGORICAL = [
    "road_id", "weather_condition", "temperature_band", "congestion_level",
]
NUMERIC = [
    "latitude", "longitude", "road_capacity", "temperature",
    "rainfall", "visibility", "public_holiday", "event_flag",
    "roadwork_flag", "hour", "day_of_week", "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos", "is_weekend", "is_peak_period",
    "rain_flag", "low_visibility_flag", "holiday_peak_interaction",
    "traffic_volume", "vehicle_count", "avg_speed", "occupancy",
    "travel_time", "accident_count", "signal_timing",
    "traffic_volume_lag_1", "traffic_volume_lag_2", "traffic_volume_lag_48",
    "traffic_volume_rolling_mean_4", "traffic_volume_rolling_std_4",
    "traffic_volume_rolling_mean_8", "traffic_volume_rolling_std_8",
    "avg_speed_lag_1", "avg_speed_lag_2", "avg_speed_lag_48",
    "avg_speed_rolling_mean_4", "avg_speed_rolling_std_4",
    "avg_speed_rolling_mean_8", "avg_speed_rolling_std_8",
]
MINMAX_NUMERIC = [
    "visibility", "public_holiday", "event_flag", "roadwork_flag", "hour",
    "day_of_week", "hour_sin", "hour_cos", "day_of_week_sin",
    "day_of_week_cos", "is_weekend", "is_peak_period", "rain_flag",
    "low_visibility_flag", "holiday_peak_interaction",
]
STANDARD_NUMERIC = [
    column for column in NUMERIC if column not in MINMAX_NUMERIC
]
SPLITS = ["train", "validation", "test"]


def prepare_horizon(
    horizon: int,
    split_dir: Path = SPLIT_DIR,
    output_dir: Path = OUTPUT_DIR,
    report_dir: Path = REPORT_DIR,
) -> dict:
    """Fit feature encoders on training rows and create within-split sequences."""
    split_file = split_dir / f"horizon_{horizon}.csv"
    if not split_file.exists():
        raise FileNotFoundError(f"Corrected M6 source split not found: {split_file}")
    frame = pd.read_csv(split_file)
    target = f"traffic_volume_t_plus_{horizon}"
    required = {"road_id", "timestamp", "split", target, *CATEGORICAL, *NUMERIC}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing corrected M6 sequence columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["road_id"] = frame["road_id"].astype(str)
    frame = frame.sort_values(["road_id", "timestamp"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["road_id", "timestamp"]).any():
        raise ValueError(f"Duplicate road/timestamp keys in {split_file}")

    split_order = {name: index for index, name in enumerate(SPLITS)}
    for road_id, road in frame.groupby("road_id", sort=False):
        order = road["split"].map(split_order).to_numpy()
        if np.any(np.diff(order) < 0):
            raise ValueError(f"Split chronology violation for road {road_id}")

    train_mask = frame["split"].eq("train")
    numeric_imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    numeric_train_values = numeric_imputer.fit_transform(
        frame.loc[train_mask, NUMERIC]
    )
    numeric_train = pd.DataFrame(
        numeric_train_values,
        columns=NUMERIC,
    )
    scaler = StandardScaler()
    scaler.fit(numeric_train[STANDARD_NUMERIC])
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(numeric_train[MINMAX_NUMERIC])
    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    categorical_values = frame[CATEGORICAL].fillna("Unknown").astype(str)
    categorical_encoder.fit(categorical_values.loc[train_mask])

    numeric_frame = frame[NUMERIC]
    imputed_numeric = numeric_imputer.transform(numeric_frame)
    numeric_frame = pd.DataFrame(imputed_numeric, columns=NUMERIC)
    numeric_values = numeric_frame.to_numpy(dtype=np.float64)
    scaled_numeric = np.full(numeric_values.shape, np.nan, dtype=np.float32)
    valid_numeric_rows = np.isfinite(numeric_values).all(axis=1)
    scaled_numeric[
        np.ix_(valid_numeric_rows, [NUMERIC.index(column) for column in STANDARD_NUMERIC])
    ] = scaler.transform(
        numeric_frame.loc[valid_numeric_rows, STANDARD_NUMERIC]
    ).astype(np.float32)
    scaled_numeric[
        np.ix_(valid_numeric_rows, [NUMERIC.index(column) for column in MINMAX_NUMERIC])
    ] = minmax_scaler.transform(
        numeric_frame.loc[valid_numeric_rows, MINMAX_NUMERIC]
    ).astype(np.float32)
    categorical_encoded = categorical_encoder.transform(
        categorical_values
    ).astype(np.float32)
    feature_matrix = np.concatenate([scaled_numeric, categorical_encoded], axis=1)
    feature_names = NUMERIC + list(
        categorical_encoder.get_feature_names_out(CATEGORICAL)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    preprocess_path = output_dir / f"sequence_preprocessing_{horizon}m.joblib"
    joblib.dump(
        {
            "numeric_imputer": numeric_imputer,
            "scaler": scaler,
            "minmax_scaler": minmax_scaler,
            "categorical_encoder": categorical_encoder,
            "numeric_columns": NUMERIC,
            "standard_scaled_columns": STANDARD_NUMERIC,
            "minmax_scaled_columns": MINMAX_NUMERIC,
            "categorical_columns": CATEGORICAL,
            "feature_names": feature_names,
            "fit_partition": "train",
            "horizon_minutes": horizon,
        },
        preprocess_path,
    )

    sequence_parts = {
        split: {"X": [], "y": [], "road_id": [], "timestamp": [], "target_timestamp": []}
        for split in SPLITS
    }
    counters = {
        split: {"candidate_windows": 0, "gap_rejections": 0, "incomplete_rejections": 0}
        for split in SPLITS
    }
    expected_step = np.timedelta64(30, "m")

    for (road_id, split_name), part in frame.groupby(
        ["road_id", "split"], sort=False
    ):
        part = part.sort_values("timestamp", kind="stable")
        if len(part) < SEQUENCE_LENGTH:
            continue
        indices = part.index.to_numpy()
        timestamps = part["timestamp"].to_numpy()
        targets = pd.to_numeric(part[target], errors="coerce").to_numpy(dtype=float)

        for end in range(SEQUENCE_LENGTH - 1, len(part)):
            counters[split_name]["candidate_windows"] += 1
            start = end - SEQUENCE_LENGTH + 1
            window_times = timestamps[start:end + 1]
            if not np.all(np.diff(window_times) == expected_step):
                counters[split_name]["gap_rejections"] += 1
                continue
            X_window = feature_matrix[indices[start:end + 1]]
            target_value = targets[end]
            if not np.isfinite(X_window).all() or not np.isfinite(target_value):
                counters[split_name]["incomplete_rejections"] += 1
                continue
            issue_time = pd.Timestamp(timestamps[end])
            target_time = issue_time + pd.Timedelta(minutes=horizon)
            sequence_parts[split_name]["X"].append(X_window)
            sequence_parts[split_name]["y"].append(target_value)
            sequence_parts[split_name]["road_id"].append(road_id)
            sequence_parts[split_name]["timestamp"].append(issue_time.isoformat())
            sequence_parts[split_name]["target_timestamp"].append(target_time.isoformat())

    summary = {
        "horizon_minutes": horizon,
        "target": target,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "preprocessing_fit_partition": "train",
        "numeric_missing_value_imputation": (
            "Training-only per-feature median; future targets are never imputed."
        ),
        "numeric_scaling": {
            "standard_scaled_columns": STANDARD_NUMERIC,
            "minmax_scaled_columns": MINMAX_NUMERIC,
            "fit_partition": "train",
        },
        "partition_boundaries_are_not_crossed": True,
        "road_boundaries_are_not_crossed": True,
        "window_continuity": "exactly 30 minutes between adjacent input rows",
        "counters": counters,
        "splits": {},
        "preprocessor_artifact": str(preprocess_path.relative_to(ROOT)),
    }
    for split_name, values in sequence_parts.items():
        if not values["X"]:
            raise ValueError(f"No valid {split_name} M6 sequences for {horizon}m.")
        X = np.stack(values["X"]).astype(np.float32)
        y = np.asarray(values["y"], dtype=np.float32)
        path = output_dir / f"{split_name}_{horizon}m.npz"
        np.savez_compressed(
            path,
            X=X,
            y=y,
            road_id=np.asarray(values["road_id"]),
            timestamp=np.asarray(values["timestamp"]),
            target_timestamp=np.asarray(values["target_timestamp"]),
        )
        summary["splits"][split_name] = {
            "sequences": len(y),
            "X_shape": list(X.shape),
            "non_finite_values": int((~np.isfinite(X)).sum()),
            "target_min": float(np.min(y)),
            "target_max": float(np.max(y)),
            "artifact": str(path.relative_to(ROOT)),
        }

    (report_dir / f"m6_sequence_{horizon}m.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def prepare_all(
    split_dir: Path = SPLIT_DIR,
    output_dir: Path = OUTPUT_DIR,
    report_dir: Path = REPORT_DIR,
    horizons: list[int] | None = None,
) -> list[dict]:
    return [
        prepare_horizon(horizon, split_dir, output_dir, report_dir)
        for horizon in (horizons or HORIZONS)
    ]


if __name__ == "__main__":
    for result in prepare_all():
        print(
            result["horizon_minutes"],
            {name: values["sequences"] for name, values in result["splits"].items()},
        )
