from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT = Path("data/processed/m5_splits/horizon_30.csv")
OUTPUT = Path("data/processed/m6_sequences")
REPORT = Path("reports/m6_sequence_report.json")
SEQ_LEN = 48
TARGET = "traffic_volume_t_plus_30"

CATEGORICAL = ["road_id", "weather_condition", "temperature_band"]

NUMERIC = [
    "latitude", "longitude", "road_capacity", "temperature",
    "rainfall", "visibility", "public_holiday", "event_flag",
    "roadwork_flag", "hour", "day_of_week", "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos", "is_weekend",
    "is_peak_period", "rain_flag", "low_visibility_flag",
    "holiday_peak_interaction", "event_proximity_flag",
    "traffic_volume_lag_1", "traffic_volume_lag_2",
    "traffic_volume_lag_48", "traffic_volume_rolling_mean_4",
    "traffic_volume_rolling_std_4", "traffic_volume_rolling_mean_8",
    "traffic_volume_rolling_std_8", "avg_speed_lag_1",
    "avg_speed_lag_2", "avg_speed_lag_48",
    "avg_speed_rolling_mean_4", "avg_speed_rolling_std_4",
    "avg_speed_rolling_mean_8", "avg_speed_rolling_std_8",
]

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    required = ["road_id", "timestamp", "split", TARGET, *CATEGORICAL, *NUMERIC]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.duplicated(["road_id", "timestamp"]).any():
        raise ValueError("Duplicate road/timestamp keys detected")

    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").astype(str)

    train_mask = df["split"].eq("train")
    train_fit_mask = train_mask & df[NUMERIC].notna().all(axis=1)

    scaler = StandardScaler()
    scaler.fit(df.loc[train_fit_mask, NUMERIC])
    df[NUMERIC] = scaler.transform(df[NUMERIC])

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    categorical_encoder.fit(df.loc[train_mask, CATEGORICAL])
    encoded = pd.DataFrame(
        categorical_encoder.transform(df[CATEGORICAL]),
        columns=categorical_encoder.get_feature_names_out(CATEGORICAL),
        index=df.index,
    )
    feature_frame = pd.concat(
        [df[NUMERIC].astype(np.float32), encoded], axis=1
    )
    feature_names = feature_frame.columns.tolist()

    joblib.dump(
        {
            "scaler": scaler,
            "categorical_encoder": categorical_encoder,
            "numeric_columns": NUMERIC,
            "categorical_columns": CATEGORICAL,
            "feature_names": feature_names,
            "fitted_on": "training partition only",
        },
        OUTPUT / "sequence_preprocessing.joblib",
    )

    X_parts = {s: [] for s in ["train", "validation", "test"]}
    y_parts = {s: [] for s in X_parts}
    road_parts = {s: [] for s in X_parts}
    timestamp_parts = {s: [] for s in X_parts}
    gap_rejections = {s: 0 for s in X_parts}

    for _, road in df.groupby("road_id", sort=False):
        road = road.sort_values("timestamp")

        for split_name in X_parts:
            part = road[road["split"].eq(split_name)]
            if len(part) < SEQ_LEN:
                continue

            values = feature_frame.loc[part.index].to_numpy(dtype=np.float32)
            targets = part[TARGET].to_numpy(dtype=np.float32)
            timestamps = part["timestamp"].to_numpy()

            for end in range(SEQ_LEN - 1, len(part)):
                start = end - SEQ_LEN + 1
                window = timestamps[start:end + 1]

                if not np.all(np.diff(window) == np.timedelta64(30, "m")):
                    gap_rejections[split_name] += 1
                    continue

                sequence = values[start:end + 1]

                # Preserve warm-up nulls; exclude incomplete sequences.
                if not np.isfinite(sequence).all():
                    continue

                if not np.isfinite(targets[end]):
                    continue

                X_parts[split_name].append(sequence)
                y_parts[split_name].append(targets[end])
                road_parts[split_name].append(str(part["road_id"].iloc[end]))
                timestamp_parts[split_name].append(str(part["timestamp"].iloc[end]))

    summary = {
        "sequence_length": SEQ_LEN,
        "target": TARGET,
        "preprocessing_fit": "scaler and categorical encoder fit on training rows only",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "road_segments": int(df["road_id"].nunique()),
        "source_rows": int(len(df)),
        "gap_rejections": gap_rejections,
        "split_counts": {},
    }

    for split_name in X_parts:
        if not X_parts[split_name]:
            raise ValueError(f"No sequences generated for {split_name}")

        X = np.stack(X_parts[split_name]).astype(np.float32)
        y = np.asarray(y_parts[split_name], dtype=np.float32)

        np.savez_compressed(
            OUTPUT / f"{split_name}.npz",
            X=X,
            y=y,
            road_id=np.asarray(road_parts[split_name]),
            timestamp=np.asarray(timestamp_parts[split_name]),
        )

        summary["split_counts"][split_name] = {
            "sequences": int(len(X)),
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "non_finite_X": int((~np.isfinite(X)).sum()),
            "non_finite_y": int((~np.isfinite(y)).sum()),
        }

    REPORT.write_text(json.dumps(summary, indent=2))
    print("M6 sequence preparation complete.")
    print(json.dumps(summary["split_counts"], indent=2))
    print("Gap rejections:", gap_rejections)
    print("Feature count:", len(feature_names))
    print("Report:", REPORT)

if __name__ == "__main__":
    main()
