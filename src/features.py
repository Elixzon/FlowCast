"""FlowCast M3: chronological feature engineering."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

INPUT = ROOT / CONFIG["data"]["interim_dir"] / "flowcast_m2_merged.csv"
OUTPUT_DIR = ROOT / CONFIG["data"]["processed_dir"]
OUTPUT = OUTPUT_DIR / "flowcast_m3_features.csv"
REPORT = ROOT / "reports" / "m3_feature_report.json"

PEAK_HOURS = [(7, 10), (16, 20)]
LOW_VISIBILITY_METRES = 1000
EVENT_PROXIMITY_DAYS = 1

VEHICLE_TYPES = ["2W", "Car", "LCV", "HCV"]


def parse_vehicle_distribution(value):
    """Parse the vehicle_type_dist JSON field safely."""
    try:
        data = json.loads(value) if isinstance(value, str) else {}
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, TypeError):
        return {}


def add_vehicle_features(df):
    parsed = df["vehicle_type_dist"].apply(parse_vehicle_distribution)

    invalid_count = int(parsed.apply(lambda x: not bool(x)).sum())

    for vehicle_type in VEHICLE_TYPES:
        column = f"vehicle_share_{vehicle_type.lower()}"
        df[column] = parsed.apply(
            lambda item: pd.to_numeric(
                item.get(vehicle_type), errors="coerce"
            )
        )

    return df, invalid_count


def add_temporal_features(df):
    df["hour"] = df["_timestamp"].dt.hour
    df["day_of_week"] = df["_timestamp"].dt.dayofweek

    # Cyclical time encodings.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    hour = df["hour"]
    df["is_peak_period"] = (
        ((hour >= 7) & (hour < 10))
        | ((hour >= 16) & (hour < 20))
    ).astype(int)

    return df


def add_lag_and_rolling_features(df):
    """Build exact 30-minute historical features without crossing time gaps."""
    required = {"road_id", "_timestamp", "traffic_volume", "avg_speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for historical features: {sorted(missing)}")
    if df["_timestamp"].isna().any():
        raise ValueError("Historical features require valid timestamps.")
    if df.duplicated(["road_id", "_timestamp"]).any():
        raise ValueError("Duplicate road/timestamp keys prevent exact lag alignment.")
    if not df.sort_values(
        ["road_id", "_timestamp"], kind="stable"
    ).index.equals(df.index):
        raise ValueError("Rows must be sorted by road segment and timestamp.")

    keys = pd.MultiIndex.from_frame(df[["road_id", "_timestamp"]])
    current_times = df["_timestamp"]

    for column in ["traffic_volume", "avg_speed"]:
        historical = pd.Series(df[column].to_numpy(), index=keys)
        for lag in [1, 2, 48]:
            source_times = current_times - pd.Timedelta(minutes=30 * lag)
            assert source_times.lt(current_times).all()
            source_keys = pd.MultiIndex.from_arrays(
                [df["road_id"].to_numpy(), source_times],
                names=keys.names,
            )
            df[f"{column}_lag_{lag}"] = historical.reindex(source_keys).to_numpy()

        for window in [4, 8]:
            historical_values = []
            for lag in range(1, window + 1):
                source_times = current_times - pd.Timedelta(minutes=30 * lag)
                assert source_times.lt(current_times).all()
                source_keys = pd.MultiIndex.from_arrays(
                    [df["road_id"].to_numpy(), source_times],
                    names=keys.names,
                )
                historical_values.append(
                    historical.reindex(source_keys).to_numpy()
                )
            previous_values = pd.DataFrame(
                np.column_stack(historical_values),
                index=df.index,
            )
            complete_window = previous_values.notna().sum(axis=1).eq(window)
            df[f"{column}_rolling_mean_{window}"] = (
                previous_values.mean(axis=1).where(complete_window)
            )
            df[f"{column}_rolling_std_{window}"] = (
                previous_values.std(axis=1, ddof=1).where(complete_window)
            )

    return df


def add_capacity_features(df):
    capacity_per_window = df["road_capacity"] / 2

    df["volume_capacity_ratio"] = (
        df["traffic_volume"] / capacity_per_window
    )

    df["capacity_headroom"] = (
        capacity_per_window - df["traffic_volume"]
    )

    return df


def add_weather_features(df):
    df["rain_flag"] = (df["rainfall"] > 0).astype(int)

    df["low_visibility_flag"] = (
        df["visibility"] < LOW_VISIBILITY_METRES
    ).astype(int)

    df["temperature_band"] = pd.cut(
        df["temperature"],
        bins=[-np.inf, 15, 25, np.inf],
        labels=["Cool", "Moderate", "Warm"],
        right=False,
    ).astype("string")

    return df


def add_calendar_features(df):
    df["holiday_peak_interaction"] = (
        df["public_holiday"] * df["is_peak_period"]
    ).astype(int)

    # Event dates are taken from the supplied calendar data.
    # Scheduled event dates are treated as known calendar information.
    event_dates = set(
        pd.to_datetime(
            df.loc[df["event_flag"] == 1, "date"],
            errors="coerce",
        ).dropna().dt.normalize()
    )

    event_dates = {
        event_date + pd.Timedelta(days=offset)
        for event_date in event_dates
        for offset in range(
            -EVENT_PROXIMITY_DAYS,
            EVENT_PROXIMITY_DAYS + 1,
        )
    }

    current_dates = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.normalize()

    df["event_proximity_flag"] = (
        current_dates.isin(event_dates)
    ).astype(int)

    return df


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"M2 input not found: {INPUT}"
        )

    df = pd.read_csv(INPUT)
    source_rows = len(df)

    required = [
        "road_id", "date", "time", "traffic_volume",
        "avg_speed", "road_capacity", "vehicle_type_dist",
        "rainfall", "visibility", "temperature",
        "public_holiday", "event_flag",
    ]

    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"M2 dataset is missing columns: {missing_columns}"
        )

    # Construct a timestamp using the configured project timezone.
    timestamps = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            f"Found {int(timestamps.isna().sum())} invalid timestamps."
        )

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(
            CONFIG["project"]["timezone"],
            ambiguous="NaT",
            nonexistent="NaT",
        )
    else:
        timestamps = timestamps.dt.tz_convert(
            CONFIG["project"]["timezone"]
        )

    if timestamps.isna().any():
        raise ValueError("Invalid or ambiguous timestamps after timezone conversion.")

    df["_timestamp"] = timestamps

    # Strict segment/time ordering.
    df = df.sort_values(
        ["road_id", "_timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    if df.duplicated(["road_id", "_timestamp"]).any():
        raise ValueError("Duplicate road/timestamp keys found in M2 input.")

    # Vehicle distribution parsing.
    df, invalid_vehicle_json = add_vehicle_features(df)

    # Feature groups.
    df = add_temporal_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_capacity_features(df)
    df = add_weather_features(df)
    df = add_calendar_features(df)

    # Retain the timestamp as separate date/time fields.
    df["date"] = df["_timestamp"].dt.strftime("%Y-%m-%d")
    df["time"] = df["_timestamp"].dt.strftime("%H:%M")
    df = df.drop(columns=["_timestamp"])

    # Preserve every M2 record. Historical warm-up nulls are reported.
    feature_columns = [
        column for column in df.columns
        if column not in required
        and column not in [
            "road_name", "latitude", "longitude",
            "weather_station_id", "vehicle_count",
            "congestion_level", "occupancy", "travel_time",
            "accident_count", "signal_timing", "weather_condition",
            "rainfall", "visibility", "temperature",
            "public_holiday", "holiday_name", "event_flag",
            "event_name", "roadwork_flag", "road_capacity",
        ]
    ]

    lag_rolling_columns = [
        column for column in df.columns
        if "_lag_" in column or "_rolling_" in column
    ]

    report = {
        "module": "M3 - Feature Engineering",
        "input": str(INPUT.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "timezone": CONFIG["project"]["timezone"],
        "source_rows": source_rows,
        "output_rows": len(df),
        "road_segments": int(df["road_id"].nunique()),
        "feature_count": len(df.columns),
        "new_feature_columns": feature_columns,
        "vehicle_distribution_invalid_rows": invalid_vehicle_json,
        "defaults": {
            "peak_periods": PEAK_HOURS,
            "temperature_bands_celsius": {
                "Cool": "below 15",
                "Moderate": "15 to below 25",
                "Warm": "25 and above",
            },
            "event_proximity_days": EVENT_PROXIMITY_DAYS,
            "low_visibility_below_metres": LOW_VISIBILITY_METRES,
            "rain_flag": "rainfall > 0",
            "rolling_windows": [4, 8],
            "lag_windows": [1, 2, 48],
        },
        "leakage_controls": {
            "sort_order": ["road_id", "timestamp ascending"],
            "rolling_features_exclude_current_row": True,
            "lags_computed_within_road_segment": True,
            "future_observations_used_for_lags_or_rolling": False,
            "event_dates_use_scheduled_calendar_information": True,
        },
        "lag_rolling_null_counts": (
            df[lag_rolling_columns].isna().sum().to_dict()
        ),
        "total_nulls": int(df.isna().sum().sum()),
        "output_columns": df.columns.tolist(),
    }

    df.to_csv(OUTPUT, index=False)
    REPORT.write_text(json.dumps(report, indent=2, default=str))

    print("FlowCast M3 feature engineering completed.")
    print(f"Input rows: {source_rows:,}")
    print(f"Output rows: {len(df):,}")
    print(f"Road segments: {df['road_id'].nunique()}")
    print(f"Output columns: {len(df.columns)}")
    print(f"New feature columns: {len(feature_columns)}")
    print(f"Invalid vehicle JSON rows: {invalid_vehicle_json}")
    print(f"Total feature nulls: {df.isna().sum().sum()}")
    print(f"Feature dataset: {OUTPUT}")
    print(f"Feature report: {REPORT}")


if __name__ == "__main__":
    run()
