"""FlowCast M2: deterministic data cleaning and integration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

RAW = ROOT / CONFIG["data"]["raw_dir"]
INTERIM = ROOT / CONFIG["data"]["interim_dir"]
REPORTS = ROOT / "reports"
QUARANTINE = REPORTS / "quarantine"

TRAFFIC_FILE = RAW / "traffic_sensor_log.csv"
WEATHER_FILE = RAW / "weather_observations.csv"
CALENDAR_FILE = RAW / "calendar_events.csv"

TRAFFIC_OUTPUT = INTERIM / "traffic_clean.csv"
WEATHER_OUTPUT = INTERIM / "weather_clean.csv"
CALENDAR_OUTPUT = INTERIM / "calendar_clean.csv"
MERGED_OUTPUT = INTERIM / "flowcast_m2_merged.csv"

TZ = CONFIG["project"]["timezone"]


def read_csv_checked(path, required_columns):
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")

    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))

    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    return df


def make_timestamp(date_series, time_series, dayfirst=False):
    values = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    timestamps = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst)

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(
            TZ, ambiguous="NaT", nonexistent="NaT"
        )
    else:
        timestamps = timestamps.dt.tz_convert(TZ)

    return timestamps


def most_complete_deduplicate(df, keys):
    """Keep the most complete record for each key; retain first on ties."""
    df = df.copy()
    df["_completeness"] = df.notna().sum(axis=1)
    df = df.sort_values("_completeness", ascending=False, kind="stable")
    df = df.drop_duplicates(subset=keys, keep="first")
    return df.drop(columns="_completeness")


def numeric_columns(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fill_segment_gaps(df, column, max_interpolation_gap=2):
    """Interpolate short gaps, then use segment/time-of-day medians."""
    df = df.sort_values(["road_id", "_timestamp"]).copy()

    df[column] = df.groupby("road_id", sort=False)[column].transform(
        lambda s: s.interpolate(
            method="linear",
            limit=max_interpolation_gap,
            limit_area="inside",
        )
    )

    time_median = df.groupby(["road_id", "time"])[column].transform("median")
    df[column] = df[column].fillna(time_median)

    segment_median = df.groupby("road_id")[column].transform("median")
    df[column] = df[column].fillna(segment_median)

    return df


def clean_traffic():
    required = [
        "road_id", "road_name", "latitude", "longitude",
        "weather_station_id", "date", "time", "traffic_volume",
        "vehicle_count", "vehicle_type_dist", "avg_speed", "occupancy",
        "congestion_level", "travel_time", "accident_count",
        "signal_timing", "road_capacity",
    ]

    df = read_csv_checked(TRAFFIC_FILE, required)
    original_rows = len(df)

    df["_timestamp"] = make_timestamp(df["date"], df["time"])

    # Invalid timestamps cannot be aligned safely.
    bad_timestamp = df["_timestamp"].isna()
    timestamp_quarantine = df.loc[bad_timestamp].copy()
    df = df.loc[~bad_timestamp].copy()

    # Convert measurement fields to numeric.
    numeric = [
        "latitude", "longitude", "traffic_volume", "vehicle_count",
        "avg_speed", "occupancy", "travel_time", "accident_count",
        "signal_timing", "road_capacity",
    ]
    df = numeric_columns(df, numeric)

    # Invalid physical measurements are marked missing for controlled recovery.
    invalid_rules = {
        "traffic_volume": df["traffic_volume"] < 0,
        "vehicle_count": df["vehicle_count"] < 0,
        "avg_speed": (df["avg_speed"] < 0) | (df["avg_speed"] > 200),
        "occupancy": (df["occupancy"] < 0) | (df["occupancy"] > 100),
        "travel_time": df["travel_time"] < 0,
        "accident_count": df["accident_count"] < 0,
        "signal_timing": df["signal_timing"] < 0,
        "road_capacity": df["road_capacity"] <= 0,
    }

    invalid_counts = {}
    for column, mask in invalid_rules.items():
        invalid_counts[column] = int(mask.sum())
        df.loc[mask, column] = np.nan

    # Resolve duplicates using the most complete record.
    before_dedup = len(df)
    df = most_complete_deduplicate(
        df, ["road_id", "_timestamp"]
    )
    duplicates_removed = before_dedup - len(df)

    # Recover short gaps and then segment/time-of-day gaps.
    for column in ["traffic_volume", "avg_speed", "occupancy"]:
        df = fill_segment_gaps(df, column)

    # These fields are not interpolated as continuous measurements.
    for column in ["vehicle_count", "travel_time", "accident_count", "signal_timing"]:
        df[column] = df[column].fillna(
            df.groupby("road_id")[column].transform("median")
        )

    # Parse and normalise congestion labels.
    congestion_map = {
        "free-flow": "Free-flow",
        "free flow": "Free-flow",
        "moderate": "Moderate",
        "heavy": "Heavy",
        "severe": "Severe",
    }

    df["congestion_level"] = (
        df["congestion_level"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(congestion_map)
    )

    # Derive congestion where it is missing, using the dictionary thresholds.
    ratio = df["traffic_volume"] / (df["road_capacity"] / 2)
    derived = pd.Series(index=df.index, dtype="object")
    derived.loc[ratio < 0.50] = "Free-flow"
    derived.loc[(ratio >= 0.50) & (ratio < 0.80)] = "Moderate"
    derived.loc[(ratio >= 0.80) & (ratio < 1.00)] = "Heavy"
    derived.loc[ratio >= 1.00] = "Severe"
    df["congestion_level"] = df["congestion_level"].fillna(derived)

    # Rows with unrecoverable required modelling fields are quarantined.
    required_model_fields = [
        "road_id", "traffic_volume", "avg_speed",
        "occupancy", "road_capacity", "congestion_level",
    ]
    unrecoverable = df[required_model_fields].isna().any(axis=1)
    missing_quarantine = df.loc[unrecoverable].copy()
    df = df.loc[~unrecoverable].copy()

    df["date"] = df["_timestamp"].dt.strftime("%Y-%m-%d")
    df["time"] = df["_timestamp"].dt.strftime("%H:%M")

    output = df.drop(columns=["_timestamp"])
    output.to_csv(TRAFFIC_OUTPUT, index=False)

    quarantine = pd.concat(
        [timestamp_quarantine, missing_quarantine],
        ignore_index=True,
    )
    if not quarantine.empty:
        quarantine.to_csv(QUARANTINE / "traffic_quarantine.csv", index=False)

    return output, {
        "source_rows": original_rows,
        "clean_rows": len(output),
        "timestamp_rows_quarantined": len(timestamp_quarantine),
        "unrecoverable_rows_quarantined": len(missing_quarantine),
        "duplicate_rows_removed": duplicates_removed,
        "invalid_measurements_marked_missing": invalid_counts,
        "remaining_nulls": output.isna().sum().to_dict(),
    }


def clean_weather():
    required = [
        "station_id", "date", "time", "temperature",
        "rainfall", "visibility", "weather_condition",
    ]

    df = read_csv_checked(WEATHER_FILE, required)
    original_rows = len(df)

    # Weather dates are DD/MM/YYYY in the supplied data dictionary.
    df["_timestamp"] = make_timestamp(
        df["date"], df["time"], dayfirst=True
    )

    bad_timestamp = df["_timestamp"].isna()
    timestamp_quarantine = df.loc[bad_timestamp].copy()
    df = df.loc[~bad_timestamp].copy()

    df = numeric_columns(
        df, ["temperature", "rainfall", "visibility"]
    )

    # Negative rainfall or visibility is invalid.
    df.loc[df["rainfall"] < 0, "rainfall"] = np.nan
    df.loc[df["visibility"] < 0, "visibility"] = np.nan

    before_dedup = len(df)
    df = most_complete_deduplicate(
        df, ["station_id", "_timestamp"]
    )
    duplicates_removed = before_dedup - len(df)

    # Normalise known variants to the controlled vocabulary.
    weather_map = {
        "clear": "Clear",
        "cloudy": "Cloudy",
        "overcast": "Cloudy",
        "rain": "Rain",
        "rainy": "Rain",
        "fog": "Fog",
        "foggy": "Fog",
    }

    df["weather_condition"] = (
        df["weather_condition"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(weather_map)
    )

    df = df.sort_values(["station_id", "_timestamp"]).copy()

    for column in ["temperature", "rainfall", "visibility"]:
        df[column] = df.groupby("station_id", sort=False)[column].transform(
            lambda s: s.interpolate(
                method="linear", limit=2, limit_area="inside"
            )
        )
        df[column] = df[column].fillna(
            df.groupby("station_id")[column].transform("median")
        )

    df["weather_condition"] = df.groupby("station_id")[
        "weather_condition"
    ].transform(lambda s: s.ffill().bfill())

    unrecoverable = df[
        ["station_id", "temperature", "rainfall", "visibility", "weather_condition"]
    ].isna().any(axis=1)

    missing_quarantine = df.loc[unrecoverable].copy()
    df = df.loc[~unrecoverable].copy()

    df["date"] = df["_timestamp"].dt.strftime("%Y-%m-%d")
    df["time"] = df["_timestamp"].dt.strftime("%H:%M")

    output = df.drop(columns=["_timestamp"])
    output.to_csv(WEATHER_OUTPUT, index=False)

    quarantine = pd.concat(
        [timestamp_quarantine, missing_quarantine],
        ignore_index=True,
    )
    if not quarantine.empty:
        quarantine.to_csv(QUARANTINE / "weather_quarantine.csv", index=False)

    return output, {
        "source_rows": original_rows,
        "clean_rows": len(output),
        "timestamp_rows_quarantined": len(timestamp_quarantine),
        "unrecoverable_rows_quarantined": len(missing_quarantine),
        "duplicate_rows_removed": duplicates_removed,
        "remaining_nulls": output.isna().sum().to_dict(),
        "normalised_weather_labels": sorted(
            output["weather_condition"].dropna().unique().tolist()
        ),
    }


def clean_calendar():
    required = [
        "date", "public_holiday", "holiday_name",
        "event_flag", "event_name", "roadwork_flag",
    ]

    df = read_csv_checked(CALENDAR_FILE, required)
    original_rows = len(df)

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    bad_date = df["date"].isna()
    quarantine = df.loc[bad_date].copy()
    df = df.loc[~bad_date].copy()

    for column in ["public_holiday", "event_flag", "roadwork_flag"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before_dedup = len(df)
    df = most_complete_deduplicate(df, ["date"])
    duplicates_removed = before_dedup - len(df)

    for column in ["public_holiday", "event_flag", "roadwork_flag"]:
        df[column] = df[column].fillna(0).astype(int)

    df["holiday_name"] = df["holiday_name"].fillna("No holiday")
    df["event_name"] = df["event_name"].fillna("No event")

    df.to_csv(CALENDAR_OUTPUT, index=False)

    if not quarantine.empty:
        quarantine.to_csv(QUARANTINE / "calendar_quarantine.csv", index=False)

    return df, {
        "source_rows": original_rows,
        "clean_rows": len(df),
        "invalid_date_rows_quarantined": len(quarantine),
        "duplicate_rows_removed": duplicates_removed,
        "remaining_nulls": df.isna().sum().to_dict(),
    }

def merge_datasets(traffic, weather, calendar):
    traffic = traffic.copy()
    weather = weather.copy()

    traffic["_timestamp"] = make_timestamp(traffic["date"], traffic["time"])
    weather["_timestamp"] = make_timestamp(weather["date"], weather["time"])

    # Align hourly weather with the corresponding traffic hour.
    traffic["_weather_hour"] = traffic["_timestamp"].dt.floor("h")
    weather["_weather_hour"] = weather["_timestamp"].dt.floor("h")

    weather_for_merge = weather.drop(
        columns=["date", "time", "_timestamp"]
    )

    merged = traffic.merge(
        weather_for_merge,
        left_on=["weather_station_id", "_weather_hour"],
        right_on=["station_id", "_weather_hour"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_weather"),
    )

    merged = merged.drop(columns=["station_id", "_weather_hour"])

    merged = merged.merge(
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
    )

    for column in ["public_holiday", "event_flag", "roadwork_flag"]:
        merged[column] = merged[column].fillna(0).astype(int)

    merged["holiday_name"] = merged["holiday_name"].fillna("No holiday")
    merged["event_name"] = merged["event_name"].fillna("No event")

    merged = merged.drop(columns=["_timestamp"])
    merged.to_csv(MERGED_OUTPUT, index=False)

    return merged


def run():
    INTERIM.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)

    traffic, traffic_report = clean_traffic()
    weather, weather_report = clean_weather()
    calendar, calendar_report = clean_calendar()

    merged = merge_datasets(traffic, weather, calendar)

    report = {
        "module": "M2 - Data Cleaning and Integration",
        "timezone": TZ,
        "traffic": traffic_report,
        "weather": weather_report,
        "calendar": calendar_report,
        "merge": {
            "merged_rows": len(merged),
            "merged_columns": len(merged.columns),
            "unmatched_weather_rows": int(
                merged["temperature"].isna().sum()
            ),
            "unmatched_calendar_rows": int(
                merged["public_holiday"].isna().sum()
            ),
            "remaining_nulls": merged.isna().sum().to_dict(),
        },
        "outputs": {
            "traffic": str(TRAFFIC_OUTPUT.relative_to(ROOT)),
            "weather": str(WEATHER_OUTPUT.relative_to(ROOT)),
            "calendar": str(CALENDAR_OUTPUT.relative_to(ROOT)),
            "merged": str(MERGED_OUTPUT.relative_to(ROOT)),
        },
    }

    report_path = REPORTS / "m2_cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    print("FlowCast M2 cleaning and integration completed.")
    print(f"Traffic: {len(traffic):,} clean rows")
    print(f"Weather: {len(weather):,} clean rows")
    print(f"Calendar: {len(calendar):,} clean rows")
    print(f"Merged: {len(merged):,} rows")
    print(f"Report: {report_path}")
    print(f"Merged data: {MERGED_OUTPUT}")


if __name__ == "__main__":
    run()
