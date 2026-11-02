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
    cleaned, _ = deduplicate_with_quarantine(df, keys)
    return cleaned


def deduplicate_with_quarantine(df, keys):
    """Keep the most complete key row and return dropped duplicate rows."""
    df = df.copy()
    df["_completeness"] = df.notna().sum(axis=1)
    df = df.sort_values("_completeness", ascending=False, kind="stable")
    duplicate_mask = df.duplicated(subset=keys, keep="first")
    duplicates = df.loc[duplicate_mask].drop(columns="_completeness")
    cleaned = df.loc[~duplicate_mask].drop(columns="_completeness")
    return cleaned, duplicates


def numeric_columns(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fill_time_gaps(
    df,
    group_column,
    column,
    interval_minutes,
    max_gap_steps=2,
):
    """Forward-fill sensor gaps only while the last value is temporally recent."""
    df = df.sort_values([group_column, "_timestamp"], kind="stable").copy()
    grouped = df.groupby(group_column, sort=False)
    last_value_times = (
        df["_timestamp"]
        .where(df[column].notna())
        .groupby(df[group_column], sort=False)
        .ffill()
    )
    last_values = grouped[column].ffill()
    elapsed = df["_timestamp"] - last_value_times
    max_age = pd.Timedelta(minutes=interval_minutes * max_gap_steps)
    df[column] = last_values.where(elapsed.le(max_age))
    return df


def fill_segment_gaps(df, column, max_interpolation_gap=2):
    """Forward-fill short 30-minute gaps within one road segment."""
    return fill_time_gaps(
        df,
        group_column="road_id",
        column=column,
        interval_minutes=30,
        max_gap_steps=max_interpolation_gap,
    )


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
    original_null_counts = {
        column: int(count) for column, count in df.isna().sum().items()
    }
    source_rows = df.copy()

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
    invalid_row_reasons = pd.Series("", index=df.index, dtype="string")
    for column, mask in invalid_rules.items():
        mask = mask.fillna(False)
        invalid_counts[column] = int(mask.sum())
        if mask.any():
            invalid_row_reasons.loc[mask] = (
                invalid_row_reasons.loc[mask]
                .where(invalid_row_reasons.loc[mask].eq(""),
                       invalid_row_reasons.loc[mask] + ";")
                + f"impossible_{column}"
            )
        df.loc[mask, column] = np.nan
    invalid_quarantine = source_rows.loc[invalid_row_reasons.ne("")].copy()
    if not invalid_quarantine.empty:
        invalid_quarantine["quarantine_reason"] = invalid_row_reasons.loc[
            invalid_quarantine.index
        ]

    # Resolve duplicates using the most complete record.
    df, duplicate_quarantine = deduplicate_with_quarantine(
        df, ["road_id", "_timestamp"]
    )
    duplicate_quarantine = source_rows.loc[duplicate_quarantine.index].copy()
    duplicates_removed = len(duplicate_quarantine)

    # Targets stay observed or missing; only non-target sensor inputs are
    # forward-filled, so no later timestamp can impute an earlier value.
    for column in ["avg_speed", "occupancy"]:
        df = fill_segment_gaps(df, column)

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

    # Keep incomplete timestamp rows so later lag features retain their
    # 30-minute alignment. Model preparation excludes missing targets/features.
    required_model_fields = [
        "road_id", "traffic_volume", "avg_speed",
        "occupancy", "road_capacity", "congestion_level",
    ]
    unrecoverable = df[required_model_fields].isna().any(axis=1)
    unrecoverable_rows = int(unrecoverable.sum())

    df["date"] = df["_timestamp"].dt.strftime("%Y-%m-%d")
    df["time"] = df["_timestamp"].dt.strftime("%H:%M")

    output = df.drop(columns=["_timestamp"])
    output.to_csv(TRAFFIC_OUTPUT, index=False)

    duplicate_quarantine["quarantine_reason"] = "duplicate_road_timestamp"
    quarantine = pd.concat(
        [
            timestamp_quarantine.assign(quarantine_reason="invalid_timestamp"),
            invalid_quarantine,
            duplicate_quarantine,
        ],
        ignore_index=True,
        sort=False,
    )
    if not quarantine.empty:
        quarantine.to_csv(QUARANTINE / "traffic_quarantine.csv", index=False)

    return output, {
        "source_rows": original_rows,
        "clean_rows": len(output),
        "timestamp_rows_quarantined": len(timestamp_quarantine),
        "rows_with_unresolved_model_fields": unrecoverable_rows,
        "duplicate_rows_removed": duplicates_removed,
        "invalid_measurements_marked_missing": invalid_counts,
        "before_null_counts": original_null_counts,
        "remaining_nulls": output.isna().sum().to_dict(),
    }


def clean_weather():
    required = [
        "station_id", "date", "time", "temperature",
        "rainfall", "visibility", "weather_condition",
    ]

    df = read_csv_checked(WEATHER_FILE, required)
    original_rows = len(df)
    original_null_counts = {
        column: int(count) for column, count in df.isna().sum().items()
    }
    source_rows = df.copy()

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
    invalid_measurement = (
        (df["rainfall"] < 0)
        | (df["visibility"] < 0)
    ).fillna(False)
    invalid_quarantine = source_rows.loc[invalid_measurement].copy()
    if not invalid_quarantine.empty:
        invalid_quarantine["quarantine_reason"] = "impossible_weather_measurement"
    df.loc[invalid_measurement, ["rainfall", "visibility"]] = np.nan

    df, duplicate_quarantine = deduplicate_with_quarantine(
        df, ["station_id", "_timestamp"]
    )
    duplicate_quarantine = source_rows.loc[duplicate_quarantine.index].copy()
    duplicates_removed = len(duplicate_quarantine)

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

    df = df.sort_values(["station_id", "_timestamp"], kind="stable").copy()
    for column in ["temperature", "rainfall", "visibility", "weather_condition"]:
        df = fill_time_gaps(
            df,
            group_column="station_id",
            column=column,
            interval_minutes=60,
            max_gap_steps=2,
        )

    df["date"] = df["_timestamp"].dt.strftime("%Y-%m-%d")
    df["time"] = df["_timestamp"].dt.strftime("%H:%M")

    output = df.drop(columns=["_timestamp"])
    output.to_csv(WEATHER_OUTPUT, index=False)

    duplicate_quarantine["quarantine_reason"] = "duplicate_station_timestamp"
    quarantine = pd.concat(
        [
            timestamp_quarantine.assign(quarantine_reason="invalid_timestamp"),
            invalid_quarantine,
            duplicate_quarantine,
        ],
        ignore_index=True,
        sort=False,
    )
    if not quarantine.empty:
        quarantine.to_csv(QUARANTINE / "weather_quarantine.csv", index=False)

    return output, {
        "source_rows": original_rows,
        "clean_rows": len(output),
        "timestamp_rows_quarantined": len(timestamp_quarantine),
        "rows_with_unresolved_weather_fields": int(
            output[["temperature", "rainfall", "visibility", "weather_condition"]]
            .isna().any(axis=1).sum()
        ),
        "duplicate_rows_removed": duplicates_removed,
        "before_null_counts": original_null_counts,
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

    df, duplicate_quarantine = deduplicate_with_quarantine(df, ["date"])
    duplicates_removed = len(duplicate_quarantine)

    for column in ["public_holiday", "event_flag", "roadwork_flag"]:
        df[column] = df[column].fillna(0).astype(int)

    df["holiday_name"] = df["holiday_name"].fillna("No holiday")
    df["event_name"] = df["event_name"].fillna("No event")

    df.to_csv(CALENDAR_OUTPUT, index=False)

    duplicate_quarantine["quarantine_reason"] = "duplicate_calendar_date"
    quarantine = pd.concat(
        [
            quarantine.assign(quarantine_reason="invalid_calendar_date"),
            duplicate_quarantine,
        ],
        ignore_index=True,
        sort=False,
    )
    if not quarantine.empty:
        QUARANTINE.mkdir(parents=True, exist_ok=True)
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
        indicator="_weather_match",
    )

    unmatched_weather_rows = int(merged["_weather_match"].ne("both").sum())
    merged = merged.drop(columns=["_weather_match"])
    merged = merged.drop(columns=["station_id", "_weather_hour"])

    merged = merged.merge(
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
        indicator="_calendar_match",
    )
    unmatched_calendar_rows = int(merged["_calendar_match"].ne("both").sum())
    merged = merged.drop(columns=["_calendar_match"])

    for column in ["public_holiday", "event_flag", "roadwork_flag"]:
        merged[column] = merged[column].fillna(0).astype(int)

    merged["holiday_name"] = merged["holiday_name"].fillna("No holiday")
    merged["event_name"] = merged["event_name"].fillna("No event")

    merged = merged.drop(columns=["_timestamp"])
    merged.to_csv(MERGED_OUTPUT, index=False)
    merged.attrs["merge_quality"] = {
        "unmatched_weather_rows": unmatched_weather_rows,
        "unmatched_calendar_rows": unmatched_calendar_rows,
    }

    return merged


def run():
    INTERIM.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)

    traffic, traffic_report = clean_traffic()
    weather, weather_report = clean_weather()
    calendar, calendar_report = clean_calendar()

    merged = merge_datasets(traffic, weather, calendar)
    merge_quality = merged.attrs["merge_quality"]

    report = {
        "module": "M2 - Data Cleaning and Integration",
        "timezone": TZ,
        "traffic": traffic_report,
        "weather": weather_report,
        "calendar": calendar_report,
        "merge": {
            "merged_rows": len(merged),
            "merged_columns": len(merged.columns),
            **merge_quality,
            "weather_rows_with_missing_measurements": int(
                merged[["temperature", "rainfall", "visibility"]]
                .isna().any(axis=1).sum()
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
