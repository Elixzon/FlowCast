"""M1: read-only validation of FlowCast source data."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
QUARANTINE = REPORTS / "quarantine"

EXPECTED = {
    "traffic_sensor_log.csv": [
        "road_id", "road_name", "latitude", "longitude",
        "weather_station_id", "date", "time", "traffic_volume",
        "vehicle_count", "vehicle_type_dist", "avg_speed", "occupancy",
        "congestion_level", "travel_time", "accident_count",
        "signal_timing", "road_capacity",
    ],
    "weather_observations.csv": [
        "station_id", "date", "time", "weather_condition",
        "temperature", "rainfall", "visibility",
    ],
    "calendar_events.csv": [
        "date", "public_holiday", "holiday_name", "event_flag",
        "event_name", "roadwork_flag",
    ],
}

NUMERIC_COLUMNS = {
    "traffic_sensor_log.csv": [
        "latitude", "longitude", "traffic_volume", "vehicle_count",
        "avg_speed", "occupancy", "travel_time", "accident_count",
        "signal_timing", "road_capacity",
    ],
    "weather_observations.csv": [
        "temperature", "rainfall", "visibility",
    ],
    "calendar_events.csv": [
        "public_holiday", "event_flag", "roadwork_flag",
    ],
}

WEATHER_LABELS = {
    "clear": "Clear",
    "cloudy": "Cloudy",
    "overcast": "Cloudy",
    "rain": "Rain",
    "rainy": "Rain",
    "fog": "Fog",
    "foggy": "Fog",
}


def add_issue(
    issues: list[dict],
    file_name: str,
    issue: str,
    count: int,
    severity: str = "warning",
    details: str | None = None,
) -> None:
    if count:
        issues.append({
            "file": file_name,
            "issue": issue,
            "count": int(count),
            "severity": severity,
            "details": details or "",
        })


def _validate_frame(
    df: pd.DataFrame, name: str
) -> tuple[dict, list[dict], pd.DataFrame]:
    expected = EXPECTED[name]
    missing_columns = [column for column in expected if column not in df]
    unexpected_columns = [column for column in df if column not in expected]
    issues: list[dict] = []
    add_issue(
        issues,
        name,
        "missing_required_columns",
        len(missing_columns),
        "error",
        ", ".join(missing_columns),
    )
    add_issue(
        issues,
        name,
        "unexpected_columns",
        len(unexpected_columns),
        "info",
        ", ".join(unexpected_columns),
    )

    reasons = pd.Series("", index=df.index, dtype="string")

    def mark(mask: pd.Series, reason: str) -> None:
        nonlocal reasons
        mask = mask.fillna(False)
        if mask.any():
            reasons.loc[mask] = reasons.loc[mask].where(
                reasons.loc[mask].eq(""),
                reasons.loc[mask] + ";",
            ) + reason
            add_issue(issues, name, reason, int(mask.sum()))

    null_counts = {column: int(count) for column, count in df.isna().sum().items()}
    optional_names = {
        "holiday_name": "public_holiday",
        "event_name": "event_flag",
    }
    for column, count in null_counts.items():
        if column in optional_names and optional_names[column] in df:
            active = pd.to_numeric(
                df[optional_names[column]], errors="coerce"
            ).eq(1)
            missing_required_name = df[column].isna() & active
            mark(missing_required_name, f"missing_{column}_for_flagged_date")
        else:
            add_issue(
                issues,
                name,
                "missing_values",
                count,
                details=column,
            )
            if count:
                mark(df[column].isna(), f"missing_{column}")

    date_format = "%d/%m/%Y" if name == "weather_observations.csv" else "%Y-%m-%d"
    if "date" in df:
        parsed_date = pd.to_datetime(
            df["date"].astype("string").str.strip(),
            format=date_format,
            errors="coerce",
        )
        mark(parsed_date.isna(), "invalid_date")
    if "time" in df:
        time_text = df["time"].astype("string").str.strip()
        valid_shape = time_text.str.fullmatch(r"\d{2}:\d{2}", na=False)
        parsed_time = pd.to_datetime(time_text, format="%H:%M", errors="coerce")
        mark(~(valid_shape & parsed_time.notna()), "invalid_time")

    numeric = {}
    for column in NUMERIC_COLUMNS[name]:
        if column not in df:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid_type = df[column].notna() & converted.isna()
        mark(invalid_type, f"invalid_numeric_{column}")
        numeric[column] = converted

    ranges = {
        "traffic_volume": lambda value: value < 0,
        "vehicle_count": lambda value: value < 0,
        "avg_speed": lambda value: (value < 0) | (value > 200),
        "occupancy": lambda value: (value < 0) | (value > 100),
        "travel_time": lambda value: value < 0,
        "accident_count": lambda value: value < 0,
        "signal_timing": lambda value: value < 0,
        "road_capacity": lambda value: value <= 0,
        "rainfall": lambda value: value < 0,
        "visibility": lambda value: value < 0,
        "public_holiday": lambda value: ~value.isin([0, 1]),
        "event_flag": lambda value: ~value.isin([0, 1]),
        "roadwork_flag": lambda value: ~value.isin([0, 1]),
    }
    for column, invalid in ranges.items():
        if column in numeric:
            mark(invalid(numeric[column]).fillna(False), f"impossible_{column}")

    duplicate_columns = {
        "traffic_sensor_log.csv": ["road_id", "date", "time"],
        "weather_observations.csv": ["station_id", "date", "time"],
        "calendar_events.csv": ["date"],
    }[name]
    if set(duplicate_columns).issubset(df.columns):
        duplicates = df.duplicated(duplicate_columns, keep=False)
        mark(duplicates, "duplicate_key")

    result = {
        "exists": True,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing_columns,
        "null_counts": null_counts,
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }

    if name == "traffic_sensor_log.csv":
        if "road_id" in df:
            result["road_segments"] = int(df["road_id"].nunique())
        if "date" in df:
            result["date_min"] = str(df["date"].min())
            result["date_max"] = str(df["date"].max())
    elif name == "weather_observations.csv":
        if "weather_condition" in df:
            normalized = (
                df["weather_condition"].astype("string").str.strip().str.lower()
            )
            labels = normalized.map(WEATHER_LABELS)
            result["weather_label_counts_normalized"] = {
                str(key): int(value)
                for key, value in labels.value_counts(dropna=False).items()
            }
            mark(labels.isna(), "unknown_weather_condition")
        if "station_id" in df:
            result["stations"] = int(df["station_id"].nunique())

    rejected = df.loc[reasons.ne("")].copy()
    if not rejected.empty:
        rejected.insert(0, "source_row", rejected.index.to_series().add(2).to_numpy())
        rejected["rejection_reasons"] = reasons.loc[rejected.index].to_numpy()
    result["quarantined_rows"] = int(len(rejected))
    return result, issues, rejected


def inspect_file(
    path: Path,
    issues: list[dict],
    quarantine_dir: Path = QUARANTINE,
) -> dict:
    name = path.name
    if name not in EXPECTED:
        raise ValueError(f"Unrecognized FlowCast source file: {name}")
    if not path.exists():
        add_issue(issues, name, "file_missing", 1, "error")
        return {"exists": False, "rows": 0, "columns": []}

    df = pd.read_csv(path)
    result, file_issues, rejected = _validate_frame(df, name)
    issues.extend(file_issues)
    if not rejected.empty:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        quarantine_path = quarantine_dir / f"{path.stem}_validation_quarantine.csv"
        rejected.to_csv(quarantine_path, index=False)
        result["quarantine_file"] = str(quarantine_path)
    return result


def run_validation(raw_dir: Path = RAW, reports_dir: Path = REPORTS) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict] = []
    files = {
        name: inspect_file(
            raw_dir / name,
            issues,
            reports_dir / "quarantine",
        )
        for name in EXPECTED
    }

    report = {
        "project": "FlowCast",
        "module": "M1 - Ingestion & Validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Raw inputs are read-only; validation never modifies them.",
        "files": files,
        "issues": issues,
        "record_policy": (
            "Rows with validation findings are copied to quarantine. Source files "
            "remain unchanged; M2 decides how recoverable values are handled."
        ),
    }
    (reports_dir / "m1_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(issues, columns=[
        "file", "issue", "count", "severity", "details",
    ]).to_csv(reports_dir / "m1_issue_register.csv", index=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FlowCast M1 validation.")
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    args = parser.parse_args()
    result = run_validation(args.raw_dir, args.reports_dir)
    print("FlowCast M1 validation completed.")
    for name, meta in result["files"].items():
        print(f"- {name}: {meta.get('rows', 0):,} rows; exists={meta.get('exists')}")
    print(f"Issues logged: {len(result['issues'])}")
    print(f"Report: {args.reports_dir / 'm1_validation_report.json'}")
