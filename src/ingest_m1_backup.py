"""M1: FlowCast raw-data ingestion and validation.

This module reads the supplied raw files without modifying or silently dropping
records. It writes a machine-readable validation report and CSV issue register.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

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

def add_issue(issues, file_name, issue, count, severity="warning", details=None):
    if count:
        issues.append({
            "file": file_name, "issue": issue, "count": int(count),
            "severity": severity, "details": details or ""
        })

def inspect_file(path: Path, issues: list) -> dict:
    name = path.name
    if not path.exists():
        add_issue(issues, name, "file_missing", 1, "error")
        return {"exists": False, "rows": 0, "columns": []}

    df = pd.read_csv(path)
    expected = EXPECTED[name]
    missing_cols = [c for c in expected if c not in df.columns]
    unexpected_cols = [c for c in df.columns if c not in expected]
    add_issue(issues, name, "missing_required_columns", len(missing_cols), "error",
              ", ".join(missing_cols))
    add_issue(issues, name, "unexpected_columns", len(unexpected_cols), "info",
              ", ".join(unexpected_cols))

    result = {
        "exists": True, "rows": len(df), "columns": list(df.columns),
        "missing_required_columns": missing_cols,
        "null_counts": {k: int(v) for k, v in df.isna().sum().items()},
    }

    if name == "traffic_sensor_log.csv":
        if {"road_id", "date", "time"}.issubset(df.columns):
            add_issue(issues, name, "rows_in_duplicate_segment_time_groups",
                      df.duplicated(["road_id", "date", "time"], keep=False).sum(),
                      "warning", "Key: road_id + date + time; M2 resolves duplicates.")
        checks = [
            ("missing_traffic_volume", df["traffic_volume"].isna().sum()),
            ("missing_avg_speed", df["avg_speed"].isna().sum()),
            ("missing_occupancy", df["occupancy"].isna().sum()),
            ("missing_congestion_level", df["congestion_level"].isna().sum()),
            ("negative_traffic_volume", (df["traffic_volume"] < 0).sum()),
            ("avg_speed_above_200", (df["avg_speed"] > 200).sum()),
            ("occupancy_above_100", (df["occupancy"] > 100).sum()),
        ]
        for issue, count in checks:
            add_issue(issues, name, issue, count)
        result["road_segments"] = int(df["road_id"].nunique())
        result["date_min"] = str(df["date"].min())
        result["date_max"] = str(df["date"].max())

    elif name == "weather_observations.csv":
        add_issue(issues, name, "missing_temperature", df["temperature"].isna().sum())
        add_issue(issues, name, "missing_visibility", df["visibility"].isna().sum())
        labels = df["weather_condition"].astype(str).str.strip().str.lower().value_counts()
        result["weather_label_counts_raw"] = {str(k): int(v) for k, v in labels.items()}
        result["stations"] = int(df["station_id"].nunique())

    elif name == "calendar_events.csv":
        add_issue(issues, name, "duplicate_calendar_dates",
                  df.duplicated(["date"], keep=False).sum())

    return result

def run_validation(raw_dir: Path = RAW) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    issues = []
    files = {}
    for name in EXPECTED:
        files[name] = inspect_file(raw_dir / name, issues)

    report = {
        "project": "FlowCast",
        "module": "M1 - Ingestion & Validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Raw inputs are read-only; no records are removed in M1.",
        "files": files,
        "issues": issues,
        "record_policy": (
            "M1 reports validation findings. Structural failures are flagged; "
            "duplicate and value-quality remediation belongs to M2. No silent drops."
        ),
    }
    (REPORTS / "m1_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(issues, columns=[
        "file", "issue", "count", "severity", "details"
    ]).to_csv(REPORTS / "m1_issue_register.csv", index=False)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FlowCast M1 validation.")
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    args = parser.parse_args()
    result = run_validation(args.raw_dir)
    print("FlowCast M1 validation completed.")
    for name, meta in result["files"].items():
        print(f"- {name}: {meta.get('rows', 0):,} rows; exists={meta.get('exists')}")
    print(f"Issues logged: {len(result['issues'])}")
    print(f"Report: {REPORTS / 'm1_validation_report.json'}")
