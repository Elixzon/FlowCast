"""Read-only statistical and domain-quality analysis for FlowCast sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TABLES = {
    "traffic_sensor_log.csv": {
        "keys": ["road_id", "date", "time"],
        "group": "road_id",
        "measurements": [
            "traffic_volume",
            "vehicle_count",
            "avg_speed",
            "occupancy",
            "travel_time",
            "accident_count",
            "signal_timing",
            "road_capacity",
        ],
        "invalid": {
            "traffic_volume": lambda values: values < 0,
            "vehicle_count": lambda values: values < 0,
            "avg_speed": lambda values: (values < 0) | (values > 200),
            "occupancy": lambda values: (values < 0) | (values > 100),
            "travel_time": lambda values: values < 0,
            "accident_count": lambda values: values < 0,
            "signal_timing": lambda values: values < 0,
            "road_capacity": lambda values: values <= 0,
        },
    },
    "weather_observations.csv": {
        "keys": ["station_id", "date", "time"],
        "group": "station_id",
        "measurements": ["temperature", "rainfall", "visibility"],
        "invalid": {
            "rainfall": lambda values: values < 0,
            "visibility": lambda values: values < 0,
        },
    },
    "calendar_events.csv": {
        "keys": ["date"],
        "group": None,
        "measurements": ["public_holiday", "event_flag", "roadwork_flag"],
        "invalid": {},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iqr_counts(
    frame: pd.DataFrame,
    column: str,
    group_column: str | None,
) -> dict[str, Any]:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite_values = values.where(np.isfinite(values))
    valid = finite_values.dropna()
    if valid.empty:
        return {
            "valid_rows": 0,
            "non_finite_numeric_rows": int((~np.isfinite(values)).sum()),
            "iqr_outlier_rows": 0,
            "zscore_outlier_rows": 0,
            "groups_with_outliers": 0,
            "group_fences": [],
        }

    groups = (
        frame.groupby(group_column, dropna=False, sort=False).groups.items()
        if group_column is not None
        else [("whole_table", frame.index)]
    )
    outlier_mask = pd.Series(False, index=frame.index)
    zscore_mask = pd.Series(False, index=frame.index)
    group_count = 0
    group_fences = []
    for group_value, indices in groups:
        group_values = finite_values.loc[indices].dropna()
        if len(group_values) < 4:
            continue
        lower_quartile = float(group_values.quantile(0.25))
        upper_quartile = float(group_values.quantile(0.75))
        spread = upper_quartile - lower_quartile
        lower_fence = lower_quartile - 1.5 * spread
        upper_fence = upper_quartile + 1.5 * spread
        finite_group_values = finite_values.loc[indices]
        flagged = finite_group_values.lt(lower_fence) | finite_group_values.gt(upper_fence)
        outlier_mask.loc[indices] = flagged.fillna(False)
        mean = float(group_values.mean())
        standard_deviation = float(group_values.std(ddof=0))
        zscore_flagged = (
            (finite_group_values - mean).abs().gt(3 * standard_deviation)
            if standard_deviation > 0
            else pd.Series(False, index=indices)
        )
        zscore_mask.loc[indices] = zscore_flagged.fillna(False)
        if flagged.any():
            group_count += 1
        group_fences.append({
            "group": str(group_value),
            "valid_rows": int(len(group_values)),
            "q1": lower_quartile,
            "q3": upper_quartile,
            "lower_fence": float(lower_fence),
            "upper_fence": float(upper_fence),
            "iqr_outlier_rows": int(flagged.sum()),
            "zscore_outlier_rows": int(zscore_flagged.sum()),
        })

    return {
        "valid_rows": int(len(valid)),
        "non_finite_numeric_rows": int((~np.isfinite(values)).sum()),
        "iqr_outlier_rows": int(outlier_mask.sum()),
        "zscore_outlier_rows": int(zscore_mask.sum()),
        "groups_with_outliers": group_count,
        "grouping": group_column or "whole_table",
        "zscore_threshold": 3.0,
        "group_fences": group_fences,
    }


def analyze_sources(raw_dir: Path) -> dict[str, Any]:
    """Summarize missingness, duplicate keys, domain errors, and IQR flags."""
    tables: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for filename, specification in TABLES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}")
        frame = pd.read_csv(path)
        missing_keys = sorted(set(specification["keys"]) - set(frame.columns))
        if missing_keys:
            raise ValueError(f"{filename} is missing quality-analysis keys: {missing_keys}")
        hashes[filename] = _sha256(path)
        group_column = specification["group"]
        missingness = {
            column: {
                "count": int(count),
                "fraction": float(count / len(frame)) if len(frame) else 0.0,
            }
            for column, count in frame.isna().sum().items()
        }
        measurements = {}
        domain_invalid = {}
        for column in specification["measurements"]:
            if column not in frame:
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            measurements[column] = _iqr_counts(frame, column, group_column)
            invalid_rule = specification["invalid"].get(column)
            domain_invalid[column] = (
                int(
                    (
                        invalid_rule(numeric)
                        & np.isfinite(numeric)
                    ).fillna(False).sum()
                )
                if invalid_rule is not None
                else 0
            )
        duplicate_mask = frame.duplicated(specification["keys"], keep=False)
        tables[filename] = {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "duplicate_key_rows": int(duplicate_mask.sum()),
            "duplicate_key_rows_to_remove": int(
                frame.duplicated(specification["keys"], keep="first").sum()
            ),
            "missingness": missingness,
            "domain_invalid_counts": domain_invalid,
            "statistical_iqr_flags": measurements,
            "statistical_outlier_policy": (
                "Review-only flags; no statistical outlier is automatically "
                "deleted, capped, or imputed."
            ),
        }
    return {
        "analysis": "Read-only source data quality and statistical outlier review",
        "iqr_rule": "Per road or weather station, 1.5 times the interquartile range",
        "source_sha256": hashes,
        "tables": tables,
        "limitations": [
            "IQR flags identify unusual values, not proven sensor faults.",
            "Domain rules identify physically impossible values; valid extremes are retained.",
            "No rows are modified by this analysis.",
        ],
    }


def write_quality_report(
    raw_dir: Path,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    """Write machine-readable and human-readable read-only quality reports."""
    result = analyze_sources(raw_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# FlowCast Source Data Quality Report",
        "",
        "This report reads raw inputs without changing them. IQR flags are review",
        "candidates, not automatic deletion criteria; domain-invalid measurements",
        "remain subject to the existing cleaning/quarantine policy.",
        "",
        f"Statistical rules: {result['iqr_rule']} and absolute z-score > 3.",
        "",
        "| Source | Rows | Duplicate-key rows | Extra duplicate rows |",
        "|---|---:|---:|---:|",
    ]
    for filename, details in result["tables"].items():
        lines.append(
            f"| `{filename}` | {details['rows']:,} | "
            f"{details['duplicate_key_rows']:,} | "
            f"{details['duplicate_key_rows_to_remove']:,} |"
        )
    lines.extend(["", "## Measurement review", ""])
    lines.extend([
        "| Source | Field | Missing | Domain-invalid | IQR flags | abs(z) > 3 flags |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for filename, details in result["tables"].items():
        for column, iqr_details in details["statistical_iqr_flags"].items():
            missing = details["missingness"].get(column, {}).get("count", 0)
            invalid = details["domain_invalid_counts"].get(column, 0)
            lines.append(
                f"| `{filename}` | `{column}` | {missing:,} | {invalid:,} | "
                f"{iqr_details['iqr_outlier_rows']:,} | "
                f"{iqr_details['zscore_outlier_rows']:,} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- IQR and z-score candidates are reported, not automatically removed: traffic peaks,",
        "  rare incidents, and unusual weather can be legitimate observations.",
        "- Domain-invalid values are separately counted using the project's",
        "  documented physical bounds.",
        "- Missingness and duplicate-key counts describe the raw source files;",
        "  downstream M1/M2 reports document validation and cleaning outcomes.",
        "",
        "Raw-file SHA-256 values:",
        "",
    ])
    lines.extend(
        f"- `{filename}`: `{digest}`"
        for filename, digest in result["source_sha256"].items()
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["json_report"] = str(json_path)
    result["markdown_report"] = str(markdown_path)
    return result
