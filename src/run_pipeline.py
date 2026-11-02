"""Reproduce the corrected FlowCast pipeline from immutable raw inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
STAGES = [
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m6_sequences",
    "m6_train",
    "m6_compare",
]
RAW_FILES = [
    "traffic_sensor_log.csv",
    "weather_observations.csv",
    "calendar_events.csv",
]


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _checkpoint(path: Path, stage: str, raw_hashes: dict[str, str], details: dict) -> None:
    _json_write(path, {
        "stage": stage,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": raw_hashes,
        "details": details,
    })


def _run_m1(raw_dir: Path, report_dir: Path) -> dict:
    from src.ingest import run_validation

    return run_validation(raw_dir=raw_dir, reports_dir=report_dir / "m1")


def _run_m2(raw_dir: Path, interim_dir: Path, report_dir: Path) -> dict:
    from src import clean

    interim_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = report_dir / "m2" / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    clean.TRAFFIC_FILE = raw_dir / "traffic_sensor_log.csv"
    clean.WEATHER_FILE = raw_dir / "weather_observations.csv"
    clean.CALENDAR_FILE = raw_dir / "calendar_events.csv"
    clean.TRAFFIC_OUTPUT = interim_dir / "traffic_clean.csv"
    clean.WEATHER_OUTPUT = interim_dir / "weather_clean.csv"
    clean.CALENDAR_OUTPUT = interim_dir / "calendar_clean.csv"
    clean.MERGED_OUTPUT = interim_dir / "flowcast_m2_merged.csv"
    clean.QUARANTINE = quarantine_dir

    traffic, traffic_report = clean.clean_traffic()
    weather, weather_report = clean.clean_weather()
    calendar, calendar_report = clean.clean_calendar()
    merged = clean.merge_datasets(traffic, weather, calendar)
    result = {
        "module": "M2 - Data Cleaning and Integration",
        "timezone": clean.TZ,
        "traffic": traffic_report,
        "weather": weather_report,
        "calendar": calendar_report,
        "merge": {
            "merged_rows": len(merged),
            "merged_columns": len(merged.columns),
            **merged.attrs["merge_quality"],
            "remaining_nulls": {
                column: int(count)
                for column, count in merged.isna().sum().items()
            },
        },
        "outputs": {
            "traffic": str(clean.TRAFFIC_OUTPUT.relative_to(ROOT)),
            "weather": str(clean.WEATHER_OUTPUT.relative_to(ROOT)),
            "calendar": str(clean.CALENDAR_OUTPUT.relative_to(ROOT)),
            "merged": str(clean.MERGED_OUTPUT.relative_to(ROOT)),
        },
    }
    _json_write(report_dir / "m2" / "m2_cleaning_report.json", result)
    return result


def _run_m3(merged_file: Path, processed_dir: Path, report_dir: Path) -> dict:
    from src import features

    features.INPUT = merged_file
    features.OUTPUT_DIR = processed_dir
    features.OUTPUT = processed_dir / "flowcast_m3_features.csv"
    features.REPORT = report_dir / "m3" / "m3_feature_report.json"
    features.run()
    return json.loads(features.REPORT.read_text(encoding="utf-8"))


def _run_m4(features_file: Path, report_dir: Path) -> dict:
    from src import eda

    eda.INPUT = features_file
    eda.FIGURES = report_dir / "m4" / "figures"
    eda.REPORT = report_dir / "m4" / "m4_eda_report.json"
    eda.QUALITY_REPORT = report_dir / "m4" / "m4_data_quality_report.md"
    eda.FIGURES.mkdir(parents=True, exist_ok=True)
    eda.REPORT.parent.mkdir(parents=True, exist_ok=True)
    eda.main()
    return json.loads(eda.REPORT.read_text(encoding="utf-8"))


def _run_m5_preparation(
    feature_file: Path,
    feature_config: Path,
    target_file: Path,
    split_dir: Path,
    preprocessed_dir: Path,
) -> dict:
    from src import feature_selection, preprocessing, splits, targets

    feature_selection.OUTPUT = feature_config
    feature_selection.main()
    targets.INPUT = feature_file
    targets.OUTPUT = target_file
    targets.main()
    splits.INPUT = target_file
    splits.OUTPUT_DIR = split_dir
    splits.main()
    preprocessing.FEATURE_CONFIG = feature_config
    preprocessing.SPLIT_DIR = split_dir
    preprocessing.OUTPUT_DIR = preprocessed_dir
    preprocessing.main()
    return json.loads(
        (preprocessed_dir / "preprocessing_report.json").read_text(
            encoding="utf-8"
        )
    )


def run_pipeline(
    run_id: str | None = None,
    raw_dir: Path = RAW_DIR,
    start_at: str = "m1",
) -> dict:
    """Run all corrected stages, stopping at errors and preserving prior runs."""
    run_id = run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id may contain only letters, numbers, hyphens, and underscores.")
    if start_at not in STAGES:
        raise ValueError(f"Unknown starting stage {start_at!r}; choose from {STAGES}.")

    raw_dir = raw_dir.resolve()
    missing_raw = [name for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing_raw:
        raise FileNotFoundError(f"Required immutable raw inputs are missing: {missing_raw}")
    raw_hashes = {name: _digest(raw_dir / name) for name in RAW_FILES}

    interim_dir = ROOT / "data/interim/corrected_v2" / run_id
    processed_dir = ROOT / "data/processed/corrected_v2" / run_id
    target_file = processed_dir / "flowcast_m5_targets.csv"
    split_dir = processed_dir / "m5_splits"
    preprocessed_dir = processed_dir / "m5_preprocessed"
    sequence_dir = processed_dir / "m6_sequences"
    m5_model_dir = ROOT / "models/m5/corrected_v2" / run_id
    m6_model_dir = ROOT / "models/m6/corrected_v2" / run_id
    report_dir = ROOT / "reports/corrected_v2" / run_id
    checkpoint_dir = report_dir / "checkpoints"
    run_manifest_path = report_dir / "run_manifest.json"
    latest_manifest = ROOT / "reports/corrected_v2/latest.json"
    if start_at == "m1" and run_manifest_path.exists():
        raise FileExistsError(
            f"Run {run_id} already exists. Choose a new run ID or resume "
            "it with --start-at."
        )
    run_manifest = {
        "project": "FlowCast",
        "pipeline_version": "corrected_v2",
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "raw_sha256": raw_hashes,
        "python": platform.python_version(),
        "stages": {},
        "paths": {
            "interim_dir": str(interim_dir.relative_to(ROOT)),
            "processed_dir": str(processed_dir.relative_to(ROOT)),
            "m5_model_dir": str(m5_model_dir.relative_to(ROOT)),
            "m6_model_dir": str(m6_model_dir.relative_to(ROOT)),
            "report_dir": str(report_dir.relative_to(ROOT)),
            "m5_split_dir": str(split_dir.relative_to(ROOT)),
            "m5_preprocessed_dir": str(preprocessed_dir.relative_to(ROOT)),
            "m6_sequence_dir": str(sequence_dir.relative_to(ROOT)),
        },
    }
    if start_at != "m1":
        if not run_manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot resume {run_id}: run manifest is missing at {run_manifest_path}"
            )
        previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_id") != run_id:
            raise ValueError("Existing manifest run ID does not match the requested ID.")
        if previous.get("pipeline_version") != "corrected_v2":
            raise ValueError("Cannot resume a run created by a different pipeline version.")
        if previous.get("raw_sha256") != raw_hashes:
            raise ValueError(
                "Raw input checksum differs from the original run; refusing to resume."
            )
        run_manifest.update(previous)
        run_manifest.pop("failure", None)
        run_manifest["status"] = "running"

    report_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    m5_model_dir.mkdir(parents=True, exist_ok=True)
    m6_model_dir.mkdir(parents=True, exist_ok=True)
    first_stage = STAGES.index(start_at)
    for pending_stage in STAGES[first_stage:]:
        run_manifest["stages"].pop(pending_stage, None)

    try:
        for stage in STAGES[first_stage:]:
            print(f"\n=== FlowCast {stage} ===", flush=True)
            if stage == "m1":
                details = _run_m1(raw_dir, report_dir)
            elif stage == "m2":
                details = _run_m2(raw_dir, interim_dir, report_dir)
            elif stage == "m3":
                details = _run_m3(
                    interim_dir / "flowcast_m2_merged.csv",
                    processed_dir,
                    report_dir,
                )
            elif stage == "m4":
                details = _run_m4(
                    processed_dir / "flowcast_m3_features.csv",
                    report_dir,
                )
            elif stage == "m5":
                preprocessing_report = _run_m5_preparation(
                    processed_dir / "flowcast_m3_features.csv",
                    report_dir / "m5_feature_selection.json",
                    target_file,
                    split_dir,
                    preprocessed_dir,
                )
                from src.m5_corrected_experiments import run_corrected_m5

                m5_results = run_corrected_m5(
                    split_dir=split_dir,
                    feature_config_path=report_dir / "m5_feature_selection.json",
                    model_dir=m5_model_dir,
                    report_dir=report_dir,
                )
                details = {
                    "preprocessing_combinations": len(preprocessing_report),
                    "selected_task_horizon_models": len(m5_results),
                    "test_rows": {
                        f"{item['task']}_{item['horizon_minutes']}m": item["test_rows"]
                        for item in m5_results
                    },
                }
            elif stage == "m6_sequences":
                from src.m6_corrected_sequences import prepare_all

                summaries = prepare_all(
                    split_dir=split_dir,
                    output_dir=sequence_dir,
                    report_dir=report_dir,
                )
                details = {
                    "horizons": {
                        str(item["horizon_minutes"]): {
                            name: count["sequences"]
                            for name, count in item["splits"].items()
                        }
                        for item in summaries
                    }
                }
            elif stage == "m6_train":
                from src.m6_corrected_train import train_all

                trained = train_all(
                    sequence_dir=sequence_dir,
                    model_dir=m6_model_dir,
                    report_dir=report_dir,
                )
                details = {
                    "model_count": len(trained),
                    "models": [
                        f"{item['model']}_{item['horizon_minutes']}m"
                        for item in trained
                    ],
                }
            else:
                from src.m6_corrected_comparison import compare_all

                comparisons = compare_all(
                    report_dir=report_dir,
                    sequence_dir=sequence_dir,
                    split_dir=split_dir,
                    m5_model_dir=m5_model_dir,
                    m6_model_dir=m6_model_dir,
                )
                details = {
                    "horizons": {
                        str(item["horizon_minutes"]): item["matched_test_records"]
                        for item in comparisons
                    },
                    "same_test_rows_per_horizon": True,
                }

            run_manifest["stages"][stage] = {
                "status": "done",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "details": details,
            }
            _checkpoint(
                checkpoint_dir / f"{stage}.json",
                stage,
                raw_hashes,
                details,
            )
            run_manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _json_write(run_manifest_path, run_manifest)

        run_manifest["status"] = "completed"
        run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _json_write(run_manifest_path, run_manifest)
        _json_write(latest_manifest, {
            "run_id": run_id,
            "manifest": str(run_manifest_path.relative_to(ROOT)),
            "paths": run_manifest["paths"],
            "completed_at_utc": run_manifest["completed_at_utc"],
        })
    except Exception as error:
        run_manifest["status"] = "failed"
        run_manifest["failure"] = {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _json_write(run_manifest_path, run_manifest)
        raise

    print(f"\nFlowCast corrected pipeline completed: {run_id}")
    print(f"Manifest: {run_manifest_path}")
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce corrected FlowCast outputs from immutable raw data."
    )
    parser.add_argument(
        "--run-id",
        help="Run identifier. Reuse an identifier only when resuming at a later stage.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing the three original source CSV files.",
    )
    parser.add_argument(
        "--start-at",
        choices=STAGES,
        default="m1",
        help="Resume at a stage using the same --run-id and unchanged raw files.",
    )
    args = parser.parse_args()
    run_pipeline(args.run_id, args.raw_dir, args.start_at)


if __name__ == "__main__":
    main()
