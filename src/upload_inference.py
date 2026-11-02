"""Isolated processing and inference for dashboard-uploaded source data."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.ingest import EXPECTED, run_validation
from src.run_pipeline import _run_m2, _run_m3

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = ROOT / "data/interim/upload_runs"
RAW_DIR = ROOT / "data/raw"
REGRESSION_TASKS = {"traffic_volume", "travel_time"}


def process_uploaded_sources(uploaded_files: Mapping[str, bytes]) -> dict:
    """Validate, clean, join, and feature-engineer uploaded raw source files."""
    unknown = sorted(set(uploaded_files) - set(EXPECTED))
    if unknown:
        raise ValueError(f"Unsupported FlowCast upload filenames: {unknown}")
    if "traffic_sensor_log.csv" not in uploaded_files:
        raise ValueError("A traffic_sensor_log.csv upload is required.")
    if any(not isinstance(content, bytes) or not content for content in uploaded_files.values()):
        raise ValueError("Uploaded CSV contents must be non-empty bytes.")

    digests = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(uploaded_files.items())
    }
    digest_prefix = hashlib.sha256(
        json.dumps(digests, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_id = (
        f"upload_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_"
        f"{digest_prefix}"
    )
    run_dir = UPLOAD_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"Upload run already exists: {run_id}")
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)

    fallback_files = []
    for name in EXPECTED:
        destination = raw_dir / name
        if name in uploaded_files:
            destination.write_bytes(uploaded_files[name])
        else:
            shutil.copyfile(RAW_DIR / name, destination)
            fallback_files.append(name)

    report_dir = run_dir / "reports"
    validation = run_validation(raw_dir=raw_dir, reports_dir=report_dir / "m1")
    schema_errors = {
        name: metadata["missing_required_columns"]
        for name, metadata in validation["files"].items()
        if metadata["missing_required_columns"]
    }
    if schema_errors:
        raise ValueError(f"Uploaded sources failed required-column checks: {schema_errors}")

    interim_dir = run_dir / "interim"
    processed_dir = run_dir / "processed"
    m2_report = _run_m2(raw_dir, interim_dir, report_dir)
    m3_report = _run_m3(
        interim_dir / "flowcast_m2_merged.csv",
        processed_dir,
        report_dir,
    )
    feature_path = processed_dir / "flowcast_m3_features.csv"
    if not feature_path.is_file():
        raise FileNotFoundError(f"Upload feature processing output is missing: {feature_path}")

    manifest = {
        "run_id": run_id,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "uploaded_files": sorted(uploaded_files),
        "fallback_files_from_project_raw_data": fallback_files,
        "uploaded_sha256": digests,
        "validation_report": str(
            (report_dir / "m1" / "m1_validation_report.json").relative_to(ROOT)
        ),
        "m2_report": str(
            (report_dir / "m2" / "m2_cleaning_report.json").relative_to(ROOT)
        ),
        "m3_report": str(
            (report_dir / "m3" / "m3_feature_report.json").relative_to(ROOT)
        ),
        "feature_file": str(feature_path.relative_to(ROOT)),
    }
    (run_dir / "upload_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "validation": validation,
        "m2": m2_report,
        "m3": m3_report,
        "feature_path": feature_path,
    }


def predict_latest_uploaded_rows(
    bundle: dict,
    feature_frame: pd.DataFrame,
    task: str,
    horizon: int,
    model_run_id: str,
) -> tuple[pd.DataFrame, int]:
    """Predict the latest feature-complete uploaded row for each road segment."""
    features = list(bundle["features"])
    missing = sorted((set(features) | {"road_id"}) - set(feature_frame))
    if missing:
        raise ValueError(f"Processed upload lacks prediction columns: {missing}")
    frame = feature_frame.copy()
    frame["road_id"] = frame["road_id"].astype(str)
    if "timestamp" not in frame:
        missing_timestamp = sorted({"date", "time"} - set(frame))
        if missing_timestamp:
            raise ValueError(
                f"Processed upload lacks timestamp columns: {missing_timestamp}"
            )
        frame["timestamp"] = pd.to_datetime(
            frame["date"].astype(str) + " " + frame["time"].astype(str),
            errors="raise",
        )
    else:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    complete_rows = frame.loc[frame[features].notna().all(axis=1)]
    latest = (
        complete_rows.sort_values(
            ["road_id", "timestamp"], kind="stable"
        )
        .groupby("road_id", sort=False)
        .tail(1)
        .copy()
    )
    skipped = int(frame["road_id"].nunique() - latest["road_id"].nunique())
    if latest.empty:
        raise ValueError(
            "No road has a feature-complete record; upload matching "
            "weather/calendar coverage and complete traffic observations."
        )

    transformed = bundle["preprocessor"].transform(latest[features])
    model = bundle["model"]
    predictions = np.asarray(model.predict(transformed)).reshape(-1)
    output = pd.DataFrame({
        "road_id": latest["road_id"].to_numpy(),
        "as_of": latest["timestamp"].astype(str).to_numpy(),
        "forecast_for": (
            latest["timestamp"] + pd.Timedelta(minutes=horizon)
        ).astype(str).to_numpy(),
        "model_run_id": model_run_id,
        "prediction": predictions,
    })

    if task in REGRESSION_TASKS:
        predictions = np.maximum(predictions, 0)
        quantile = bundle.get("conformal_quantile")
        if quantile is None or not np.isfinite(quantile):
            raise ValueError(
                f"No finite split-conformal quantile is saved for {task}/{horizon}m."
            )
        output["prediction"] = predictions
        output["lower_90"] = np.maximum(predictions - float(quantile), 0)
        output["upper_90"] = predictions + float(quantile)
    elif task == "congestion_level":
        encoder = bundle.get("label_encoder")
        if encoder is None:
            raise ValueError("Congestion model artifact has no saved label encoder.")
        output["prediction"] = encoder.inverse_transform(
            predictions.astype(int)
        )
        if hasattr(model, "predict_proba"):
            output["confidence"] = model.predict_proba(transformed).max(axis=1)
    elif task == "accident_risk":
        threshold = bundle.get("validation_threshold")
        if threshold is None or not np.isfinite(float(threshold)):
            raise ValueError("Accident-risk model has no validation threshold.")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed)
            classes = np.asarray(model.classes_)
            positive_columns = np.flatnonzero(classes.astype(str) == "1")
            if len(positive_columns) != 1:
                raise ValueError("Accident-risk model has no unique positive class.")
            scores = probabilities[:, positive_columns[0]]
            output["confidence"] = probabilities.max(axis=1)
            output["risk_score_type"] = "positive_class_probability"
        elif hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(transformed)).reshape(-1)
            output["risk_score_type"] = "uncalibrated_decision_score"
        else:
            raise TypeError("Accident-risk model exposes no ranking score method.")
        output["prediction"] = (scores >= float(threshold)).astype(int)
        output["risk_score"] = scores
    else:
        raise ValueError(f"Unsupported FlowCast prediction task: {task}")

    output["model_artifact"] = str(bundle["artifact"].relative_to(ROOT))
    return output, skipped
