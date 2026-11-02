"""Matched validation/test comparison for corrected M5 and M6 volume models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from src.m5_corrected_experiments import (
    HORIZONS,
    _regression_metrics,
    _split_conformal_quantile,
)
from src.m6_corrected_train import (
    MODEL_TYPES,
    _validation_selection_calibration_masks,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/processed/m5_splits_corrected_v2"
SEQUENCE_DIR = ROOT / "data/processed/m6_sequences_corrected_v2"
M5_MODEL_DIR = ROOT / "models/m5/corrected_v2"
M6_MODEL_DIR = ROOT / "models/m6/corrected_v2"
REPORT_DIR = ROOT / "reports/corrected_v2"
CARD_DIR = REPORT_DIR / "model_cards"
INTERVAL_COVERAGE = 0.90
FEATURE_ENCODER_PREFIXES = {
    "road_id_": "road_id",
    "weather_condition_": "weather_condition",
    "temperature_band_": "temperature_band",
    "congestion_level_": "congestion_level",
}


def _raw_feature_names(encoded_names: list[str]) -> set[str]:
    raw_names = set()
    for name in encoded_names:
        for prefix, column in FEATURE_ENCODER_PREFIXES.items():
            if name.startswith(prefix):
                raw_names.add(column)
                break
        else:
            raw_names.add(name)
    return raw_names


def _sequence_keys(data: np.lib.npyio.NpzFile, horizon: int) -> pd.DataFrame:
    target = np.asarray(data["y"], dtype=float)
    keys = pd.DataFrame({
        "sequence_index": np.arange(len(target)),
        "road_id": data["road_id"].astype(str),
        "timestamp": pd.to_datetime(data["timestamp"], errors="raise"),
        "sequence_target": target,
    })
    if keys.duplicated(["road_id", "timestamp"]).any():
        raise ValueError(f"Duplicate sequence keys at horizon {horizon}.")
    target_timestamps = pd.to_datetime(data["target_timestamp"], errors="raise")
    expected_target_timestamps = keys["timestamp"] + pd.Timedelta(minutes=horizon)
    if not np.array_equal(
        target_timestamps.to_numpy(), expected_target_timestamps.to_numpy()
    ):
        raise ValueError(
            f"Sequence target timestamps do not align with the {horizon}-minute horizon."
        )
    return keys


def _match_sequences_to_rows(
    keys: pd.DataFrame,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    split: str,
) -> pd.DataFrame:
    row_columns = list(dict.fromkeys(
        ["road_id", "timestamp", target] + feature_columns
    ))
    rows = frame.loc[frame["split"].eq(split), row_columns].dropna(
        subset=feature_columns + [target]
    )
    matched = keys.merge(
        rows,
        on=["road_id", "timestamp"],
        how="inner",
        validate="one_to_one",
    )
    if not np.allclose(
        matched["sequence_target"].to_numpy(dtype=float),
        matched[target].to_numpy(dtype=float),
        rtol=0,
        atol=1e-5,
    ):
        raise ValueError(f"Matched sequence and M5 targets differ for {split}.")
    return matched.sort_values("sequence_index", kind="stable").reset_index(drop=True)


def _predict_deep(
    network: tf.keras.Model,
    sequences: np.ndarray,
    metadata: dict,
) -> np.ndarray:
    scaled = network.predict(
        sequences.astype(np.float32), batch_size=256, verbose=0
    ).reshape(-1)
    scaling = metadata["target_scaling"]
    return np.maximum(
        scaled * scaling["training_standard_deviation"]
        + scaling["training_mean"],
        0,
    )


def _interval_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    calibration_residuals: np.ndarray,
) -> dict:
    quantile = _split_conformal_quantile(
        calibration_residuals, INTERVAL_COVERAGE
    )
    lower = np.maximum(prediction - quantile, 0)
    upper = prediction + quantile
    return {
        "method": "split conformal absolute residual",
        "nominal_coverage": INTERVAL_COVERAGE,
        "calibration_rows": int(len(calibration_residuals)),
        "quantile": quantile,
        "observed_test_coverage": float(
            np.mean((target >= lower) & (target <= upper))
        ),
        "test_mean_interval_width": float(np.mean(upper - lower)),
        "limitations": (
            "The final chronological validation block is reserved for calibration "
            "after model selection. The finite-sample guarantee requires "
            "exchangeability; serial dependence and distribution shift mean "
            "nominal coverage is not guaranteed for this time series."
        ),
    }


def compare_horizon(
    horizon: int,
    sequence_dir: Path = SEQUENCE_DIR,
    split_dir: Path = SPLIT_DIR,
    m5_model_dir: Path = M5_MODEL_DIR,
    m6_model_dir: Path = M6_MODEL_DIR,
    report_dir: Path = REPORT_DIR,
) -> dict:
    validation_file = sequence_dir / f"validation_{horizon}m.npz"
    test_file = sequence_dir / f"test_{horizon}m.npz"
    split_file = split_dir / f"horizon_{horizon}.csv"
    if not validation_file.exists() or not test_file.exists() or not split_file.exists():
        raise FileNotFoundError(f"Matched validation/test data missing for {horizon}m.")

    classical = joblib.load(m5_model_dir / f"traffic_volume_{horizon}m.joblib")
    if not isinstance(classical, dict) or not {
        "model", "preprocessor", "features"
    }.issubset(classical):
        raise ValueError("Corrected M5 volume artifact has an invalid bundle format.")
    candidate_path = m5_model_dir / (
        f"traffic_volume_{horizon}m_validation_candidates.joblib"
    )
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Validation-fitted classical candidates missing: {candidate_path}"
        )
    candidate_bundle = joblib.load(candidate_path)
    feature_columns = list(classical["features"])
    if list(candidate_bundle["features"]) != feature_columns:
        raise ValueError("M5 selected and validation-candidate features differ.")

    validation_data = np.load(validation_file)
    test_data = np.load(test_file)
    validation_X = validation_data["X"].astype(np.float32)
    test_X = test_data["X"].astype(np.float32)
    validation_keys = _sequence_keys(validation_data, horizon)
    test_keys = _sequence_keys(test_data, horizon)
    validation_selection_mask, validation_calibration_mask = (
        _validation_selection_calibration_masks(validation_data["timestamp"])
    )
    validation_selection_keys = validation_keys.loc[
        validation_selection_mask
    ].copy()
    validation_calibration_keys = validation_keys.loc[
        validation_calibration_mask
    ].copy()

    frame = pd.read_csv(split_file)
    frame["road_id"] = frame["road_id"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    target = f"traffic_volume_t_plus_{horizon}"

    m6_feature_names = json.loads(
        (m6_model_dir / f"lstm_{horizon}m_metadata.json").read_text(
            encoding="utf-8"
        )
    )["hyperparameters"]["feature_names"]
    m5_raw_features = set(feature_columns)
    m6_raw_features = _raw_feature_names(m6_feature_names)
    if m5_raw_features != m6_raw_features:
        raise ValueError(
            "M5/M6 issue-time feature parity failed: "
            f"M5-only={sorted(m5_raw_features - m6_raw_features)}, "
            f"M6-only={sorted(m6_raw_features - m5_raw_features)}"
        )

    validation_selection = _match_sequences_to_rows(
        validation_selection_keys, frame, feature_columns, target, "validation"
    )
    validation_calibration = _match_sequences_to_rows(
        validation_calibration_keys, frame, feature_columns, target, "validation"
    )
    matched_test = _match_sequences_to_rows(
        test_keys, frame, feature_columns, target, "test"
    )
    if any(part.empty for part in (
        validation_selection, validation_calibration, matched_test
    )):
        raise ValueError(f"Empty common M5/M6 partition at horizon {horizon}.")

    validation_selection_indices = validation_selection[
        "sequence_index"
    ].to_numpy(dtype=int)
    validation_calibration_indices = validation_calibration[
        "sequence_index"
    ].to_numpy(dtype=int)
    test_indices = matched_test["sequence_index"].to_numpy(dtype=int)
    y_validation_selection = validation_selection[target].to_numpy(dtype=float)
    y_validation_calibration = validation_calibration[target].to_numpy(dtype=float)
    y_test = matched_test[target].to_numpy(dtype=float)

    classical_X_selection = candidate_bundle["preprocessor"].transform(
        validation_selection[feature_columns]
    )
    classical_validation_rmse = {}
    for name, model in candidate_bundle["models"].items():
        prediction = np.maximum(
            model.predict(classical_X_selection).reshape(-1), 0
        )
        classical_validation_rmse[name] = float(np.sqrt(mean_squared_error(
            y_validation_selection, prediction
        )))
    selected_classical_name = min(
        classical_validation_rmse, key=classical_validation_rmse.get
    )
    selected_classical_model = candidate_bundle["models"][selected_classical_name]

    network_models = {}
    metadata_by_model = {}
    deep_validation_rmse = {}
    deep_validation_rmse_all = {}
    deep_validation_predictions = {}
    for model_type in MODEL_TYPES:
        stem = f"{model_type.lower().replace(' ', '_')}_{horizon}m"
        model_path = m6_model_dir / f"{stem}.keras"
        metadata_path = m6_model_dir / f"{stem}_metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Corrected deep model missing: {stem}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        network = tf.keras.models.load_model(model_path)
        network_models[model_type] = network
        metadata_by_model[model_type] = metadata
        prediction_selection = _predict_deep(
            network,
            validation_X[validation_selection_indices],
            metadata,
        )
        deep_validation_rmse_all[model_type] = float(np.sqrt(mean_squared_error(
            y_validation_selection, prediction_selection
        )))
        if model_type in {"LSTM", "GRU"}:
            deep_validation_rmse[model_type] = deep_validation_rmse_all[model_type]
        deep_validation_predictions[model_type] = prediction_selection
    selected_deep = min(deep_validation_rmse, key=deep_validation_rmse.get)

    classical_X_test = candidate_bundle["preprocessor"].transform(
        matched_test[feature_columns]
    )
    classical_test_prediction = np.maximum(
        selected_classical_model.predict(classical_X_test).reshape(-1), 0
    )
    classical_X_calibration = candidate_bundle["preprocessor"].transform(
        validation_calibration[feature_columns]
    )
    classical_calibration_prediction = np.maximum(
        selected_classical_model.predict(classical_X_calibration).reshape(-1), 0
    )
    classical_interval = _interval_metrics(
        y_test,
        classical_test_prediction,
        np.abs(y_validation_calibration - classical_calibration_prediction),
    )
    classical_metrics = _regression_metrics(y_test, classical_test_prediction)
    classical_metrics["prediction_interval_coverage_90"] = (
        classical_interval["observed_test_coverage"]
    )
    results = [{
        "model": f"Classical {selected_classical_name}",
        "family": "classical",
        "selected_classical_model": True,
        "selected_operational_deep_model": False,
        "historical_only_bidirectional": False,
        "test_records": len(y_test),
        "validation_selection_rmse": classical_validation_rmse[
            selected_classical_name
        ],
        "validation_calibration_records": len(validation_calibration),
        "prediction_interval": classical_interval,
        **classical_metrics,
    }]

    test_predictions = {}
    calibration_predictions = {}
    for model_type, network in network_models.items():
        metadata = metadata_by_model[model_type]
        prediction = _predict_deep(network, test_X[test_indices], metadata)
        calibration_prediction = _predict_deep(
            network,
            validation_X[validation_calibration_indices],
            metadata,
        )
        test_predictions[model_type] = prediction
        calibration_predictions[model_type] = calibration_prediction
        interval = _interval_metrics(
            y_test,
            prediction,
            np.abs(y_validation_calibration - calibration_prediction),
        )
        metrics = _regression_metrics(y_test, prediction)
        metrics["prediction_interval_coverage_90"] = (
            interval["observed_test_coverage"]
        )
        results.append({
            "model": model_type,
            "family": "deep_learning",
            "selected_classical_model": False,
            "selected_operational_deep_model": model_type == selected_deep,
            "historical_only_bidirectional": model_type == "Bidirectional LSTM",
            "test_records": len(y_test),
            "validation_selection_rmse": deep_validation_rmse_all[model_type],
            "validation_calibration_records": len(validation_calibration),
            "prediction_interval": interval,
            **metrics,
        })

    for row in results:
        row["difference_vs_classical"] = {
            metric: float(row[metric] - classical_metrics[metric])
            for metric in ["MAE", "RMSE", "MAPE_percent", "R2"]
        }

    card_dir = report_dir / "model_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    validation_selection_period = {
        "start": str(validation_selection["timestamp"].min()),
        "end": str(validation_selection["timestamp"].max()),
        "matched_records": int(len(validation_selection)),
    }
    calibration_period = {
        "start": str(validation_calibration["timestamp"].min()),
        "end": str(validation_calibration["timestamp"].max()),
        "matched_records": int(len(validation_calibration)),
    }
    for model_type in MODEL_TYPES:
        metadata = metadata_by_model[model_type]
        row = next(item for item in results if item["model"] == model_type)
        interval = row["prediction_interval"]
        card = {
            "model_name": model_type,
            "model_type": (
                "Bidirectional recurrent network"
                if model_type.startswith("Bidirectional")
                else "Recurrent neural network"
            ),
            "model_usage": (
                "Historical-analysis comparison only; not selected for operational forecasts."
                if model_type == "Bidirectional LSTM"
                else "Operational unidirectional historical-window forecast."
            ),
            "target": metadata["target"],
            "forecast_horizon_minutes": horizon,
            "training_period": metadata["training_period"],
            "validation_selection_period": validation_selection_period,
            "validation_calibration_period": calibration_period,
            "test_period": {
                "start": str(matched_test["timestamp"].min()),
                "end": str(matched_test["timestamp"].max()),
            },
            "feature_list": metadata["hyperparameters"]["feature_names"],
            "feature_count": metadata["hyperparameters"]["feature_count"],
            "feature_parity": (
                "The raw M5 and M6 issue-time predictor columns are identical. "
                "Each model family uses its own training-fitted encoding and "
                "scaling; M6 additionally receives the prior 47 contiguous "
                "half-hour feature vectors."
            ),
            "preprocessing": (
                "Training-only median imputation, scaling, and one-hot encoding; "
                "48-step exact 30-minute sequences are road/split isolated. "
                "M5/M6 comparison rows require all issue-time M5 features."
            ),
            "hyperparameters": metadata["hyperparameters"],
            "validation_metrics": {
                "selection_rmse_on_matched_records": row[
                    "validation_selection_rmse"
                ],
                "calibration_records": len(validation_calibration),
            },
            "test_metrics": _regression_metrics(y_test, test_predictions[model_type]),
            "prediction_interval": interval,
            "training_duration_seconds": metadata["training_duration_seconds"],
            "training_history_artifact": metadata["history_artifact"],
            "training_curve_artifact": metadata["loss_figure"],
            "test_records": len(y_test),
            "selected_operational_deep_model": model_type == selected_deep,
            "model_artifact_path": str(
                (m6_model_dir / (
                    f"{model_type.lower().replace(' ', '_')}_{horizon}m.keras"
                )).relative_to(ROOT)
            ),
            "test_used_for_selection": False,
            "random_seed": metadata["hyperparameters"]["seed"],
            "software_versions": metadata["software_versions"],
        }
        (card_dir / (
            f"{model_type.lower().replace(' ', '_')}_{horizon}m_model_card.json"
        )).write_text(json.dumps(card, indent=2), encoding="utf-8")
        metadata["matched_validation_selection_rmse"] = (
            row["validation_selection_rmse"]
        )
        metadata["matched_validation_calibration_records"] = len(
            validation_calibration
        )
        metadata["test_metrics_matched_m5"] = card["test_metrics"]
        metadata["test_interval_coverage"] = interval["observed_test_coverage"]
        metadata["test_records_matched_m5"] = len(y_test)
        metadata["selected_operational_model"] = model_type == selected_deep
        metadata["test_used_for_selection"] = False
        (m6_model_dir / (
            f"{model_type.lower().replace(' ', '_')}_{horizon}m_metadata.json"
        )).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    table = pd.DataFrame(results)
    table_path = report_dir / f"m6_corrected_head_to_head_{horizon}m.csv"
    table.to_csv(table_path, index=False)
    return {
        "horizon_minutes": horizon,
        "target": target,
        "matched_test_records": len(y_test),
        "same_records_for_all_models": True,
        "matched_validation_selection_records": len(validation_selection),
        "matched_validation_calibration_records": len(validation_calibration),
        "same_validation_records_for_all_models": True,
        "m5_m6_issue_time_feature_parity": True,
        "classical_validation_rmse": classical_validation_rmse,
        "deep_validation_rmse": deep_validation_rmse,
        "all_deep_validation_rmse": deep_validation_rmse_all,
        "selected_classical_model": results[0]["model"],
        "selected_deep_model_by_validation_rmse": selected_deep,
        "models": results,
        "comparison_table": str(table_path.relative_to(ROOT)),
    }


def compare_all(
    horizons: list[int] | None = None,
    report_dir: Path = REPORT_DIR,
    sequence_dir: Path = SEQUENCE_DIR,
    split_dir: Path = SPLIT_DIR,
    m5_model_dir: Path = M5_MODEL_DIR,
    m6_model_dir: Path = M6_MODEL_DIR,
) -> list[dict]:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "model_cards").mkdir(parents=True, exist_ok=True)
    comparisons = [
        compare_horizon(
            horizon,
            sequence_dir,
            split_dir,
            m5_model_dir,
            m6_model_dir,
            report_dir,
        )
        for horizon in (horizons or HORIZONS)
    ]
    rows = []
    for report in comparisons:
        for model in report["models"]:
            rows.append({"horizon_minutes": report["horizon_minutes"], **model})
    pd.DataFrame(rows).to_csv(
        report_dir / "m6_corrected_head_to_head.csv", index=False
    )
    (report_dir / "m6_corrected_head_to_head.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    lines = [
        "# Corrected M6 / M5 Matched Comparison",
        "",
        "Classical and deep models are selected on the same matched chronological",
        "validation-selection records, calibrated on a later validation block, and",
        "evaluated once on exactly the same held-out test records. Issue-time",
        "features are identical; M6 additionally uses the prior 47 contiguous",
        "half-hour feature vectors. All preprocessing is fitted on training data.",
        "",
        "| Horizon | Model | Validation RMSE | Test rows | Test RMSE | Test MAPE (%) | 90% interval coverage | Δ RMSE vs classical |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in comparisons:
        classical_rmse = report["models"][0]["RMSE"]
        for model in report["models"]:
            lines.append(
                f"| {report['horizon_minutes']} | {model['model']} "
                f"| {model['validation_selection_rmse']:.4f} "
                f"| {model['test_records']} | {model['RMSE']:.4f} "
                f"| {model['MAPE_percent']:.4f} "
                f"| {model['prediction_interval_coverage_90']:.4f} "
                f"| {model['RMSE'] - classical_rmse:+.4f} |"
            )
    (report_dir / "m6_corrected_head_to_head.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return comparisons


if __name__ == "__main__":
    for item in compare_all():
        print(item["horizon_minutes"], item["matched_test_records"])
