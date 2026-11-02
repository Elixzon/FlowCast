import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import m6_corrected_sequences, run_pipeline
from src import feature_selection, preprocessing, upload_inference
from src.m5_corrected_experiments import (
    CONGESTION_ORDER,
    NumpyGradientRegressor,
    StableLinearRegressor,
    _build_preprocessor,
    _binary_label_text,
    _split_conformal_quantile,
    _validation_selection_calibration_masks,
    run_corrected_m5,
    _select_accident_threshold,
)
from src.m6_corrected_sequences import (
    CATEGORICAL,
    MINMAX_NUMERIC,
    NUMERIC,
    STANDARD_NUMERIC,
)
from dashboard.app import build_selected_range_report


def test_stable_linear_regression_handles_rank_deficient_features():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    X = np.column_stack([X, X[:, 0], np.ones(len(X))])
    y = 2 * X[:, 0] + 5

    model = StableLinearRegressor().fit(X, y)

    assert model.rank_ < X.shape[1] + 1
    assert np.mean(np.abs(model.predict(X) - y)) < 1e-8


def test_numpy_gradient_regressor_uses_validation_and_predicts_original_scale():
    X_train = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y_train = 3 * X_train[:, 0] + 7
    X_validation = np.array([[0.5], [0.8]])
    y_validation = 3 * X_validation[:, 0] + 7

    model = NumpyGradientRegressor(
        learning_rate=0.05, max_epochs=300, patience=20
    ).fit(X_train, y_train, (X_validation, y_validation))

    assert model.epochs_completed_ <= 300
    assert np.mean(np.abs(model.predict(X_validation) - y_validation)) < 1.0
    assert np.isfinite(model.best_validation_mse_scaled_)


def test_feature_preprocessor_handles_validation_categories_not_seen_in_train():
    features = ["road_id", "temperature"]
    frame = pd.DataFrame({
        "road_id": ["R1", "R2", "new-road"],
        "temperature": [10.0, 20.0, 30.0],
    })
    preprocessor = _build_preprocessor(features)
    train_matrix = preprocessor.fit_transform(frame.iloc[:2])
    validation_matrix = preprocessor.transform(frame.iloc[2:])

    assert train_matrix.shape[0] == 2
    assert validation_matrix.shape[0] == 1
    assert np.isfinite(validation_matrix).all()


def test_accident_threshold_is_selected_from_validation_scores():
    threshold = _select_accident_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.3, 0.7, 0.9]),
    )

    assert 0.3 <= threshold <= 0.9


def test_split_conformal_quantile_uses_finite_sample_order_statistic():
    quantile = _split_conformal_quantile(
        np.arange(1, 10, dtype=float),
        coverage=0.8,
    )

    assert quantile == 8.0


def test_validation_calibration_block_is_strictly_later_than_selection():
    timestamps = pd.date_range("2025-01-01", periods=10, freq="30min")
    frame = pd.DataFrame({"timestamp": timestamps})
    validation = pd.Series(True, index=frame.index)

    selection, calibration = _validation_selection_calibration_masks(
        frame,
        validation,
        calibration_fraction=0.3,
    )

    assert selection.any() and calibration.any()
    assert frame.loc[selection, "timestamp"].max() < frame.loc[
        calibration, "timestamp"
    ].min()
    assert not (selection & calibration).any()


def test_operational_feature_config_includes_issue_time_observations():
    features = feature_selection.FEATURE_SETS["accident_risk"]

    assert {
        "traffic_volume",
        "avg_speed",
        "accident_count",
        "congestion_level",
    }.issubset(features)
    assert not any("_t_plus_" in feature for feature in features)


def test_legacy_preprocessor_one_hot_encodes_current_congestion_input():
    frame = pd.DataFrame({
        "traffic_volume": [100.0, 200.0],
        "congestion_level": ["Free-flow", "Heavy"],
    })
    preprocessor = preprocessing._build_preprocessor(
        ["traffic_volume", "congestion_level"]
    )
    transformed = preprocessor.fit_transform(frame)
    encoded_names = preprocessor.get_feature_names_out()

    assert transformed.shape == (2, 3)
    assert "categorical__congestion_level_Free-flow" in encoded_names
    assert np.isfinite(transformed).all()


def test_accident_labels_normalize_csv_float_encodings():
    labels = _binary_label_text(pd.Series([0.0, 1.0, 0.0]))

    assert labels.tolist() == ["0", "1", "0"]


def test_congestion_label_order_is_explicit_and_stable():
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(CONGESTION_ORDER)

    assert encoder.transform(CONGESTION_ORDER).tolist() == [0, 1, 2, 3]
    assert encoder.inverse_transform([0, 1, 2, 3]).tolist() == CONGESTION_ORDER


def test_m6_sequences_are_contiguous_and_confined_to_each_partition(tmp_path, monkeypatch):
    rows = []
    split_names = ["train", "validation", "test"]
    start = pd.Timestamp("2025-01-01")
    for split_index, split_name in enumerate(split_names):
        timestamps = pd.date_range(
            start + pd.Timedelta(days=split_index * 3),
            periods=60,
            freq="30min",
        )
        for index, timestamp in enumerate(timestamps):
            row = {
                "road_id": "R1",
                "timestamp": timestamp.isoformat(),
                "split": split_name,
                "traffic_volume_t_plus_30": float(index + 100 * split_index),
                "weather_condition": "Clear",
                "temperature_band": "Mild",
                "congestion_level": "Moderate",
            }
            row.update({column: float(index + 1) for column in NUMERIC})
            if split_name == "test" and index == 50:
                row["rainfall"] = np.nan
            rows.append(row)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    pd.DataFrame(rows).to_csv(split_dir / "horizon_30.csv", index=False)
    monkeypatch.setattr(m6_corrected_sequences, "ROOT", tmp_path)

    summary = m6_corrected_sequences.prepare_horizon(
        30,
        split_dir=split_dir,
        output_dir=tmp_path / "sequences",
        report_dir=tmp_path / "reports",
    )

    assert summary["preprocessing_fit_partition"] == "train"
    assert summary["partition_boundaries_are_not_crossed"]
    assert summary["road_boundaries_are_not_crossed"]
    assert summary["feature_count"] >= len(NUMERIC) + len(CATEGORICAL)
    for split_name in split_names:
        sequences = np.load(
            tmp_path / "sequences" / f"{split_name}_30m.npz"
        )
        assert len(sequences["y"]) == 13
        assert sequences["X"].shape[1] == 48
        assert np.isfinite(sequences["X"]).all()
        issue_time = pd.Timestamp(str(sequences["timestamp"][0]))
        target_time = pd.Timestamp(str(sequences["target_timestamp"][0]))
        assert target_time == issue_time + pd.Timedelta(minutes=30)

    preprocessing = json.loads(
        (tmp_path / "reports" / "m6_sequence_30m.json").read_text()
    )
    assert preprocessing["preprocessing_fit_partition"] == "train"
    preprocessor = joblib.load(
        tmp_path / "sequences" / "sequence_preprocessing_30m.joblib"
    )
    assert preprocessor["fit_partition"] == "train"
    assert len(preprocessor["numeric_imputer"].statistics_) == len(NUMERIC)
    assert preprocessor["numeric_imputer"].statistics_[NUMERIC.index("rainfall")] == 30.5
    assert len(preprocessor["scaler"].mean_) == len(STANDARD_NUMERIC)
    assert len(preprocessor["minmax_scaler"].data_min_) == len(MINMAX_NUMERIC)


def test_m5_rejects_all_future_target_columns_as_features(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    pd.DataFrame({
        "road_id": ["R1"],
        "timestamp": ["2025-01-01 00:00:00"],
        "split": ["train"],
        "traffic_volume_t_plus_30": [100.0],
        "traffic_volume_t_plus_60": [120.0],
    }).to_csv(split_dir / "horizon_30.csv", index=False)
    feature_config = tmp_path / "features.json"
    feature_config.write_text(json.dumps({
        "feature_sets": {"traffic_volume": ["traffic_volume_t_plus_60"]}
    }))

    with pytest.raises(ValueError, match="Forbidden/current/uncertain features"):
        run_corrected_m5(
            split_dir=split_dir,
            feature_config_path=feature_config,
            model_dir=tmp_path / "models",
            report_dir=tmp_path / "reports",
            horizons=[30],
            tasks=["traffic_volume"],
        )


def test_pipeline_refuses_to_overwrite_an_existing_run(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for filename in run_pipeline.RAW_FILES:
        (raw_dir / filename).write_text("unchanged")
    manifest_path = (
        tmp_path / "reports" / "corrected_v2" / "repeat" / "run_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}")
    monkeypatch.setattr(run_pipeline, "ROOT", tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        run_pipeline.run_pipeline("repeat", raw_dir=raw_dir)


def test_selected_range_report_summarizes_real_filtered_records():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 08:30"]),
        "road_id": ["R1", "R2"],
        "traffic_volume": [100.0, 200.0],
        "avg_speed": [30.0, 40.0],
        "travel_time": [2.0, 3.0],
        "rainfall": [0.0, 1.0],
        "congestion_level": ["Free-flow", "Heavy"],
    })

    report = build_selected_range_report(frame)

    assert "Observations: 2" in report
    assert "Road segments: 2" in report
    assert "Mean traffic volume: 150.000" in report
    assert "Highest mean-volume segments" in report


def test_upload_processor_preserves_raw_inputs_and_writes_isolated_outputs(
    tmp_path,
    monkeypatch,
):
    root = tmp_path
    upload_root = root / "data/interim/upload_runs"
    raw_dir = root / "data/raw"
    raw_dir.mkdir(parents=True)
    for filename in upload_inference.EXPECTED:
        (raw_dir / filename).write_bytes(f"original-{filename}".encode())
    original_raw = {
        path.name: path.read_bytes()
        for path in raw_dir.iterdir()
    }

    def fake_validation(raw_dir, reports_dir):
        assert all((raw_dir / filename).is_file() for filename in original_raw)
        return {
            "files": {
                name: {"missing_required_columns": []}
                for name in upload_inference.EXPECTED
            }
        }

    def fake_m2(raw_dir, interim_dir, report_dir):
        interim_dir.mkdir(parents=True)
        (interim_dir / "flowcast_m2_merged.csv").write_text("road_id\nR1\n")
        return {"merge": {"unmatched_weather_rows": 0, "unmatched_calendar_rows": 0}}

    def fake_m3(merged_file, processed_dir, report_dir):
        processed_dir.mkdir(parents=True)
        pd.DataFrame({"road_id": ["R1"]}).to_csv(
            processed_dir / "flowcast_m3_features.csv",
            index=False,
        )
        return {"output_rows": 1}

    monkeypatch.setattr(upload_inference, "ROOT", root)
    monkeypatch.setattr(upload_inference, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(upload_inference, "RAW_DIR", raw_dir)
    monkeypatch.setattr(upload_inference, "run_validation", fake_validation)
    monkeypatch.setattr(upload_inference, "_run_m2", fake_m2)
    monkeypatch.setattr(upload_inference, "_run_m3", fake_m3)

    result = upload_inference.process_uploaded_sources({
        "traffic_sensor_log.csv": b"uploaded-traffic",
    })

    assert result["manifest"]["status"] == "completed"
    assert set(result["manifest"]["fallback_files_from_project_raw_data"]) == {
        "calendar_events.csv",
        "weather_observations.csv",
    }
    assert (result["manifest"]["run_id"] != "")
    assert (result["feature_path"]).is_file()
    assert {
        path.name: path.read_bytes()
        for path in raw_dir.iterdir()
    } == original_raw


def test_upload_prediction_uses_latest_complete_row_and_conformal_bounds(
    tmp_path,
    monkeypatch,
):
    class IdentityPreprocessor:
        def transform(self, values):
            return values.to_numpy()

    class CapacityModel:
        def predict(self, values):
            return values[:, 0]

    monkeypatch.setattr(upload_inference, "ROOT", tmp_path)
    bundle = {
        "features": ["road_capacity"],
        "preprocessor": IdentityPreprocessor(),
        "model": CapacityModel(),
        "conformal_quantile": 10.0,
        "artifact": tmp_path / "models/traffic_volume_30m.joblib",
    }
    frame = pd.DataFrame({
        "road_id": ["R1", "R1", "R2", "R2", "R3"],
        "date": ["2025-01-01"] * 5,
        "time": ["08:00", "08:30", "08:00", "08:30", "08:30"],
        "road_capacity": [100.0, 120.0, 90.0, np.nan, np.nan],
    })

    result, skipped = upload_inference.predict_latest_uploaded_rows(
        bundle,
        frame,
        task="traffic_volume",
        horizon=30,
        model_run_id="verified-run",
    )

    assert skipped == 1
    assert result["road_id"].tolist() == ["R1", "R2"]
    assert result["prediction"].tolist() == [120, 90]
    assert result["lower_90"].tolist() == [110, 80]
    assert result["upper_90"].tolist() == [130, 100]
    assert result.loc[1, "as_of"] == "2025-01-01 08:00:00"
    assert result.loc[0, "model_run_id"] == "verified-run"


def test_upload_prediction_labels_svm_decision_scores_as_uncalibrated(tmp_path, monkeypatch):
    class IdentityPreprocessor:
        def transform(self, values):
            return values.to_numpy()

    class LinearSVM:
        def predict(self, values):
            return (values[:, 0] > 0).astype(int)

        def decision_function(self, values):
            return values[:, 0] - 0.25

    monkeypatch.setattr(upload_inference, "ROOT", tmp_path)
    bundle = {
        "features": ["sensor_signal"],
        "preprocessor": IdentityPreprocessor(),
        "model": LinearSVM(),
        "validation_threshold": 0.1,
        "artifact": tmp_path / "models/accident_risk_30m.joblib",
    }
    frame = pd.DataFrame({
        "road_id": ["R1", "R2"],
        "date": ["2025-01-01", "2025-01-01"],
        "time": ["08:00", "08:00"],
        "sensor_signal": [0.5, 0.0],
    })

    result, skipped = upload_inference.predict_latest_uploaded_rows(
        bundle,
        frame,
        task="accident_risk",
        horizon=30,
        model_run_id="verified-run",
    )

    assert skipped == 0
    assert result["risk_score_type"].tolist() == [
        "uncalibrated_decision_score",
        "uncalibrated_decision_score",
    ]
    assert result["risk_score"].tolist() == [0.25, -0.25]
    assert result["prediction"].tolist() == [1, 0]
