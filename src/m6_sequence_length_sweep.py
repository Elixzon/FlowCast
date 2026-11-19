"""Validation-selected M6 sequence-length experiments with matched test rows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from src.m5_corrected_experiments import _split_conformal_quantile
from src.m6_corrected_comparison import (
    _match_sequences_to_rows,
    _predict_deep,
    _raw_feature_names,
    _sequence_keys,
)
from src.m6_corrected_sequences import HORIZONS, prepare_all
from src.m6_corrected_train import (
    _regression_metrics,
    _validation_selection_calibration_masks,
    train_one,
)


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_LENGTHS = (24, 48, 96)
MODEL_TYPES_TO_SELECT = ("LSTM", "GRU")


def _key_set(keys: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    return set(zip(keys["road_id"].astype(str), keys["timestamp"]))


def _sequence_indices_in_order(
    sequence_keys: pd.DataFrame,
    requested_keys: pd.DataFrame,
) -> np.ndarray:
    sequence_pairs = list(zip(
        sequence_keys["road_id"].astype(str),
        sequence_keys["timestamp"],
    ))
    if len(set(sequence_pairs)) != len(sequence_pairs):
        raise ValueError("Sequence candidate contains duplicate road/timestamp keys.")
    index_by_key = dict(zip(sequence_pairs, sequence_keys["sequence_index"]))
    requested_pairs = list(zip(
        requested_keys["road_id"].astype(str),
        requested_keys["timestamp"],
    ))
    missing = [key for key in requested_pairs if key not in index_by_key]
    if missing:
        raise ValueError(
            f"Sequence candidate is missing {len(missing)} requested matched keys."
        )
    return np.asarray(
        [index_by_key[key] for key in requested_pairs],
        dtype=int,
    )


def _prediction_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    calibration_residuals: np.ndarray,
) -> dict[str, Any]:
    metrics = _regression_metrics(y_true, predictions)
    quantile = _split_conformal_quantile(calibration_residuals)
    lower = np.maximum(predictions - quantile, 0)
    upper = predictions + quantile
    metrics["prediction_interval"] = {
        "method": "90% split conformal absolute residual",
        "calibration_rows": int(len(calibration_residuals)),
        "quantile": quantile,
        "observed_test_coverage": float(
            np.mean((y_true >= lower) & (y_true <= upper))
        ),
        "mean_test_width": float(np.mean(upper - lower)),
        "limitation": (
            "The finite-sample guarantee requires exchangeability; time-series "
            "dependence and temporal distribution shift mean nominal coverage "
            "is not guaranteed."
        ),
    }
    return metrics


def _load_candidate(
    model_type: str,
    horizon: int,
    sequence_length: int,
    sequence_dirs: dict[int, Path],
    model_dirs: dict[int, Path],
) -> tuple[tf.keras.Model, dict, dict[str, np.lib.npyio.NpzFile]]:
    suffix = f"_seq{sequence_length}" if sequence_length != 48 else ""
    stem = f"{model_type.lower().replace(' ', '_')}_{horizon}m{suffix}"
    model_path = model_dirs[sequence_length] / f"{stem}.keras"
    metadata_path = model_dirs[sequence_length] / f"{stem}_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"M6 sequence candidate is incomplete: {stem}")
    model = tf.keras.models.load_model(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = {
        split: np.load(
            sequence_dirs[sequence_length] / f"{split}_{horizon}m.npz"
        )
        for split in ("validation", "test")
    }
    return model, metadata, arrays


def run_sequence_length_sweep(
    split_dir: Path,
    base_sequence_dir: Path,
    m5_model_dir: Path,
    model_dir: Path,
    report_dir: Path,
    horizons: list[int] | None = None,
) -> list[dict]:
    """Train 24/48/96-step variants against one shared validation boundary."""
    requested_horizons = horizons or HORIZONS
    output_dir = report_dir / "sequence_length_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = {48: base_sequence_dir}
    model_dirs = {}
    report_dirs = {}
    for sequence_length in SEQUENCE_LENGTHS:
        sequence_dir = (
            base_sequence_dir
            if sequence_length == 48
            else output_dir / f"sequences_{sequence_length}"
        )
        sequence_report_dir = output_dir / f"reports_{sequence_length}"
        sequence_model_dir = model_dir / f"sequence_{sequence_length}"
        if sequence_length == 48:
            sequence_report_dir.mkdir(parents=True, exist_ok=True)
            for horizon in requested_horizons:
                source_report = report_dir / f"m6_sequence_{horizon}m.json"
                if not source_report.is_file():
                    raise FileNotFoundError(
                        f"48-step sequence report missing for {horizon}m."
                    )
                (sequence_report_dir / source_report.name).write_text(
                    source_report.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
        else:
            prepare_all(
                split_dir=split_dir,
                output_dir=sequence_dir,
                report_dir=sequence_report_dir,
                horizons=requested_horizons,
                sequence_length=sequence_length,
            )
        sequence_dirs[sequence_length] = sequence_dir
        model_dirs[sequence_length] = sequence_model_dir
        report_dirs[sequence_length] = sequence_report_dir

    validation_cutoffs = {}
    validation_keys_by_horizon = {}
    for horizon in requested_horizons:
        per_length_keys = [
            _sequence_keys(
                np.load(
                    sequence_dirs[sequence_length]
                    / f"validation_{horizon}m.npz"
                ),
                horizon,
            )
            for sequence_length in SEQUENCE_LENGTHS
        ]
        shared_validation_keys = set.intersection(*(
            _key_set(keys) for keys in per_length_keys
        ))
        if not shared_validation_keys:
            raise ValueError(
                f"No shared validation sequence rows across lengths at {horizon}m."
            )
        shared_times = sorted({timestamp for _, timestamp in shared_validation_keys})
        validation_keys_by_horizon[horizon] = shared_validation_keys
        _, calibration_mask = _validation_selection_calibration_masks(
            np.asarray([timestamp.isoformat() for timestamp in shared_times])
        )
        validation_cutoffs[horizon] = shared_times[
            int(np.flatnonzero(calibration_mask)[0])
        ]

    for sequence_length in SEQUENCE_LENGTHS:
        model_types = (
            [*MODEL_TYPES_TO_SELECT, "Bidirectional LSTM"]
            if sequence_length == 48
            else list(MODEL_TYPES_TO_SELECT)
        )
        for horizon in requested_horizons:
            for model_type in model_types:
                train_one(
                    horizon,
                    model_type,
                    sequence_dir=sequence_dirs[sequence_length],
                    model_dir=model_dirs[sequence_length],
                    report_dir=report_dirs[sequence_length],
                    sequence_length=sequence_length,
                    validation_selection_end=str(validation_cutoffs[horizon]),
                    validation_keys=validation_keys_by_horizon[horizon],
                )

    all_results = []
    for horizon in requested_horizons:
        candidate_keys = {}
        candidate_data = {}
        for sequence_length in SEQUENCE_LENGTHS:
            model_types = (
                [*MODEL_TYPES_TO_SELECT, "Bidirectional LSTM"]
                if sequence_length == 48
                else MODEL_TYPES_TO_SELECT
            )
            for model_type in model_types:
                model, metadata, arrays = _load_candidate(
                    model_type,
                    horizon,
                    sequence_length,
                    sequence_dirs,
                    model_dirs,
                )
                validation_keys = _sequence_keys(arrays["validation"], horizon)
                test_keys = _sequence_keys(arrays["test"], horizon)
                candidate_name = f"{model_type} · {sequence_length} steps"
                candidate_keys[candidate_name] = {
                    "validation": validation_keys,
                    "test": test_keys,
                }
                candidate_data[candidate_name] = {
                    "model": model,
                    "metadata": metadata,
                    "arrays": arrays,
                }

        validation_intersection = set.intersection(*(
            _key_set(item["validation"])
            for item in candidate_keys.values()
        ))
        test_intersection = set.intersection(*(
            _key_set(item["test"])
            for item in candidate_keys.values()
        ))
        if not validation_intersection or not test_intersection:
            raise ValueError(f"No common sequence rows across candidates at {horizon}m.")

        split_file = split_dir / f"horizon_{horizon}.csv"
        frame = pd.read_csv(split_file)
        frame["road_id"] = frame["road_id"].astype(str)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        target = f"traffic_volume_t_plus_{horizon}"
        m5_artifact = joblib.load(
            m5_model_dir / f"traffic_volume_{horizon}m.joblib"
        )
        candidate_bundle = joblib.load(
            m5_model_dir
            / f"traffic_volume_{horizon}m_validation_candidates.joblib"
        )
        feature_columns = list(m5_artifact["features"])
        metadata_features = candidate_data["LSTM · 48 steps"]["metadata"][
            "hyperparameters"
        ]["feature_names"]
        if _raw_feature_names(metadata_features) != set(feature_columns):
            raise ValueError(f"M5/M6 issue-time feature parity failed at {horizon}m.")


        # The sequence-row helper also verifies targets; build exact common keys
        # from a single candidate and then filter by the shared intersection.
        reference_validation = next(iter(candidate_keys.values()))["validation"]
        reference_test = next(iter(candidate_keys.values()))["test"]
        reference_validation = reference_validation.loc[
            [
                (str(road), timestamp) in validation_intersection
                for road, timestamp in zip(
                    reference_validation["road_id"],
                    reference_validation["timestamp"],
                )
            ]
        ]
        reference_test = reference_test.loc[
            [
                (str(road), timestamp) in test_intersection
                for road, timestamp in zip(
                    reference_test["road_id"], reference_test["timestamp"]
                )
            ]
        ]
        validation_selection_mask, validation_calibration_mask = (
            _validation_selection_calibration_masks(
                reference_validation["timestamp"].astype(str).to_numpy()
            )
        )
        validation_selection_keys = reference_validation.loc[
            validation_selection_mask
        ].copy()
        validation_calibration_keys = reference_validation.loc[
            validation_calibration_mask
        ].copy()
        test_keys = reference_test.copy()
        validation_selection = _match_sequences_to_rows(
            validation_selection_keys, frame, feature_columns, target, "validation"
        )
        validation_calibration = _match_sequences_to_rows(
            validation_calibration_keys, frame, feature_columns, target, "validation"
        )
        matched_test = _match_sequences_to_rows(
            test_keys, frame, feature_columns, target, "test"
        )
        y_validation = validation_selection[target].to_numpy(dtype=float)
        y_calibration = validation_calibration[target].to_numpy(dtype=float)
        y_test = matched_test[target].to_numpy(dtype=float)

        classical_X_validation = candidate_bundle["preprocessor"].transform(
            validation_selection[feature_columns]
        )
        classical_validation_rmse = {}
        for name, estimator in candidate_bundle["models"].items():
            prediction = np.maximum(
                estimator.predict(classical_X_validation).reshape(-1), 0
            )
            classical_validation_rmse[name] = float(np.sqrt(mean_squared_error(
                y_validation, prediction
            )))
        selected_classical_name = min(
            classical_validation_rmse, key=classical_validation_rmse.get
        )
        selected_classical_model = candidate_bundle["models"][
            selected_classical_name
        ]
        classical_X_calibration = candidate_bundle["preprocessor"].transform(
            validation_calibration[feature_columns]
        )
        classical_X_test = candidate_bundle["preprocessor"].transform(
            matched_test[feature_columns]
        )
        classical_calibration_prediction = np.maximum(
            selected_classical_model.predict(classical_X_calibration).reshape(-1),
            0,
        )
        classical_test_prediction = np.maximum(
            selected_classical_model.predict(classical_X_test).reshape(-1), 0
        )
        classical_metrics = _prediction_metrics(
            y_test,
            classical_test_prediction,
            np.abs(y_calibration - classical_calibration_prediction),
        )
        results = [{
            "model": f"Classical {selected_classical_name}",
            "family": "classical",
            "sequence_length": None,
            "selected": True,
            "validation_selection_rmse": classical_validation_rmse[
                selected_classical_name
            ],
            "test_rows": len(y_test),
            **classical_metrics,
        }]

        candidate_predictions = {}
        for name, payload in candidate_data.items():
            sequence_length = int(payload["metadata"]["hyperparameters"]["sequence_length"])
            keys = candidate_keys[name]["validation"]
            selection_indices = _sequence_indices_in_order(
                keys,
                validation_selection_keys,
            )
            validation_sequences = payload["arrays"]["validation"]["X"][
                selection_indices
            ]
            validation_prediction = _predict_deep(
                payload["model"], validation_sequences, payload["metadata"]
            )
            validation_rmse = float(np.sqrt(mean_squared_error(
                y_validation, validation_prediction
            )))
            if name in candidate_predictions:
                raise ValueError(f"Duplicate M6 candidate name: {name}")
            candidate_predictions[name] = {
                "validation_rmse": validation_rmse,
                "sequence_length": sequence_length,
                "model_type": payload["metadata"]["model"],
            }

        selected_deep_name = min(
            (
                name for name, value in candidate_predictions.items()
                if value["model_type"] in MODEL_TYPES_TO_SELECT
            ),
            key=lambda name: candidate_predictions[name]["validation_rmse"],
        )
        for name, payload in candidate_data.items():
            sequence_keys = candidate_keys[name]
            sequence_data = payload["arrays"]

            def indices_for(
                split_name: str,
                selected_rows: pd.DataFrame,
            ) -> np.ndarray:
                return _sequence_indices_in_order(
                    sequence_keys[split_name],
                    selected_rows,
                )

            validation_calibration_prediction = _predict_deep(
                payload["model"],
                sequence_data["validation"]["X"][
                    indices_for("validation", validation_calibration_keys)
                ],
                payload["metadata"],
            )
            test_prediction = _predict_deep(
                payload["model"],
                sequence_data["test"]["X"][indices_for("test", test_keys)],
                payload["metadata"],
            )
            metrics = _prediction_metrics(
                y_test,
                test_prediction,
                np.abs(y_calibration - validation_calibration_prediction),
            )
            row = {
                "model": name,
                "family": "deep_learning",
                "sequence_length": payload["metadata"]["hyperparameters"][
                    "sequence_length"
                ],
                "selected": name == selected_deep_name,
                "validation_selection_rmse": candidate_predictions[name][
                    "validation_rmse"
                ],
                "test_rows": len(y_test),
                **metrics,
            }
            results.append(row)

        for row in results:
            row["difference_vs_classical_rmse"] = float(
                row["RMSE"] - classical_metrics["RMSE"]
            )
        validation_period = {
            "start": str(validation_selection_keys["timestamp"].min()),
            "end": str(validation_selection_keys["timestamp"].max()),
        }
        calibration_period = {
            "start": str(validation_calibration_keys["timestamp"].min()),
            "end": str(validation_calibration_keys["timestamp"].max()),
        }
        selected = next(row for row in results if row["selected"] and row["family"] == "deep_learning")
        selected_payload = candidate_data[selected_deep_name]
        card = {
            "model_name": selected_deep_name,
            "target": target,
            "forecast_horizon_minutes": horizon,
            "selected_on": "matched early chronological validation RMSE",
            "validation_selection_period": validation_period,
            "validation_calibration_period": calibration_period,
            "matched_test_rows": len(y_test),
            "sequence_length": selected["sequence_length"],
            "hyperparameters": selected_payload["metadata"]["hyperparameters"],
            "validation_rmse": selected["validation_selection_rmse"],
            "test_metrics": selected,
            "classical_comparator": results[0],
            "test_used_for_selection": False,
            "model_artifact": selected_payload["metadata"]["model_artifact"],
            "training_history": selected_payload["metadata"]["history_artifact"],
            "training_curve": selected_payload["metadata"]["loss_figure"],
        }
        card_dir = output_dir / "model_cards"
        card_dir.mkdir(parents=True, exist_ok=True)
        (card_dir / f"traffic_volume_{horizon}m_sequence_sweep.json").write_text(
            json.dumps(card, indent=2), encoding="utf-8"
        )
        all_results.append({
            "horizon_minutes": horizon,
            "matched_test_records": len(y_test),
            "validation_selection_period": validation_period,
            "validation_calibration_period": calibration_period,
            "selected_deep_model": selected_deep_name,
            "selected_deep_test_rmse": selected["RMSE"],
            "classical_model": results[0]["model"],
            "classical_test_rmse": results[0]["RMSE"],
            "difference_vs_classical_rmse": selected["difference_vs_classical_rmse"],
            "candidate_results": results,
            "test_records_used_for_selection": False,
        })

    (output_dir / "m6_sequence_length_sweep.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    rows = []
    for result in all_results:
        for candidate in result["candidate_results"]:
            rows.append({
                "horizon_minutes": result["horizon_minutes"],
                **candidate,
            })
    pd.DataFrame(rows).to_csv(
        output_dir / "m6_sequence_length_sweep.csv", index=False
    )
    markdown = [
        "# M6 Sequence-Length Sweep",
        "",
        "Candidates were selected using the matched early chronological validation",
        "window only. Models were evaluated once on the same matched test records",
        "per horizon; no test result was used to select architecture or sequence length.",
        "",
        "| Horizon | Matched test rows | Validation-selected deep model | Deep test RMSE | Classical test RMSE | Deep minus classical |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for result in all_results:
        markdown.append(
            f"| {result['horizon_minutes']}m | {result['matched_test_records']:,} | "
            f"{result['selected_deep_model']} | "
            f"{result['selected_deep_test_rmse']:.4f} | "
            f"{result['classical_test_rmse']:.4f} | "
            f"{result['difference_vs_classical_rmse']:+.4f} |"
        )
    (output_dir / "m6_sequence_length_sweep.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return all_results
