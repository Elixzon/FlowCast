"""Train corrected LSTM, GRU, and historical bidirectional comparison models."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.m6_corrected_sequences import HORIZONS, OUTPUT_DIR as SEQUENCE_DIR

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/m6/corrected_v2"
REPORT_DIR = ROOT / "reports/corrected_v2"
FIGURE_DIR = REPORT_DIR / "figures"
SEED = 42
EPOCHS = 40
BATCH_SIZE = 128
PATIENCE = 8
MODEL_TYPES = ["LSTM", "GRU", "Bidirectional LSTM"]
VALIDATION_CALIBRATION_FRACTION = 0.30


def _validation_selection_calibration_masks(
    timestamps: np.ndarray,
    calibration_fraction: float = VALIDATION_CALIBRATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    unique_times = pd.Index(pd.to_datetime(timestamps, errors="raise")).unique().sort_values()
    if len(unique_times) < 3 or not 0 < calibration_fraction < 1:
        raise ValueError("M6 validation data are too small for split calibration.")
    cutoff_index = int(np.floor(len(unique_times) * (1 - calibration_fraction)))
    cutoff_index = min(max(cutoff_index, 1), len(unique_times) - 1)
    cutoff = unique_times[cutoff_index]
    timestamp_values = pd.to_datetime(timestamps, errors="raise")
    selection_mask = np.asarray(timestamp_values < cutoff)
    calibration_mask = np.asarray(timestamp_values >= cutoff)
    if not selection_mask.any() or not calibration_mask.any():
        raise ValueError("M6 validation selection/calibration split is empty.")
    return selection_mask, calibration_mask


def build_model(model_type: str, sequence_shape: tuple[int, int]) -> tf.keras.Model:
    """Create an uninitialized recurrent volume-regression model."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown M6 model type: {model_type}")
    recurrent = (
        tf.keras.layers.LSTM(64)
        if model_type == "LSTM"
        else tf.keras.layers.GRU(64)
        if model_type == "GRU"
        else tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=sequence_shape),
        recurrent,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    prediction = np.maximum(np.asarray(prediction).reshape(-1), 0)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "MAPE_percent": float(
            np.mean(np.abs(y_true - prediction) / np.maximum(np.abs(y_true), 1e-8))
            * 100
        ),
        "R2": float(r2_score(y_true, prediction)),
    }


def train_one(
    horizon: int,
    model_type: str,
    sequence_dir: Path = SEQUENCE_DIR,
    model_dir: Path = MODEL_DIR,
    report_dir: Path = REPORT_DIR,
) -> dict:
    """Fit on train, select epoch on validation, and do not read test data."""
    train_file = sequence_dir / f"train_{horizon}m.npz"
    validation_file = sequence_dir / f"validation_{horizon}m.npz"
    sequence_report = report_dir / f"m6_sequence_{horizon}m.json"
    if not train_file.exists() or not validation_file.exists():
        raise FileNotFoundError(f"Corrected M6 sequence split missing for {horizon}m.")
    train_data = np.load(train_file)
    validation_data = np.load(validation_file)
    X_train = train_data["X"].astype(np.float32)
    y_train = train_data["y"].astype(np.float32)
    X_validation = validation_data["X"].astype(np.float32)
    y_validation = validation_data["y"].astype(np.float32)
    validation_timestamps = pd.to_datetime(
        validation_data["timestamp"], errors="raise"
    )
    validation_selection_mask, validation_calibration_mask = (
        _validation_selection_calibration_masks(
            validation_data["timestamp"]
        )
    )
    X_validation_selection = X_validation[validation_selection_mask]
    y_validation_selection = y_validation[validation_selection_mask]
    if X_train.ndim != 3 or X_train.shape[1] != 48:
        raise ValueError(f"Unexpected M6 input shape: {X_train.shape}")
    if not np.isfinite(X_train).all() or not np.isfinite(X_validation).all():
        raise ValueError("Non-finite sequence values reached M6 training.")

    target_mean = float(y_train.mean())
    target_scale = float(y_train.std())
    if not target_scale:
        target_scale = 1.0
    y_train_scaled = (y_train - target_mean) / target_scale
    y_validation_scaled = (
        y_validation_selection - target_mean
    ) / target_scale

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_type.lower().replace(' ', '_')}_{horizon}m"
    model_path = model_dir / f"{stem}.keras"
    model = build_model(model_type, (X_train.shape[1], X_train.shape[2]))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    started = time.perf_counter()
    history = model.fit(
        X_train,
        y_train_scaled,
        validation_data=(X_validation_selection, y_validation_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
        shuffle=False,
    )
    duration = float(time.perf_counter() - started)
    best_model = tf.keras.models.load_model(model_path)
    val_prediction_scaled = best_model.predict(
        X_validation_selection, batch_size=256, verbose=0
    ).reshape(-1)
    val_prediction = val_prediction_scaled * target_scale + target_mean
    metrics = _regression_metrics(y_validation_selection, val_prediction)

    history_frame = pd.DataFrame(history.history)
    history_path = report_dir / f"{stem}_training_history.csv"
    history_frame.to_csv(history_path, index_label="epoch")
    figure_path = figure_dir / f"{stem}_loss.png"
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["loss"], label="Training MSE")
    plt.plot(history.history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled-target MSE")
    plt.title(f"{model_type} · {horizon}-minute volume forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()

    seq_metadata = json.loads(sequence_report.read_text(encoding="utf-8"))
    train_timestamps = pd.to_datetime(train_data["timestamp"], errors="raise")
    validation_calibration_timestamps = validation_timestamps[
        validation_calibration_mask
    ]
    result = {
        "model": model_type,
        "target": f"traffic_volume_t_plus_{horizon}",
        "horizon_minutes": horizon,
        "partition": "validation",
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "training_period": {
            "start": str(train_timestamps.min()),
            "end": str(train_timestamps.max()),
        },
        "validation_period": {
            "start": str(validation_timestamps.min()),
            "end": str(validation_timestamps.max()),
        },
        "validation_selection_rows": int(validation_selection_mask.sum()),
        "validation_selection_period": {
            "start": str(validation_timestamps[validation_selection_mask].min()),
            "end": str(validation_timestamps[validation_selection_mask].max()),
        },
        "validation_calibration_rows": int(validation_calibration_mask.sum()),
        "validation_calibration_period": {
            "start": str(validation_calibration_timestamps.min()),
            "end": str(validation_calibration_timestamps.max()),
        },
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "training_duration_seconds": duration,
        "validation_metrics": metrics,
        "target_scaling": {
            "training_mean": target_mean,
            "training_standard_deviation": target_scale,
        },
        "hyperparameters": {
            "sequence_length": 48,
            "units": 64,
            "dropout": 0.2,
            "dense_units": 32,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "MSE",
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
            "early_stopping_patience": PATIENCE,
            "reduce_lr_on_plateau": True,
            "shuffle": False,
            "seed": SEED,
            "feature_count": seq_metadata["feature_count"],
            "feature_names": seq_metadata["feature_names"],
        },
        "test_used_for_training_or_selection": False,
        "model_artifact": str(model_path.relative_to(ROOT)),
        "history_artifact": str(history_path.relative_to(ROOT)),
        "loss_figure": str(figure_path.relative_to(ROOT)),
        "software_versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    metadata_path = model_dir / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_all(
    sequence_dir: Path = SEQUENCE_DIR,
    model_dir: Path = MODEL_DIR,
    report_dir: Path = REPORT_DIR,
    horizons: list[int] | None = None,
    model_types: list[str] | None = None,
) -> list[dict]:
    """Train LSTM/GRU at every horizon and a historical BiLSTM comparison."""
    requested_horizons = horizons or HORIZONS
    requested_models = model_types or MODEL_TYPES
    results = []
    for horizon in requested_horizons:
        for model_type in requested_models:
            results.append(train_one(
                horizon,
                model_type,
                sequence_dir,
                model_dir,
                report_dir,
            ))
            print(
                f"Trained {model_type} {horizon}m: "
                f"validation RMSE={results[-1]['validation_metrics']['RMSE']:.4f}",
                flush=True,
            )
    (report_dir / "m6_corrected_validation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    train_all()
