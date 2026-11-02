from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

tf.keras.utils.set_random_seed(42)

DATA_DIR = Path("data/processed/m6_sequences")
MODEL_DIR = Path("models/m6")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data(name):
    data = np.load(DATA_DIR / f"{name}.npz")
    return data["X"].astype(np.float32), data["y"].astype(np.float32)

def main():
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("validation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "gru_30m.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    started_at = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )
    training_duration_seconds = time.perf_counter() - started_at

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = REPORT_DIR / "figures" / "m6"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(
        REPORT_DIR / "m6_gru_30m_training_history.csv",
        index_label="epoch",
    )
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("GRU 30-minute volume forecast loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "gru_30m_loss.png", dpi=160)
    plt.close()

    predictions = np.maximum(
        model.predict(X_val, batch_size=256).ravel(), 0
    )

    metrics = {
        "model": "GRU",
        "target": "traffic_volume_t_plus_30",
        "partition": "validation",
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "training_duration_seconds": training_duration_seconds,
        "hyperparameters": {
            "sequence_length": int(X_train.shape[1]),
            "gru_units": 64,
            "dropout": 0.2,
            "dense_units": 32,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "max_epochs": 40,
            "early_stopping_patience": 8,
            "reduce_lr_patience": 3,
            "seed": 42,
        },
        "MAE": float(mean_absolute_error(y_val, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_val, predictions))),
        "MAPE_percent": float(
            mean_absolute_percentage_error(y_val, predictions) * 100
        ),
        "R2": float(r2_score(y_val, predictions)),
        "test_used": False,
    }

    (REPORT_DIR / "m6_gru_30m.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("\nGRU validation results:")
    print(json.dumps(metrics, indent=2))
    print("Model saved:", MODEL_DIR / "gru_30m.keras")

if __name__ == "__main__":
    main()
