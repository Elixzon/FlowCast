from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data/processed/m5_preprocessed/"
    "horizon_30/traffic_volume/prepared_data.npz"
)

REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models/m5"

LEARNING_RATE = 0.01
EPOCHS = 500
PATIENCE = 30
RANDOM_SEED = 42


def mse_loss(X, y, weights, bias):
    predictions = X @ weights + bias
    return np.mean((predictions - y) ** 2)


def train():
    np.random.seed(RANDOM_SEED)

    data = np.load(DATA_FILE)

    X_train = data["X_train"].astype(np.float64)
    X_val = data["X_val"].astype(np.float64)

    y_train = data["y_train"].astype(np.float64)
    y_val = data["y_val"].astype(np.float64)

    # Standardize target using training statistics only.
    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    weights = np.zeros(X_train.shape[1])
    bias = 0.0

    best_val_loss = float("inf")
    best_weights = weights.copy()
    best_bias = bias
    patience_counter = 0

    history = []

    for epoch in range(EPOCHS):
        predictions = X_train @ weights + bias
        errors = predictions - y_train_scaled

        # NumPy gradient calculations.
        dw = (2 / len(y_train)) * (X_train.T @ errors)
        db = 2 * np.mean(errors)

        weights -= LEARNING_RATE * dw
        bias -= LEARNING_RATE * db

        train_loss = mse_loss(
            X_train, y_train_scaled, weights, bias
        )

        val_loss = mse_loss(
            X_val, y_val_scaled, weights, bias
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "validation_loss": float(val_loss),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = weights.copy()
            best_bias = bias
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Validation MSE: {val_loss:.6f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        MODEL_DIR / "numpy_linear_regression_30m.npz",
        weights=best_weights,
        bias=best_bias,
        target_mean=y_mean,
        target_std=y_std,
    )

    report = {
        "model": "NumPy Linear Regression",
        "target": "traffic_volume_t_plus_30",
        "horizon_minutes": 30,
        "features": int(X_train.shape[1]),
        "best_validation_mse_scaled": float(best_val_loss),
        "epochs_completed": len(history),
        "learning_rate": LEARNING_RATE,
        "history": history,
    }

    (REPORT_DIR / "m5_numpy_linear_regression_30m.json").write_text(
        json.dumps(report, indent=4)
    )

    print("\nTraining completed.")
    print("Best validation MSE:", best_val_loss)
    print("Model saved:", MODEL_DIR)
    print("Report saved:", REPORT_DIR)


if __name__ == "__main__":
    train()
