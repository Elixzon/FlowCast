from pathlib import Path
import json
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data/processed/m5_preprocessed/"
    "horizon_30/traffic_volume/prepared_data.npz"
)

MODEL_DIR = ROOT / "models/m5"
REPORT_DIR = ROOT / "reports"

CONFIGURATIONS = [
    {"n_estimators": 100, "max_depth": 14, "min_samples_leaf": 10},
    {"n_estimators": 150, "max_depth": 18, "min_samples_leaf": 10},
    {"n_estimators": 200, "max_depth": 18, "min_samples_leaf": 20},
]


def evaluate(y_true, predictions):
    predictions = np.maximum(predictions, 0)

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mape": float(
            np.mean(np.abs((y_true - predictions) / y_true)) * 100
        ),
    }


def main():
    data = np.load(DATA_FILE)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    results = []
    best_model = None
    best_config = None
    best_rmse = float("inf")

    for config in CONFIGURATIONS:
        print(f"\nTraining configuration: {config}")

        model = RandomForestRegressor(
            **config,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        metrics = evaluate(y_val, predictions)

        result = {**config, **metrics}
        results.append(result)

        print(
            f"MAE={metrics['mae']:.3f} | "
            f"RMSE={metrics['rmse']:.3f} | "
            f"MAPE={metrics['mape']:.3f}%"
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model = model
            best_config = result

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "random_forest_regression_30m.joblib"
    joblib.dump(best_model, model_path)

    report = {
        "model": "Random Forest Regression",
        "target": "traffic_volume_t_plus_30",
        "horizon_minutes": 30,
        "selection_metric": "validation RMSE",
        "best_configuration": best_config,
        "all_configurations": results,
        "test_data_used": False,
    }

    report_path = REPORT_DIR / "m5_random_forest_regression_30m.json"
    report_path.write_text(json.dumps(report, indent=4))

    print("\nBest configuration:")
    print(json.dumps(best_config, indent=4))
    print("\nModel saved:", model_path)
    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
