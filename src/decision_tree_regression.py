from pathlib import Path
import json
import numpy as np
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data/processed/m5_preprocessed/"
    "horizon_30/traffic_volume/prepared_data.npz"
)

MODEL_DIR = ROOT / "models/m5"
REPORT_DIR = ROOT / "reports"

DEPTHS = [6, 10, 14, 18]
LEAF_SIZES = [20, 50, 100]


def calculate_metrics(y_true, predictions):
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
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

    for depth in DEPTHS:
        for leaf_size in LEAF_SIZES:

            model = DecisionTreeRegressor(
                max_depth=depth,
                min_samples_leaf=leaf_size,
                random_state=42,
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            metrics = calculate_metrics(y_val, predictions)

            result = {
                "max_depth": depth,
                "min_samples_leaf": leaf_size,
                **metrics,
            }

            results.append(result)

            print(
                f"Depth={depth}, Leaf={leaf_size} | "
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

    model_path = MODEL_DIR / "decision_tree_regression_30m.joblib"
    joblib.dump(best_model, model_path)

    report = {
        "model": "Decision Tree Regression",
        "target": "traffic_volume_t_plus_30",
        "horizon_minutes": 30,
        "selection_metric": "validation RMSE",
        "best_configuration": best_config,
        "all_configurations": results,
        "test_data_used": False,
    }

    report_path = REPORT_DIR / "m5_decision_tree_regression_30m.json"
    report_path.write_text(json.dumps(report, indent=4))

    print("\nBest configuration:")
    print(json.dumps(best_config, indent=4))
    print("\nModel saved:", model_path)
    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
