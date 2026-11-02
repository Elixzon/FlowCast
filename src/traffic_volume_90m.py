from pathlib import Path
import json
import numpy as np
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data/processed/m5_preprocessed/"
    "horizon_90/traffic_volume/prepared_data.npz"
)

MODEL_DIR = ROOT / "models/m5"
REPORT_DIR = ROOT / "reports"

CONFIGURATIONS = {
    "Decision Tree": {
        "max_depth": 18,
        "min_samples_leaf": 50,
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": 14,
        "min_samples_leaf": 10,
    },
    "XGBoost": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
    },
}


def evaluate(y_true, predictions):
    predictions = np.maximum(predictions, 0)

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mape": float(
            np.mean(np.abs((y_true - predictions) / y_true)) * 100
        ),
    }


def create_model(name, config):
    if name == "Decision Tree":
        return DecisionTreeRegressor(
            **config,
            random_state=42,
        )

    if name == "Random Forest":
        return RandomForestRegressor(
            **config,
            random_state=42,
            n_jobs=-1,
        )

    return XGBRegressor(
        **config,
        objective="reg:squarederror",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
    )


def main():
    print("Loading 90-minute prepared data...", flush=True)

    data = np.load(DATA_FILE)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    print("Training rows:", len(y_train))
    print("Validation rows:", len(y_val))
    print("Features:", X_train.shape[1])

    results = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for name, config in CONFIGURATIONS.items():
        print(f"\nTraining {name}...", flush=True)

        model = create_model(name, config)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        metrics = evaluate(y_val, predictions)

        result = {
            "model": name,
            "target": "traffic_volume_t_plus_90",
            "horizon_minutes": 90,
            "configuration": config,
            **metrics,
        }

        results.append(result)

        model_filename = (
            name.lower().replace(" ", "_")
            + "_regression_90m.joblib"
        )

        joblib.dump(model, MODEL_DIR / model_filename)

        print(
            f"MAE={metrics['mae']:.3f} | "
            f"RMSE={metrics['rmse']:.3f} | "
            f"MAPE={metrics['mape']:.3f}%",
            flush=True,
        )

    results.sort(key=lambda x: x["rmse"])

    report = {
        "experiment": "90-minute traffic-volume regression",
        "target": "traffic_volume_t_plus_90",
        "horizon_minutes": 90,
        "selection_metric": "validation RMSE",
        "test_data_used": False,
        "results": results,
        "best_model": results[0],
    }

    report_path = REPORT_DIR / "m5_traffic_volume_90m.json"
    report_path.write_text(json.dumps(report, indent=4))

    print("\n90-MINUTE MODEL COMPARISON")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['model']} | "
            f"RMSE={result['rmse']:.3f} | "
            f"MAPE={result['mape']:.3f}%"
        )

    print("\nBest model:", results[0]["model"])
    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
