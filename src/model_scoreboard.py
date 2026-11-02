from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

REPORT_FILES = {
    "Decision Tree": "m5_decision_tree_regression_30m.json",
    "Random Forest": "m5_random_forest_regression_30m.json",
    "XGBoost": "m5_xgboost_regression_30m.json",
    "Linear SVR": "m5_svm_regression_30m.json",
}

rows = []

# Load conventional model reports.
for model_name, filename in REPORT_FILES.items():
    report = json.loads((REPORT_DIR / filename).read_text())
    config = report["best_configuration"]

    rows.append({
        "model": model_name,
        "MAE": config["mae"],
        "RMSE": config["rmse"],
        "MAPE_percent": config["mape"],
        "target": report["target"],
        "horizon_minutes": report["horizon_minutes"],
    })

# Recalculate NumPy baseline metrics.
data = np.load(
    ROOT / "data/processed/m5_preprocessed/"
    "horizon_30/traffic_volume/prepared_data.npz"
)

model = np.load(
    ROOT / "models/m5/numpy_linear_regression_30m.npz"
)

X_val = data["X_val"]
y_val = data["y_val"]

predictions = (
    X_val @ model["weights"] + float(model["bias"])
) * float(model["target_std"]) + float(model["target_mean"])

predictions = np.maximum(predictions, 0)

errors = y_val - predictions

rows.append({
    "model": "NumPy Linear Regression",
    "MAE": float(np.mean(np.abs(errors))),
    "RMSE": float(np.sqrt(np.mean(errors ** 2))),
    "MAPE_percent": float(
        np.mean(np.abs(errors / y_val)) * 100
    ),
    "target": "traffic_volume_t_plus_30",
    "horizon_minutes": 30,
})

df = pd.DataFrame(rows)
df = df.sort_values("RMSE").reset_index(drop=True)
df.insert(0, "rank_by_RMSE", range(1, len(df) + 1))

df.to_csv(REPORT_DIR / "m5_model_scoreboard_30m.csv", index=False)

(REPORT_DIR / "m5_model_scoreboard_30m.json").write_text(
    df.to_json(orient="records", indent=4)
)

print("\nM5 MODEL SCOREBOARD — 30-MINUTE TRAFFIC VOLUME")
print(df.to_string(index=False))
print("\nPRD MAPE target: <= 12%")
print("\nScoreboard updated successfully.")
