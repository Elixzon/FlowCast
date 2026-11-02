from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/interim/flowcast_accident_history_experiment.csv"
REPORT = ROOT / "reports/m5_accident_threshold_cv_30m.csv"

df = pd.read_csv(FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

FEATURES = [
    "road_id", "latitude", "longitude", "road_capacity",
    "weather_condition", "temperature_band", "temperature",
    "rainfall", "visibility", "public_holiday", "event_flag",
    "roadwork_flag", "hour", "day_of_week", "hour_sin",
    "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "is_weekend", "is_peak_period", "rain_flag",
    "low_visibility_flag", "holiday_peak_interaction",
    "event_proximity_flag", "traffic_volume_lag_1",
    "traffic_volume_lag_2", "traffic_volume_lag_48",
    "traffic_volume_rolling_mean_4", "traffic_volume_rolling_std_4",
    "traffic_volume_rolling_mean_8", "traffic_volume_rolling_std_8",
    "avg_speed_lag_1", "avg_speed_lag_2", "avg_speed_lag_48",
    "avg_speed_rolling_mean_4", "avg_speed_rolling_std_4",
    "avg_speed_rolling_mean_8", "avg_speed_rolling_std_8"
]
CATS = ["road_id", "weather_condition", "temperature_band"]
TARGET = "accident_risk_t_plus_30"

train = df[df["split"] == "train"].copy()
train = train[train[FEATURES].notna().all(axis=1)].copy()

timestamps = np.array(sorted(train["timestamp"].unique()))
blocks = np.array_split(timestamps, 6)

alert_rates = [0.005, 0.01, 0.02, 0.05]
results = []

for fold in range(1, 4):
    val_times = blocks[fold + 2]
    cutoff = pd.Timestamp(val_times[0]) - pd.Timedelta(minutes=30)

    tr = train[train["timestamp"] < cutoff].copy()
    va = train[train["timestamp"].isin(val_times)].copy()

    numeric = [c for c in FEATURES if c not in CATS]

    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATS)
    ])

    X_train = prep.fit_transform(tr[FEATURES])
    X_val = prep.transform(va[FEATURES])

    y_train = tr[TARGET].astype(int).to_numpy()
    y_val = va[TARGET].astype(int).to_numpy()

    model = LinearSVC(
        C=0.01,
        class_weight="balanced",
        max_iter=20000,
        random_state=42
    )
    model.fit(X_train, y_train)
    scores = model.decision_function(X_val)

    print(f"\nFOLD {fold}: {len(va)} rows, {y_val.sum()} positives")

    for rate in alert_rates:
        threshold = np.quantile(scores, 1 - rate)
        pred = (scores >= threshold).astype(int)

        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())

        row = {
            "fold": fold,
            "alert_rate_target": rate,
            "actual_alert_rate": float(pred.mean()),
            "precision": float(precision_score(y_val, pred, zero_division=0)),
            "recall": float(recall_score(y_val, pred, zero_division=0)),
            "f1": float(f1_score(y_val, pred, zero_division=0)),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
        results.append(row)

        print(
            f"Top {rate:.1%} alerts | "
            f"Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f} | "
            f"F1={row['f1']:.4f} | "
            f"FP={fp} | FN={fn}"
        )

out = pd.DataFrame(results)
REPORT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(REPORT, index=False)

print("\nAVERAGE ACROSS CHRONOLOGICAL FOLDS")
print(
    out.groupby("alert_rate_target")[
        ["actual_alert_rate", "precision", "recall", "f1"]
    ].mean().round(4).to_string()
)

print("\nSaved:", REPORT)
print("Test partition was not used.")
