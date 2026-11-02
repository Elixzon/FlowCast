from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/interim/flowcast_accident_history_experiment.csv"
REPORT = ROOT / "reports/m5_accident_svm_tuning_30m.json"

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

CATEGORICAL = [
    "road_id", "weather_condition", "temperature_band"
]
TARGET = "accident_risk_t_plus_30"

# Training partition only; test data is never loaded into a model.
train = df[df["split"] == "train"].copy()
train = train[train[FEATURES].notna().all(axis=1)].copy()

timestamps = np.array(sorted(train["timestamp"].unique()))
blocks = np.array_split(timestamps, 6)

folds = []
for fold_number in range(1, 4):
    validation_times = blocks[fold_number + 2]
    validation_start = pd.Timestamp(validation_times[0])
    cutoff = validation_start - pd.Timedelta(minutes=30)

    fold_train = train[train["timestamp"] < cutoff].copy()
    fold_val = train[
        train["timestamp"].isin(validation_times)
    ].copy()

    if fold_train.empty or fold_val.empty:
        raise ValueError(f"Empty fold {fold_number}")

    folds.append((fold_number, fold_train, fold_val))

def prepare_data(fold_train, fold_val):
    numeric = [c for c in FEATURES if c not in CATEGORICAL]

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", add_indicator=True
            )),
            ("scaler", StandardScaler())
        ]), numeric),

        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])

    X_train = preprocessor.fit_transform(fold_train[FEATURES])
    X_val = preprocessor.transform(fold_val[FEATURES])

    return X_train, X_val

C_VALUES = [0.01, 0.1, 1.0, 10.0]
results = []

for C in C_VALUES:
    print("\n" + "=" * 55)
    print(f"LINEAR SVM — C={C}")
    print("=" * 55)

    for fold_number, fold_train, fold_val in folds:
        X_train, X_val = prepare_data(fold_train, fold_val)

        y_train = fold_train[TARGET].astype(int)
        y_val = fold_val[TARGET].astype(int)

        model = LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=20000,
            random_state=42
        )

        model.fit(X_train, y_train)
        scores = model.decision_function(X_val)

        roc = roc_auc_score(y_val, scores)
        pr = average_precision_score(y_val, scores)

        row = {
            "C": C,
            "fold": fold_number,
            "train_rows": len(fold_train),
            "validation_rows": len(fold_val),
            "validation_positives": int(y_val.sum()),
            "roc_auc": float(roc),
            "pr_auc": float(pr)
        }
        results.append(row)

        print(
            f"Fold {fold_number}: "
            f"ROC-AUC={roc:.4f} | PR-AUC={pr:.4f}"
        )

result_df = pd.DataFrame(results)

summary = (
    result_df.groupby("C")[["roc_auc", "pr_auc"]]
    .agg(["mean", "std"])
    .round(5)
)

print("\nTUNING SUMMARY")
print(summary.to_string())

# Select using mean ROC-AUC; use PR-AUC as a secondary comparison.
means = result_df.groupby("C")[["roc_auc", "pr_auc"]].mean()
best_C = means.sort_values(
    ["roc_auc", "pr_auc"],
    ascending=False
).index[0]

print("\nBest C by mean ROC-AUC:", best_C)

REPORT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT, "w") as f:
    json.dump({
        "selection_metric": "mean_chronological_cv_roc_auc",
        "best_C": float(best_C),
        "fold_results": results,
        "summary": {
            str(index): {
                "mean_roc_auc": float(means.loc[index, "roc_auc"]),
                "mean_pr_auc": float(means.loc[index, "pr_auc"])
            }
            for index in means.index
        }
    }, f, indent=2)

print("\nReport saved:", REPORT)
print("Test partition was not used.")
