from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/interim/flowcast_accident_history_experiment.csv"
REPORT = ROOT / "reports/m5_accident_feature_cv_30m.json"

df = pd.read_csv(FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

BASE = [
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

HISTORY = [
    "accident_lag_1", "accident_lag_2", "accident_lag_48",
    "accident_rolling_sum_4", "accident_rolling_sum_8",
    "road_historical_accident_rate"
]

CATS = ["road_id", "weather_condition", "temperature_band"]
TARGET = "accident_risk_t_plus_30"

train = df[df["split"] == "train"].copy()
train = train[train[BASE].notna().all(axis=1)].copy()

timestamps = np.array(sorted(train["timestamp"].unique()))
blocks = np.array_split(timestamps, 6)

experiments = {
    "baseline_38": BASE,
    "augmented_44": BASE + HISTORY,
}

results = []

def preprocess(features, train_frame, val_frame):
    cats = [c for c in CATS if c in features]
    nums = [c for c in features if c not in cats]

    transformer = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ]), nums),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cats)
    ])

    x_train = transformer.fit_transform(train_frame[features])
    x_val = transformer.transform(val_frame[features])
    return x_train, x_val

for fold_number in range(1, 4):
    train_times = np.concatenate(blocks[:fold_number + 2])
    val_times = blocks[fold_number + 2]

    # Purge one 30-minute interval before validation.
    val_start = pd.Timestamp(val_times[0])
    cutoff = val_start - pd.Timedelta(minutes=30)

    fold_train = train[train["timestamp"] < cutoff].copy()
    fold_val = train[train["timestamp"].isin(val_times)].copy()

    if fold_train.empty or fold_val.empty:
        continue

    y_train = fold_train[TARGET].astype(int)
    y_val = fold_val[TARGET].astype(int)

    if y_train.nunique() < 2 or y_val.nunique() < 2:
        print("Skipping fold with only one class:", fold_number)
        continue

    weight = (len(y_train) - y_train.sum()) / y_train.sum()

    print(f"\nFOLD {fold_number}")
    print("Train:", len(fold_train), "positives:", int(y_train.sum()))
    print("Validation:", len(fold_val), "positives:", int(y_val.sum()))

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=weight, eval_metric="aucpr",
            tree_method="hist", random_state=42, n_jobs=4
        ),
        "Linear SVM": LinearSVC(
            C=1.0, class_weight="balanced",
            max_iter=10000, random_state=42
        )
    }

    for exp_name, features in experiments.items():
        x_train, x_val = preprocess(features, fold_train, fold_val)

        for model_name, model in models.items():
            model.fit(x_train, y_train)

            if model_name == "Linear SVM":
                scores = model.decision_function(x_val)
            else:
                scores = model.predict_proba(x_val)[:, 1]

            row = {
                "fold": fold_number,
                "experiment": exp_name,
                "model": model_name,
                "train_rows": len(fold_train),
                "validation_rows": len(fold_val),
                "validation_positives": int(y_val.sum()),
                "roc_auc": float(roc_auc_score(y_val, scores)),
                "pr_auc": float(average_precision_score(y_val, scores))
            }
            results.append(row)

            print(
                exp_name, model_name,
                f"ROC-AUC={row['roc_auc']:.4f}",
                f"PR-AUC={row['pr_auc']:.4f}"
            )

result_df = pd.DataFrame(results)
REPORT.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT, "w") as f:
    json.dump(results, f, indent=2)

print("\nCROSS-VALIDATION SUMMARY")
print(
    result_df.groupby(["experiment", "model"])[["roc_auc", "pr_auc"]]
    .agg(["mean", "std"])
    .round(4)
    .to_string()
)

print("\nSaved:", REPORT)
print("Test partition was not used.")
