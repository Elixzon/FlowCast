from pathlib import Path
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/interim/flowcast_accident_history_experiment.csv"
REPORT = ROOT / "reports/m5_accident_feature_comparison_30m.json"

df = pd.read_csv(SOURCE)

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
    "accident_lag_1",
    "accident_lag_2",
    "accident_lag_48",
    "accident_rolling_sum_4",
    "accident_rolling_sum_8",
    "road_historical_accident_rate",
]

CATEGORICAL = ["road_id", "weather_condition", "temperature_band"]
TARGET = "accident_risk_t_plus_30"

# Same complete original-feature rows for both experiments.
base_complete = df[BASE].notna().all(axis=1)
train_mask = (df["split"] == "train") & base_complete
val_mask = (df["split"] == "validation") & base_complete

train = df.loc[train_mask].copy()
val = df.loc[val_mask].copy()

y_train = train[TARGET].astype(int)
y_val = val[TARGET].astype(int)

print("TRAIN ROWS:", len(train))
print("TRAIN POSITIVES:", int(y_train.sum()))
print("VALIDATION ROWS:", len(val))
print("VALIDATION POSITIVES:", int(y_val.sum()))

def make_preprocessor(features):
    categorical = [c for c in CATEGORICAL if c in features]
    numeric = [c for c in features if c not in categorical]

    transformers = [
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="median",
                    add_indicator=True
                ))
            ]),
            numeric,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ]

    return ColumnTransformer(transformers)

def make_models(positive_weight):
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=positive_weight,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            n_jobs=4,
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
        ),
    }

experiments = {
    "baseline_38_features": BASE,
    "augmented_44_features": BASE + HISTORY,
}

results = []

for experiment_name, features in experiments.items():
    print("\n" + "=" * 65)
    print("EXPERIMENT:", experiment_name)
    print("=" * 65)

    X_train = train[features]
    X_val = val[features]

    preprocessor = make_preprocessor(features)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    positive_weight = (
        (len(y_train) - int(y_train.sum())) / int(y_train.sum())
    )

    for model_name, model in make_models(positive_weight).items():
        print("\nTraining:", model_name)
        model.fit(X_train_t, y_train)

        if model_name == "Linear SVM":
            scores = model.decision_function(X_val_t)
            predictions = (scores >= 0).astype(int)
        else:
            scores = model.predict_proba(X_val_t)[:, 1]
            predictions = (scores >= 0.5).astype(int)

        metrics = {
            "experiment": experiment_name,
            "model": model_name,
            "roc_auc": float(roc_auc_score(y_val, scores)),
            "pr_auc": float(average_precision_score(y_val, scores)),
            "precision": float(precision_score(
                y_val, predictions, zero_division=0
            )),
            "recall": float(recall_score(
                y_val, predictions, zero_division=0
            )),
            "f1": float(f1_score(
                y_val, predictions, zero_division=0
            )),
            "validation_positive_count": int(y_val.sum()),
            "validation_rows": len(y_val),
        }

        results.append(metrics)

        print(
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

REPORT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT, "w") as f:
    json.dump(results, f, indent=2)

print("\nFINAL COMPARISON")
comparison = pd.DataFrame(results)
print(
    comparison[
        ["experiment", "model", "roc_auc", "pr_auc",
         "precision", "recall", "f1"]
    ].sort_values("roc_auc", ascending=False).to_string(index=False)
)

print("\nReport saved:", REPORT)
print("Test partition was not used.")
