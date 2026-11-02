from pathlib import Path
import json
import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/m5_preprocessed/horizon_60/accident_risk/prepared_data.npz"
MODEL_DIR = ROOT / "models/m5"
REPORT_DIR = ROOT / "reports"


def evaluate(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def main():
    data = np.load(DATA_FILE)

    X_train, y_train = data["X_train"], data["y_train"].astype(int)
    X_val, y_val = data["X_val"], data["y_val"].astype(int)

    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    scale_pos_weight = negatives / positives

    print("Training rows:", len(y_train))
    print("Validation rows:", len(y_val))
    print("Positive cases:", positives)
    print("Negative cases:", negatives)
    print("Scale positive weight:", round(scale_pos_weight, 3))

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            tree_method="hist",
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        model.fit(X_train, y_train)

        if name == "Linear SVM":
            scores = model.decision_function(X_val)
            predictions = model.predict(X_val)
        else:
            scores = model.predict_proba(X_val)[:, 1]
            predictions = (scores >= 0.5).astype(int)

        metrics = evaluate(y_val, predictions, scores)
        results.append({"model": name, **metrics})

        filename = name.lower().replace(" ", "_") + "_accident_risk_60m.joblib"
        joblib.dump(model, MODEL_DIR / filename)

        print(
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f}",
            flush=True,
        )

    results.sort(key=lambda x: x["roc_auc"], reverse=True)

    report = {
        "experiment": "60-minute accident-risk classification",
        "target": "accident_risk_t_plus_60",
        "horizon_minutes": 60,
        "selection_metric": "validation ROC-AUC",
        "test_data_used": False,
        "results": results,
        "best_model": results[0],
    }

    report_path = REPORT_DIR / "m5_accident_risk_60m.json"
    report_path.write_text(json.dumps(report, indent=4))

    print("\nACCIDENT-RISK MODEL COMPARISON")

    for rank, result in enumerate(results, 1):
        print(
            f"{rank}. {result['model']} | "
            f"ROC-AUC={result['roc_auc']:.4f} | "
            f"PR-AUC={result['pr_auc']:.4f} | "
            f"Recall={result['recall']:.4f}"
        )

    print("\nBest model:", results[0]["model"])
    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
