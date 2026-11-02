from pathlib import Path
import json
import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/processed/m5_preprocessed/horizon_60/congestion_level/prepared_data.npz"
MODEL_DIR = ROOT / "models/m5"
REPORT_DIR = ROOT / "reports"

def main():
    data = np.load(DATA_FILE, allow_pickle=True)

    X_train, X_val = data["X_train"], data["X_val"]
    y_train, y_val = data["y_train"], data["y_val"]

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)

    classes = encoder.classes_
    print("Classes:", list(classes))
    print("Training:", len(y_train))
    print("Validation:", len(y_val))

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=18, min_samples_leaf=50, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=18, min_samples_leaf=10,
            random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            objective="multi:softprob", num_class=len(classes),
            eval_metric="mlogloss", subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=4, tree_method="hist"
        ),
        "Linear SVM": LinearSVC(
            C=1.0, class_weight="balanced",
            random_state=42, max_iter=5000
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)

        model.fit(X_train, y_train_encoded)
        predictions = model.predict(X_val)

        metrics = {
            "accuracy": float(accuracy_score(y_val_encoded, predictions)),
            "macro_f1": float(f1_score(
                y_val_encoded, predictions, average="macro", zero_division=0
            )),
            "macro_precision": float(precision_score(
                y_val_encoded, predictions, average="macro", zero_division=0
            )),
            "macro_recall": float(recall_score(
                y_val_encoded, predictions, average="macro", zero_division=0
            )),
            "classification_report": classification_report(
                y_val_encoded, predictions,
                labels=np.arange(len(classes)),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_val_encoded, predictions,
                labels=np.arange(len(classes))
            ).tolist(),
        }

        result = {"model": name, **metrics}
        results.append(result)

        filename = name.lower().replace(" ", "_") + "_congestion_60m.joblib"
        joblib.dump(
            {"model": model, "label_encoder": encoder},
            MODEL_DIR / filename
        )

        print(
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} | "
            f"Precision={metrics['macro_precision']:.4f} | "
            f"Recall={metrics['macro_recall']:.4f}",
            flush=True,
        )

    results.sort(key=lambda x: x["macro_f1"], reverse=True)

    report = {
        "experiment": "60-minute congestion classification",
        "target": "congestion_level_t_plus_60",
        "horizon_minutes": 60,
        "selection_metric": "validation macro-F1",
        "test_data_used": False,
        "class_order": classes.tolist(),
        "results": results,
        "best_model": results[0],
    }

    report_path = REPORT_DIR / "m5_congestion_classification_60m.json"
    report_path.write_text(json.dumps(report, indent=4))

    print("\nCONGESTION MODEL COMPARISON")
    for rank, result in enumerate(results, 1):
        print(
            f"{rank}. {result['model']} | "
            f"Macro-F1={result['macro_f1']:.4f} | "
            f"Recall={result['macro_recall']:.4f}"
        )

    print("\nBest model:", results[0]["model"])
    print("Report saved:", report_path)


if __name__ == "__main__":
    main()
