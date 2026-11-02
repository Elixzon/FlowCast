from pathlib import Path
import json
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/processed/m5_preprocessed/horizon_30/congestion_level/prepared_data.npz"
MODEL_DIR = ROOT / "models/m5"
REPORT_DIR = ROOT / "reports"

def main():
    data = np.load(DATA_FILE, allow_pickle=True)

    X_train, X_val = data["X_train"], data["X_val"]
    y_train, y_val = data["y_train"], data["y_val"]

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.04,
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        tree_method="hist"
    )

    print("Training class-weighted XGBoost...", flush=True)
    model.fit(X_train, y_train_encoded, sample_weight=weights)

    predictions = model.predict(X_val)

    report = classification_report(
        y_val_encoded,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    results = {
        "model": "Class-weighted XGBoost",
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
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_val_encoded, predictions,
            labels=np.arange(len(encoder.classes_))
        ).tolist()
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": model, "label_encoder": encoder},
        MODEL_DIR / "xgboost_congestion_30m_balanced.joblib"
    )

    report_path = REPORT_DIR / "m5_congestion_classification_30m_balanced.json"
    report_path.write_text(json.dumps(results, indent=4))

    print("\nCLASS-WEIGHTED XGBOOST RESULTS")
    print("Accuracy:", round(results["accuracy"], 4))
    print("Macro-F1:", round(results["macro_f1"], 4))
    print("Macro Precision:", round(results["macro_precision"], 4))
    print("Macro Recall:", round(results["macro_recall"], 4))

    for cls in encoder.classes_:
        values = report[cls]
        print(
            f"{cls}: F1={values['f1-score']:.4f}, "
            f"Recall={values['recall']:.4f}"
        )

    print("\nConfusion matrix:")
    for row in results["confusion_matrix"]:
        print(row)

    print("\nReport saved:", report_path)

if __name__ == "__main__":
    main()
