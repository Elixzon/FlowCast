import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

DATA = "data/processed/m5_preprocessed/horizon_30/accident_risk/prepared_data.npz"
MODEL = "models/m5/linear_svm_accident_risk_30m.joblib"

data = np.load(DATA)
model = joblib.load(MODEL)

X_val = data["X_val"]
y_val = data["y_val"].astype(int)

scores = model.decision_function(X_val)

print("Validation ROC-AUC:", round(roc_auc_score(y_val, scores), 4))
print("Validation PR-AUC:", round(average_precision_score(y_val, scores), 4))

print("\nTHRESHOLD COMPARISON")
print("Threshold | Precision | Recall | F1")

thresholds = np.unique(np.quantile(scores, np.linspace(0.90, 0.999, 15)))

for threshold in thresholds:
    predictions = (scores >= threshold).astype(int)

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(
        f"{threshold:9.4f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f}"
    )

print("\nDefault threshold confusion matrix:")
print(confusion_matrix(y_val, model.predict(X_val)))
