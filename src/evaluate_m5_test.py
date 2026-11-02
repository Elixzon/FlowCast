from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/m5"
DATA_DIR = ROOT / "data/processed/m5_preprocessed"
REPORT_DIR = ROOT / "reports"

HORIZONS = [30, 60, 90, 120]

# Frozen model selections from validation.
SELECTIONS = {
    "traffic_volume": {
        30: "xgboost_regression_30m.joblib",
        60: "xgboost_regression_60m.joblib",
        90: "xgboost_regression_90m.joblib",
        120: "xgboost_regression_120m.joblib",
    },
    "travel_time": {
        30: "xgboost_travel_time_30m.joblib",
        60: "random_forest_travel_time_60m.joblib",
        90: "xgboost_travel_time_90m.joblib",
        120: "xgboost_travel_time_120m.joblib",
    },
    "congestion_level": {
        30: "xgboost_congestion_30m.joblib",
        60: "xgboost_congestion_60m.joblib",
        90: "xgboost_congestion_90m.joblib",
        120: "xgboost_congestion_120m.joblib",
    },
    "accident_risk": {
        30: "linear_svm_accident_risk_30m.joblib",
        60: "linear_svm_accident_risk_60m.joblib",
        90: "linear_svm_accident_risk_90m.joblib",
        120: "random_forest_accident_risk_120m.joblib",
    },
}


def evaluate_regression(y_true, predictions):
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "MAPE_percent": float(
            mean_absolute_percentage_error(y_true, predictions) * 100
        ),
        "R2": float(r2_score(y_true, predictions)),
    }


def evaluate_classification(y_true, predictions, task, scores=None):
    labels = sorted(np.unique(y_true).tolist())

    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=labels
        ).tolist(),
        "class_labels": [str(x) for x in labels],
    }

    if task == "accident_risk":
        result["positive_precision"] = float(
            precision_score(y_true, predictions, pos_label=1, zero_division=0)
        )
        result["positive_recall"] = float(
            recall_score(y_true, predictions, pos_label=1, zero_division=0)
        )
        result["positive_f1"] = float(
            f1_score(y_true, predictions, pos_label=1, zero_division=0)
        )

        if scores is not None:
            result["ROC_AUC"] = float(roc_auc_score(y_true, scores))
            result["PR_AUC"] = float(average_precision_score(y_true, scores))

    return result


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for task, horizon_models in SELECTIONS.items():
        for horizon in HORIZONS:
            model_file = horizon_models[horizon]
            model_path = MODEL_DIR / model_file
            data_path = (
                DATA_DIR / f"horizon_{horizon}" / task / "prepared_data.npz"
            )

            if not model_path.exists():
                raise FileNotFoundError(model_path)
            if not data_path.exists():
                raise FileNotFoundError(data_path)

            with np.load(data_path, allow_pickle=True) as data:
                X_test = data["X_test"]
                y_test = data["y_test"]

            saved = joblib.load(model_path)

            if isinstance(saved, dict):
                model = saved["model"]
                label_encoder = saved.get("label_encoder")
            else:
                model = saved
                label_encoder = None

            if task == "congestion_level":
                encoded_predictions = model.predict(X_test).astype(int)
                predictions = label_encoder.inverse_transform(
                    encoded_predictions
                )
                metrics = evaluate_classification(
                    y_test.astype(str),
                    predictions.astype(str),
                    task,
                )

            elif task == "accident_risk":
                predictions = model.predict(X_test).astype(int)

                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_test)
                else:
                    scores = None

                metrics = evaluate_classification(
                    y_test.astype(int),
                    predictions,
                    task,
                    scores,
                )

            else:
                predictions = model.predict(X_test)
                metrics = evaluate_regression(y_test, predictions)

            row = {
                "task": task,
                "horizon_minutes": horizon,
                "model": model_file,
                "test_rows": int(len(y_test)),
                **{
                    key: value
                    for key, value in metrics.items()
                    if not isinstance(value, (list, dict))
                },
            }

            results.append({
                **row,
                "details": metrics,
            })

            print(
                f"{horizon:3}m | {task:16} | "
                f"{model_file:42} | "
                f"Evaluation complete"
            )

    csv_rows = [item for item in results]
    flat_rows = [
        {key: value for key, value in item.items() if key != "details"}
        for item in csv_rows
    ]

    csv_path = REPORT_DIR / "m5_test_scoreboard.csv"
    json_path = REPORT_DIR / "m5_test_evaluation.json"
    md_path = REPORT_DIR / "m5_test_evaluation.md"

    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=4))

    lines = [
        "# M5 Final Test Evaluation",
        "",
        "Evaluation uses the untouched chronological test partition.",
        "Models and configurations were selected using validation results.",
        "No retraining or threshold tuning was performed on test data.",
        "",
    ]

    for item in results:
        task = item["task"]
        horizon = item["horizon_minutes"]
        metrics = item["details"]

        lines += [
            f"## {task.replace('_', ' ').title()} — {horizon} minutes",
            "",
            f"- Model: `{item['model']}`",
            f"- Test rows: {item['test_rows']}",
        ]

        for key, value in metrics.items():
            if isinstance(value, (list, dict)):
                lines.append(f"- {key}: `{json.dumps(value)}`")
            elif isinstance(value, float):
                lines.append(f"- {key}: {value:.6f}")
            else:
                lines.append(f"- {key}: {value}")

        lines.append("")

    md_path.write_text("\n".join(lines))

    print("\nFinal test evaluation completed.")
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Markdown:", md_path)
    print("Total evaluated combinations:", len(results))


if __name__ == "__main__":
    main()
