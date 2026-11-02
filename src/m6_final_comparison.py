from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data/processed/m5_splits/horizon_30.csv"
FEATURE_FILE = ROOT / "reports/m5_feature_selection.json"
SEQUENCE_FILE = ROOT / "data/processed/m6_sequences/test.npz"

XGB_FILE = ROOT / "models/m5/xgboost_regression_30m.joblib"
LSTM_FILE = ROOT / "models/m6/lstm_30m.keras"
GRU_FILE = ROOT / "models/m6/gru_30m.keras"

OUTPUT = ROOT / "reports"
OUTPUT.mkdir(exist_ok=True)

CATEGORICAL = ["road_id", "weather_condition", "temperature_band"]
TARGET = "traffic_volume_t_plus_30"


def metrics(y_true, y_pred):
    y_pred = np.maximum(np.asarray(y_pred).reshape(-1), 0)
    y_true = np.asarray(y_true).reshape(-1)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_percent": float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        ),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    print("Loading sequence test data...")
    seq = np.load(SEQUENCE_FILE)

    X_seq = seq["X"]
    y_seq = seq["y"].astype(float)
    keys = pd.DataFrame({
        "road_id": seq["road_id"].astype(str),
        "timestamp": pd.to_datetime(seq["timestamp"], utc=True),
        "sequence_y": y_seq,
    })

    print("Loading M5 split and feature configuration...")
    df = pd.read_csv(SPLIT_FILE)
    df["road_id"] = df["road_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    with open(FEATURE_FILE) as f:
        config = json.load(f)

    features = config["feature_sets"]["traffic_volume"]

    X = df[features].copy()
    y = df[TARGET]
    split = df["split"]

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    split = split.loc[valid]

    numeric = [c for c in features if c not in CATEGORICAL]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
    ])

    train_mask = split.eq("train")
    test_mask = split.eq("test")

    print("Fitting XGBoost preprocessing on training rows only...")
    preprocessor.fit(X.loc[train_mask])

    test_frame = df.loc[X.index[test_mask]].copy()
    test_frame = test_frame[["road_id", "timestamp", TARGET]]

    print("Matching sequence samples to original test rows...")
    matched = keys.merge(
        test_frame,
        on=["road_id", "timestamp"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not matched["_merge"].eq("both").all():
        missing = int((matched["_merge"] != "both").sum())
        raise ValueError(f"{missing} sequence samples could not be matched.")

    if not np.allclose(
        matched["sequence_y"].to_numpy(),
        matched[TARGET].to_numpy(),
        rtol=0,
        atol=1e-4,
    ):
        raise ValueError("Sequence targets do not match original test targets.")

    extra_features = [
        col for col in features
        if col not in ["road_id", "timestamp"]
    ]

    xgb_X_raw = matched[["road_id", "timestamp"]].merge(
        df[["road_id", "timestamp"] + extra_features],
        on=["road_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )[features]

    xgb_X = preprocessor.transform(xgb_X_raw)

    print("Loading and predicting with XGBoost...")
    xgb_model = joblib.load(XGB_FILE)
    xgb_pred = xgb_model.predict(xgb_X)

    print("Loading and predicting with LSTM...")
    lstm = tf.keras.models.load_model(LSTM_FILE)
    lstm_pred = lstm.predict(X_seq, verbose=0).reshape(-1)

    print("Loading and predicting with GRU...")
    gru = tf.keras.models.load_model(GRU_FILE)
    gru_pred = gru.predict(X_seq, verbose=0).reshape(-1)

    if not (len(y_seq) == len(xgb_pred) == len(lstm_pred) == len(gru_pred)):
        raise ValueError("Prediction lengths do not match.")

    results = []
    for name, pred in [
        ("XGBoost", xgb_pred),
        ("LSTM", lstm_pred),
        ("GRU", gru_pred),
    ]:
        row = {
            "model": name,
            "test_samples": len(y_seq),
            **metrics(y_seq, pred),
        }
        results.append(row)

    result_df = pd.DataFrame(results).sort_values("RMSE")

    csv_path = OUTPUT / "m6_final_test_comparison.csv"
    json_path = OUTPUT / "m6_final_test_comparison.json"
    md_path = OUTPUT / "m6_final_test_comparison.md"

    result_df.to_csv(csv_path, index=False)

    report = {
        "evaluation": "Final untouched test comparison",
        "horizon_minutes": 30,
        "test_samples": len(y_seq),
        "sequence_length": int(X_seq.shape[1]),
        "features_per_step": int(X_seq.shape[2]),
        "target": TARGET,
        "same_test_samples_for_all_models": True,
        "preprocessing_fit_on_training_only": True,
        "models": results,
        "best_by_rmse": result_df.iloc[0]["model"],
        "best_by_mae": result_df.sort_values("MAE").iloc[0]["model"],
    }

    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        "# M6 Final Test Comparison",
        "",
        "- Horizon: 30 minutes",
        f"- Identical test samples: {len(y_seq)}",
        "- Models: XGBoost, LSTM, GRU",
        "- Preprocessing fitted on training data only",
        "- Test data used only for final evaluation",
        "",
        "| Model | Samples | MAE | RMSE | MAPE (%) | R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for item in result_df.to_dict("records"):
        lines.append(
            f"| {item['model']} | {item['test_samples']} "
            f"| {item['MAE']:.4f} | {item['RMSE']:.4f} "
            f"| {item['MAPE_percent']:.4f} | {item['R2']:.4f} |"
        )

    lines += [
        "",
        f"Best model by RMSE: **{report['best_by_rmse']}**",
        f"Best model by MAE: **{report['best_by_mae']}**",
        "",
        "All models were evaluated on the same matched test samples.",
    ]

    md_path.write_text("\n".join(lines))

    print("\nFinal comparison completed.")
    print(result_df.to_string(index=False))
    print("\nReports saved:")
    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
