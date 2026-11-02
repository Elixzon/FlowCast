from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]

FEATURE_CONFIG = ROOT / "reports/m5_feature_selection.json"
SPLIT_DIR = ROOT / "data/processed/m5_splits"
OUTPUT_DIR = ROOT / "data/processed/m5_preprocessed"

HORIZONS = [30, 60, 90, 120]

TARGETS = {
    "traffic_volume": ("traffic_volume_t_plus_", "regression"),
    "travel_time": ("travel_time_t_plus_", "regression"),
    "congestion_level": ("congestion_level_t_plus_", "classification"),
    "accident_risk": ("accident_risk_t_plus_", "classification"),
}

CATEGORICAL = [
    "road_id",
    "weather_condition",
    "temperature_band",
    "congestion_level",
]


def _build_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [column for column in features if column in CATEGORICAL]
    numeric = [column for column in features if column not in categorical]
    return ColumnTransformer(
        transformers=[
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
                categorical,
            ),
        ]
    )


def main():
    config = json.loads(FEATURE_CONFIG.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for horizon in HORIZONS:
        path = SPLIT_DIR / f"horizon_{horizon}.csv"
        df = pd.read_csv(path)

        for task, (target_prefix, task_type) in TARGETS.items():
            target = f"{target_prefix}{horizon}"
            features = config["feature_sets"][task]

            X = df[features].copy()
            y = df[target].copy()
            split = df["split"].copy()

            # Remove incomplete feature rows without imputing warm-up history.
            valid = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid]
            y = y.loc[valid]
            split = split.loc[valid]

            preprocessor = _build_preprocessor(features)

            train_mask = split == "train"
            val_mask = split == "validation"
            test_mask = split == "test"

            if not train_mask.any() or not val_mask.any() or not test_mask.any():
                raise ValueError(f"Missing partition for {horizon}m {task}")

            # Fit transformations on training data only.
            preprocessor.fit(X.loc[train_mask])

            X_train = preprocessor.transform(X.loc[train_mask])
            X_val = preprocessor.transform(X.loc[val_mask])
            X_test = preprocessor.transform(X.loc[test_mask])

            task_dir = OUTPUT_DIR / f"horizon_{horizon}" / task
            task_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                task_dir / "prepared_data.npz",
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y.loc[train_mask].to_numpy(),
                y_val=y.loc[val_mask].to_numpy(),
                y_test=y.loc[test_mask].to_numpy(),
            )

            summary.append({
                "horizon": horizon,
                "task": task,
                "task_type": task_type,
                "features": len(features),
                "encoded_features": int(X_train.shape[1]),
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(val_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "removed_rows": int((~valid).sum()),
            })

            print(
                f"{horizon}m | {task} | "
                f"Train: {train_mask.sum()} | "
                f"Validation: {val_mask.sum()} | "
                f"Test: {test_mask.sum()} | "
                f"Encoded features: {X_train.shape[1]}"
            )

    report = OUTPUT_DIR / "preprocessing_report.json"
    report.write_text(json.dumps(summary, indent=4))

    print("\nPreprocessing completed.")
    print("Report:", report)


if __name__ == "__main__":
    main()
