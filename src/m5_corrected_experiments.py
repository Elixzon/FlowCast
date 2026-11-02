"""Corrected chronological M5 model selection and untouched-test evaluation."""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy.linalg import lstsq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/processed/m5_splits_corrected_v2"
FEATURE_CONFIG = ROOT / "reports/m5_feature_selection.json"
MODEL_DIR = ROOT / "models/m5/corrected_v2"
REPORT_DIR = ROOT / "reports/corrected_v2"
CARD_DIR = REPORT_DIR / "model_cards"
SEED = 42
HORIZONS = [30, 60, 90, 120]
TASKS = {
    "traffic_volume": ("traffic_volume_t_plus_", "regression"),
    "travel_time": ("travel_time_t_plus_", "regression"),
    "congestion_level": ("congestion_level_t_plus_", "classification"),
    "accident_risk": ("accident_risk_t_plus_", "classification"),
}
CATEGORICAL = [
    "road_id", "weather_condition", "temperature_band", "congestion_level",
]
CONGESTION_ORDER = ["Free-flow", "Moderate", "Heavy", "Severe"]
VALIDATION_CALIBRATION_FRACTION = 0.30
INTERVAL_COVERAGE = 0.90


@dataclass
class NumpyGradientRegressor:
    """Linear regression optimized with NumPy gradient descent."""

    learning_rate: float = 0.01
    max_epochs: int = 1000
    patience: int = 50
    seed: int = SEED

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray],
    ) -> "NumpyGradientRegressor":
        X_train = np.asarray(X, dtype=np.float64)
        y_train = np.asarray(y, dtype=np.float64).reshape(-1)
        X_validation = np.asarray(validation_data[0], dtype=np.float64)
        y_validation = np.asarray(validation_data[1], dtype=np.float64).reshape(-1)
        if X_train.ndim != 2 or X_train.shape[0] != y_train.size:
            raise ValueError("NumPy regression training features and targets do not align.")
        if X_validation.shape[1] != X_train.shape[1]:
            raise ValueError("NumPy regression validation feature width differs from train.")
        if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
            raise ValueError("NumPy regression training data contain non-finite values.")

        self.target_mean_ = float(y_train.mean())
        self.target_scale_ = float(y_train.std())
        if self.target_scale_ == 0:
            self.target_scale_ = 1.0
        y_train_scaled = (y_train - self.target_mean_) / self.target_scale_
        y_validation_scaled = (y_validation - self.target_mean_) / self.target_scale_

        weights = np.zeros(X_train.shape[1], dtype=np.float64)
        bias = 0.0
        best_loss = float("inf")
        best_weights = weights.copy()
        best_bias = bias
        patience_count = 0
        self.history_ = []
        learning_rate = self.learning_rate

        for epoch in range(self.max_epochs):
            errors = X_train @ weights + bias - y_train_scaled
            weight_gradient = (2.0 / len(y_train)) * (X_train.T @ errors)
            bias_gradient = 2.0 * float(errors.mean())
            weights -= learning_rate * weight_gradient
            bias -= learning_rate * bias_gradient

            train_loss = float(np.mean((X_train @ weights + bias - y_train_scaled) ** 2))
            validation_loss = float(np.mean(
                (X_validation @ weights + bias - y_validation_scaled) ** 2
            ))
            self.history_.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            })
            if not np.isfinite(train_loss + validation_loss):
                raise FloatingPointError("NumPy gradient descent diverged.")
            if validation_loss < best_loss - 1e-10:
                best_loss = validation_loss
                best_weights = weights.copy()
                best_bias = bias
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= self.patience:
                break

        self.weights_ = best_weights
        self.bias_ = float(best_bias)
        self.best_validation_mse_scaled_ = float(best_loss)
        self.epochs_completed_ = len(self.history_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("NumPy gradient regressor has not been fitted.")
        scaled = np.asarray(X, dtype=np.float64) @ self.weights_ + self.bias_
        return scaled * self.target_scale_ + self.target_mean_

    def get_params(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "seed": self.seed,
        }


@dataclass
class StableLinearRegressor:
    """Ordinary least squares using rank-revealing QR for wide, collinear data."""

    fit_intercept: bool = True
    rank_tolerance: float = 1e-8

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StableLinearRegressor":
        features = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or features.shape[0] != target.size:
            raise ValueError("Linear-regression features and targets do not align.")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("Linear-regression training data contain non-finite values.")
        design = (
            np.column_stack((features, np.ones(features.shape[0])))
            if self.fit_intercept
            else features
        )
        coefficients, _, self.rank_, _ = lstsq(
            design,
            target,
            cond=self.rank_tolerance,
            lapack_driver="gelsy",
            check_finite=False,
        )
        if self.fit_intercept:
            self.coef_ = coefficients[:-1]
            self.intercept_ = float(coefficients[-1])
        else:
            self.coef_ = coefficients
            self.intercept_ = 0.0
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Linear regressor has not been fitted.")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("Linear-regression prediction feature width differs from train.")
        return features @ self.coef_ + self.intercept_

    def get_params(self) -> dict[str, Any]:
        return {
            "fit_intercept": self.fit_intercept,
            "rank_tolerance": self.rank_tolerance,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _build_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [column for column in CATEGORICAL if column in features]
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
        ],
        remainder="drop",
    )


def _regression_candidates() -> dict[str, Any]:
    return {
        "Linear Regression": StableLinearRegressor(),
        "NumPy Gradient Descent": NumpyGradientRegressor(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=16, min_samples_leaf=10, random_state=SEED,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            objective="reg:squarederror",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=4,
            tree_method="hist",
        ),
    }


def _classification_candidates(task: str, class_count: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=16, min_samples_leaf=10, class_weight="balanced",
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
        ),
    }
    if task == "accident_risk":
        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=4,
            tree_method="hist",
        )
        models["XGBoost (shallow)"] = XGBClassifier(
            n_estimators=400,
            max_depth=2,
            learning_rate=0.03,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=SEED,
            n_jobs=4,
            tree_method="hist",
        )
        models["XGBoost (extra positive weight)"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=4,
            tree_method="hist",
        )
    else:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=class_count,
            eval_metric="mlogloss",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=4,
            tree_method="hist",
        )
    return models


def _regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    prediction = np.maximum(np.asarray(prediction).reshape(-1), 0)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    denominator = np.maximum(np.abs(y_true), 1e-8)
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "MAPE_percent": float(np.mean(np.abs(y_true - prediction) / denominator) * 100),
        "R2": float(r2_score(y_true, prediction)),
    }


def _positive_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).reshape(-1)
    raise TypeError(f"{type(model).__name__} has no positive-class score method.")


def _classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None,
    class_labels: list[str],
    task: str,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=np.arange(len(class_labels))
        ).tolist(),
        "class_labels": class_labels,
    }
    if task == "accident_risk" and scores is not None:
        metrics.update({
            "positive_precision": float(precision_score(
                y_true, predictions, pos_label=1, zero_division=0
            )),
            "positive_recall": float(recall_score(
                y_true, predictions, pos_label=1, zero_division=0
            )),
            "positive_f1": float(f1_score(
                y_true, predictions, pos_label=1, zero_division=0
            )),
            "ROC_AUC": float(roc_auc_score(y_true, scores)),
            "PR_AUC": float(average_precision_score(y_true, scores)),
        })
    return metrics


def _selection_score(task: str, metrics: dict) -> float:
    if task == "accident_risk":
        return float(metrics["PR_AUC"])
    if task == "congestion_level":
        return float(metrics["macro_f1"])
    return -float(metrics["RMSE"])


def _select_accident_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        raise ValueError("Validation partition cannot support accident threshold selection.")
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[int(np.nanargmax(f1))])


def _validation_selection_calibration_masks(
    frame: pd.DataFrame,
    validation_mask: pd.Series,
    calibration_fraction: float = VALIDATION_CALIBRATION_FRACTION,
) -> tuple[pd.Series, pd.Series]:
    """Reserve the final chronological validation block for interval calibration."""
    timestamps = (
        frame.loc[validation_mask, "timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if len(timestamps) < 3 or not 0 < calibration_fraction < 1:
        raise ValueError("Validation partition is too small for split calibration.")
    cutoff_index = int(np.floor(len(timestamps) * (1 - calibration_fraction)))
    cutoff_index = min(max(cutoff_index, 1), len(timestamps) - 1)
    calibration_start = timestamps.iloc[cutoff_index]
    selection_mask = validation_mask & frame["timestamp"].lt(calibration_start)
    calibration_mask = validation_mask & frame["timestamp"].ge(calibration_start)
    if not selection_mask.any() or not calibration_mask.any():
        raise ValueError("Validation selection/calibration split is empty.")
    return selection_mask, calibration_mask


def _split_conformal_quantile(
    absolute_residuals: np.ndarray,
    coverage: float = INTERVAL_COVERAGE,
) -> float:
    """Compute the finite-sample split-conformal absolute-residual quantile."""
    scores = np.asarray(absolute_residuals, dtype=float).reshape(-1)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Conformal calibration scores must be non-empty and finite.")
    if not 0 < coverage < 1:
        raise ValueError("Conformal coverage must be between zero and one.")
    rank = int(np.ceil((len(scores) + 1) * coverage))
    if rank > len(scores):
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def _binary_label_text(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError("Accident-risk labels must contain only binary 0/1 values.")
    return numeric.astype("int64").astype(str)


def _periods(frame: pd.DataFrame, split: str) -> dict:
    part = frame.loc[frame["split"].eq(split), "timestamp"]
    return {
        "start": str(part.min()),
        "end": str(part.max()),
        "rows_before_feature_filter": int(len(part)),
    }


def _fit_candidate(
    name: str,
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    task: str,
    class_labels: list[str],
) -> tuple[Any, dict, float | None]:
    started = time.perf_counter()
    if isinstance(estimator, NumpyGradientRegressor):
        estimator.fit(
            X_train,
            y_train,
            validation_data=(X_validation, y_validation),
        )
    else:
        if task == "accident_risk" and name == "XGBoost":
            positives = max(int(np.sum(y_train == 1)), 1)
            estimator.set_params(scale_pos_weight=(len(y_train) - positives) / positives)
        elif (
            task == "accident_risk"
            and name == "XGBoost (extra positive weight)"
        ):
            positives = max(int(np.sum(y_train == 1)), 1)
            estimator.set_params(
                scale_pos_weight=2 * (len(y_train) - positives) / positives
            )
        elif task == "accident_risk" and name == "XGBoost (shallow)":
            positives = max(int(np.sum(y_train == 1)), 1)
            estimator.set_params(scale_pos_weight=(len(y_train) - positives) / positives)
        if task == "congestion_level" and name == "XGBoost":
            estimator.fit(
                X_train,
                y_train,
                sample_weight=compute_sample_weight("balanced", y_train),
            )
        else:
            estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    threshold = None
    if task == "accident_risk":
        scores = _positive_scores(estimator, X_validation)
        threshold = _select_accident_threshold(y_validation, scores)
        predictions = (scores >= threshold).astype(int)
        metrics = _classification_metrics(
            y_validation,
            predictions,
            scores,
            class_labels,
            task,
        )
    elif task == "congestion_level":
        predictions = estimator.predict(X_validation).astype(int)
        metrics = _classification_metrics(
            y_validation, predictions, None, class_labels, task
        )
    else:
        predictions = estimator.predict(X_validation)
        metrics = _regression_metrics(y_validation, predictions)

    metrics["fit_seconds"] = float(fit_seconds)
    metrics["selection_score"] = _selection_score(task, metrics)
    return estimator, metrics, threshold


def run_corrected_m5(
    split_dir: Path = SPLIT_DIR,
    feature_config_path: Path = FEATURE_CONFIG,
    model_dir: Path = MODEL_DIR,
    report_dir: Path = REPORT_DIR,
    horizons: list[int] | None = None,
    tasks: list[str] | None = None,
) -> list[dict]:
    """Select all M5 models on validation and evaluate frozen winners on test."""
    horizons = horizons or HORIZONS
    tasks = tasks or list(TASKS)
    config = json.loads(feature_config_path.read_text(encoding="utf-8"))
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    card_dir = report_dir / "model_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for horizon in horizons:
        data_path = split_dir / f"horizon_{horizon}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Corrected horizon split not found: {data_path}")
        frame = pd.read_csv(data_path)
        frame["road_id"] = frame["road_id"].astype(str)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        frame = frame.sort_values(["road_id", "timestamp"], kind="stable")
        if frame.duplicated(["road_id", "timestamp"]).any():
            raise ValueError(f"Duplicate road/timestamp rows in {data_path}")
        split_order = {"train": 0, "validation": 1, "test": 2}
        if not frame["split"].isin(split_order).all():
            raise ValueError(f"Invalid split label found in {data_path}")
        for road_id, road in frame.groupby("road_id", sort=False):
            ordered = road["split"].map(split_order).to_numpy()
            if np.any(np.diff(ordered) < 0):
                raise ValueError(f"Chronological split ordering failed for {road_id}")

        for task in tasks:
            if task not in TASKS:
                raise ValueError(f"Unknown M5 task: {task}")
            target_prefix, kind = TASKS[task]
            target = f"{target_prefix}{horizon}"
            features = config["feature_sets"][task]
            forbidden = {
                target,
                f"traffic_volume_t_plus_{horizon}",
                f"travel_time_t_plus_{horizon}",
                f"congestion_level_t_plus_{horizon}",
                f"accident_count_t_plus_{horizon}",
                f"accident_risk_t_plus_{horizon}",
                "event_proximity_flag",
            }
            future_features = [
                feature for feature in features if "_t_plus_" in feature
            ]
            if forbidden.intersection(features) or future_features:
                raise ValueError(
                    f"Forbidden/current/uncertain features present for {task}: "
                    f"{sorted(forbidden.intersection(features) | set(future_features))}"
                )
            missing = sorted((set(features) | {target}) - set(frame.columns))
            if missing:
                raise ValueError(f"Missing corrected modeling columns: {missing}")

            X_raw = frame[features]
            y_raw = frame[target]
            valid = X_raw.notna().all(axis=1) & y_raw.notna()
            masks = {
                split: frame["split"].eq(split) & valid
                for split in ["train", "validation", "test"]
            }
            (
                masks["validation_selection"],
                masks["validation_calibration"],
            ) = _validation_selection_calibration_masks(
                frame, masks["validation"]
            )
            if any(not mask.any() for mask in masks.values()):
                raise ValueError(f"Empty valid partition for {task}/{horizon}m")
            preprocessor = _build_preprocessor(features)
            X_train = preprocessor.fit_transform(X_raw.loc[masks["train"]])
            X_validation = preprocessor.transform(
                X_raw.loc[masks["validation_selection"]]
            )
            X_calibration = preprocessor.transform(
                X_raw.loc[masks["validation_calibration"]]
            )
            X_test = preprocessor.transform(X_raw.loc[masks["test"]])
            if hasattr(X_train, "toarray"):
                X_train = X_train.toarray()
                X_validation = X_validation.toarray()
                X_calibration = X_calibration.toarray()
                X_test = X_test.toarray()

            label_encoder = None
            threshold = None
            if kind == "regression":
                y_train = y_raw.loc[masks["train"]].to_numpy(dtype=float)
                y_validation = y_raw.loc[
                    masks["validation_selection"]
                ].to_numpy(dtype=float)
                y_calibration = y_raw.loc[
                    masks["validation_calibration"]
                ].to_numpy(dtype=float)
                y_test = y_raw.loc[masks["test"]].to_numpy(dtype=float)
                class_labels: list[str] = []
                candidates = _regression_candidates()
            else:
                y_train_values = y_raw.loc[masks["train"]]
                y_validation_values = y_raw.loc[masks["validation_selection"]]
                y_calibration_values = y_raw.loc[
                    masks["validation_calibration"]
                ]
                y_test_values = y_raw.loc[masks["test"]]
                if task == "accident_risk":
                    y_train_text = _binary_label_text(y_train_values)
                    y_validation_text = _binary_label_text(y_validation_values)
                    y_calibration_text = _binary_label_text(y_calibration_values)
                    y_test_text = _binary_label_text(y_test_values)
                else:
                    y_train_text = y_train_values.astype(str)
                    y_validation_text = y_validation_values.astype(str)
                    y_calibration_text = y_calibration_values.astype(str)
                    y_test_text = y_test_values.astype(str)
                label_encoder = LabelEncoder()
                if task == "congestion_level":
                    observed = set(y_train_text.unique())
                    if observed != set(CONGESTION_ORDER):
                        raise ValueError(
                            f"Expected all congestion classes in train: {observed}"
                        )
                    label_encoder.classes_ = np.asarray(CONGESTION_ORDER)
                    y_train = label_encoder.transform(y_train_text)
                else:
                    y_train = label_encoder.fit_transform(y_train_text)
                y_validation = label_encoder.transform(
                    y_validation_text
                )
                y_calibration = label_encoder.transform(y_calibration_text)
                y_test = label_encoder.transform(y_test_text)
                class_labels = [str(value) for value in label_encoder.classes_]
                if task == "congestion_level" and class_labels != CONGESTION_ORDER:
                    raise ValueError(
                        f"Unexpected congestion class set/order: {class_labels}"
                    )
                if task == "accident_risk" and class_labels != ["0", "1"]:
                    raise ValueError(
                        f"Accident-risk train labels must contain both classes: {class_labels}"
                    )
                candidates = _classification_candidates(task, len(class_labels))

            target_counts = None
            if kind == "classification":
                target_counts = {
                    partition: {
                        label: int(np.sum(values == class_index))
                        for class_index, label in enumerate(class_labels)
                    }
                    for partition, values in [
                        ("train", y_train),
                        ("validation_selection", y_validation),
                        ("validation_calibration", y_calibration),
                        ("test", y_test),
                    ]
                }

            validation_results = []
            fitted_candidates = {}
            for name, estimator in candidates.items():
                fitted, metrics, candidate_threshold = _fit_candidate(
                    name,
                    estimator,
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    task,
                    class_labels,
                )
                validation_results.append({
                    "model": name,
                    "metrics": metrics,
                    "threshold": candidate_threshold,
                })
                fitted_candidates[name] = (fitted, candidate_threshold)
                print(
                    f"{horizon:3}m {task:16} {name:24} "
                    f"validation selection={metrics['selection_score']:.5f}",
                    flush=True,
                )
            winner = max(
                validation_results,
                key=lambda item: item["metrics"]["selection_score"],
            )
            selected_model, threshold = fitted_candidates[winner["model"]]
            test_start = time.perf_counter()

            if task == "accident_risk":
                test_scores = _positive_scores(selected_model, X_test)
                test_predictions = (test_scores >= float(threshold)).astype(int)
                test_metrics = _classification_metrics(
                    y_test, test_predictions, test_scores, class_labels, task
                )
                test_metrics["operating_threshold"] = float(threshold)
            elif task == "congestion_level":
                test_predictions = selected_model.predict(X_test).astype(int)
                test_metrics = _classification_metrics(
                    y_test, test_predictions, None, class_labels, task
                )
            else:
                test_predictions = selected_model.predict(X_test)
                test_metrics = _regression_metrics(y_test, test_predictions)
            test_metrics["evaluation_seconds"] = float(time.perf_counter() - test_start)

            candidates_path = None
            if task == "traffic_volume":
                candidates_path = (
                    model_dir / f"traffic_volume_{horizon}m_validation_candidates.joblib"
                )
                joblib.dump({
                    "models": {
                        name: model for name, (model, _) in fitted_candidates.items()
                    },
                    "features": features,
                    "preprocessor": preprocessor,
                    "preprocessing_fit_partition": "train",
                    "training_period": _periods(frame.loc[valid], "train"),
                    "validation_selection_period": {
                        "start": str(frame.loc[masks["validation_selection"], "timestamp"].min()),
                        "end": str(frame.loc[masks["validation_selection"], "timestamp"].max()),
                    },
                }, candidates_path, compress=3)

            conformal_quantile = None
            calibration_period = {
                "start": str(frame.loc[masks["validation_calibration"], "timestamp"].min()),
                "end": str(frame.loc[masks["validation_calibration"], "timestamp"].max()),
            }
            if task in {"traffic_volume", "travel_time"}:
                calibration_predictions = np.maximum(
                    selected_model.predict(X_calibration).reshape(-1), 0
                )
                conformal_quantile = _split_conformal_quantile(
                    np.abs(y_calibration - calibration_predictions)
                )
                test_predictions = np.maximum(
                    selected_model.predict(X_test).reshape(-1), 0
                )
                lower = np.maximum(test_predictions - conformal_quantile, 0)
                upper = test_predictions + conformal_quantile
                test_metrics["prediction_interval_coverage_90"] = float(
                    np.mean((y_test >= lower) & (y_test <= upper))
                )
                test_metrics["prediction_interval_mean_width"] = float(
                    np.mean(upper - lower)
                )

            stem = f"{task}_{horizon}m"
            artifact_path = model_dir / f"{stem}.joblib"
            artifact = {
                "model": selected_model,
                "preprocessor": preprocessor,
                "label_encoder": label_encoder,
                "task": task,
                "target": target,
                "horizon_minutes": horizon,
                "features": features,
                "random_seed": SEED,
                "selection_metric": (
                    "validation PR-AUC"
                    if task == "accident_risk"
                    else "validation macro-F1"
                    if task == "congestion_level"
                    else "validation RMSE (minimized)"
                ),
                "validation_threshold": threshold,
                "conformal_quantile_90": conformal_quantile,
                "conformal_calibration_rows": int(masks["validation_calibration"].sum()),
                "conformal_calibration_period": calibration_period,
                "conformal_nominal_coverage": (
                    INTERVAL_COVERAGE if conformal_quantile is not None else None
                ),
            }
            joblib.dump(artifact, artifact_path, compress=3)

            result = {
                "task": task,
                "horizon_minutes": horizon,
                "target": target,
                "selected_model": winner["model"],
                "selection_metric": artifact["selection_metric"],
                "selected_validation_metrics": winner["metrics"],
                "validation_candidate_results": validation_results,
                "validation_selection_period": {
                    "start": str(frame.loc[masks["validation_selection"], "timestamp"].min()),
                    "end": str(frame.loc[masks["validation_selection"], "timestamp"].max()),
                },
                "validation_calibration_period": calibration_period,
                "validation_calibration_rows": int(
                    masks["validation_calibration"].sum()
                ),
                "target_class_counts": target_counts,
                "test_rows": int(masks["test"].sum()),
                "test_metrics": test_metrics,
                "training_rows": int(masks["train"].sum()),
                "validation_rows": int(masks["validation"].sum()),
                "periods": {
                    split: _periods(frame.loc[valid], split)
                    for split in ["train", "validation", "test"]
                },
                "features": features,
                "validation_candidate_artifact": (
                    str(candidates_path.relative_to(ROOT))
                    if candidates_path is not None
                    else None
                ),
                "preprocessing": (
                    "Median imputation and standard scaling for numeric inputs; "
                    "OneHotEncoder(handle_unknown='ignore') for categories. "
                    "All transformers fit on training rows only."
                ),
                "artifact_path": str(artifact_path.relative_to(ROOT)),
                "test_partition_used_for_selection": False,
                "software_versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__,
                    "xgboost": xgboost.__version__,
                },
                "random_seed": SEED,
            }
            report_path = report_dir / f"{stem}_results.json"
            report_path.write_text(
                json.dumps(result, indent=2, default=_jsonable),
                encoding="utf-8",
            )

            card = {
                "model_name": winner["model"],
                "model_type": kind,
                "target": target,
                "forecast_horizon_minutes": horizon,
                "training_period": result["periods"]["train"],
                "validation_period": result["periods"]["validation"],
                "test_period": result["periods"]["test"],
                "feature_list": features,
                "preprocessing": result["preprocessing"],
                "hyperparameters": {
                    **selected_model.get_params(),
                    "epochs_completed": selected_model.epochs_completed_,
                    "best_validation_mse_scaled":
                        selected_model.best_validation_mse_scaled_,
                }
                if isinstance(selected_model, NumpyGradientRegressor)
                else selected_model.get_params()
                if hasattr(selected_model, "get_params")
                else {
                    "learning_rate": selected_model.learning_rate,
                    "max_epochs": selected_model.max_epochs,
                    "patience": selected_model.patience,
                    "epochs_completed": selected_model.epochs_completed_,
                    "best_validation_mse_scaled":
                        selected_model.best_validation_mse_scaled_,
                },
                "validation_metrics": winner["metrics"],
                "test_metrics": test_metrics,
                "validation_selected_operating_threshold": threshold,
                "uncertainty": (
                    {
                        "method": "Split conformal absolute-residual interval",
                        "nominal_coverage": INTERVAL_COVERAGE,
                        "calibration_rows": int(
                            masks["validation_calibration"].sum()
                        ),
                        "calibration_period": calibration_period,
                        "quantile": conformal_quantile,
                        "observed_test_coverage": test_metrics[
                            "prediction_interval_coverage_90"
                        ],
                        "test_mean_interval_width": test_metrics[
                            "prediction_interval_mean_width"
                        ],
                        "limitations": (
                            "The final chronological validation block is reserved "
                            "for split-conformal calibration and is not used for "
                            "candidate selection. The finite-sample guarantee "
                            "requires exchangeability; serial dependence and "
                            "distribution shift mean nominal coverage is not "
                            "guaranteed for this time series."
                        ),
                    }
                    if conformal_quantile is not None
                    else {
                        "method": "No calibrated probability interval",
                        "limitations": (
                            "Classification scores/thresholds are not calibrated "
                            "uncertainty intervals."
                        ),
                    }
                ),
                "model_artifact_path": str(artifact_path.relative_to(ROOT)),
                "software_versions": result["software_versions"],
                "random_seed": SEED,
                "test_used_for_model_or_threshold_selection": False,
            }
            (card_dir / f"{stem}_model_card.json").write_text(
                json.dumps(card, indent=2, default=_jsonable),
                encoding="utf-8",
            )
            if task == "accident_risk":
                precision, recall, thresholds = precision_recall_curve(
                    y_test, test_scores
                )
                pd.DataFrame({
                    "precision": precision,
                    "recall": recall,
                    "threshold": np.append(thresholds, np.nan),
                }).to_csv(report_dir / f"{stem}_test_precision_recall.csv", index=False)
            all_results.append(result)

    (report_dir / "m5_corrected_test_evaluation.json").write_text(
        json.dumps(all_results, indent=2, default=_jsonable),
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "task": item["task"],
            "horizon_minutes": item["horizon_minutes"],
            "model": item["selected_model"],
            "test_rows": item["test_rows"],
            **item["test_metrics"],
        }
        for item in all_results
    ]).to_csv(report_dir / "m5_corrected_test_scoreboard.csv", index=False)
    return all_results


if __name__ == "__main__":
    results = run_corrected_m5()
    print(f"Corrected M5 completed {len(results)} target/horizon evaluations.")
