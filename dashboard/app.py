"""Streamlit analytics and forecast interface for FlowCast."""
from __future__ import annotations

import json
from io import BytesIO
from datetime import time, timedelta
from pathlib import Path
import time as clock

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.ingest import EXPECTED, _validate_frame
from src.upload_inference import (
    predict_latest_uploaded_rows,
    process_uploaded_sources,
)

ROOT = Path(__file__).resolve().parents[1]
LATEST_MANIFEST = ROOT / "reports/corrected_v2/latest.json"

HORIZONS = [30, 60, 90, 120]
CATEGORICAL = [
    "road_id",
    "weather_condition",
    "temperature_band",
    "congestion_level",
]
TASKS = {
    "traffic_volume": {
        "target_prefix": "traffic_volume_t_plus_",
        "kind": "regression",
    },
    "travel_time": {
        "target_prefix": "travel_time_t_plus_",
        "kind": "regression",
    },
    "congestion_level": {
        "target_prefix": "congestion_level_t_plus_",
        "kind": "classification",
    },
    "accident_risk": {
        "target_prefix": "accident_risk_t_plus_",
        "kind": "classification",
    },
}
CONGESTION_ORDER = ["Free-flow", "Moderate", "Heavy", "Severe"]
CONGESTION_COLOR = {
    "Free-flow": "#2ca25f",
    "Moderate": "#fdd835",
    "Heavy": "#fb8c00",
    "Severe": "#c62828",
}


@st.cache_data
def load_run_manifest() -> dict:
    if not LATEST_MANIFEST.exists():
        raise FileNotFoundError(
            "Corrected FlowCast run not found. Run `python -m src.run_pipeline` first."
        )
    latest = json.loads(LATEST_MANIFEST.read_text(encoding="utf-8"))
    manifest_path = ROOT / latest["manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Corrected run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"Latest corrected run is not complete: {manifest.get('status')}")
    return manifest


@st.cache_data
def load_features() -> pd.DataFrame:
    manifest = load_run_manifest()
    feature_file = ROOT / manifest["paths"]["processed_dir"] / "flowcast_m3_features.csv"
    if not feature_file.exists():
        raise FileNotFoundError(f"Processed feature data not found: {feature_file}")
    frame = pd.read_csv(feature_file)
    frame["timestamp"] = pd.to_datetime(
        frame["date"].astype(str) + " " + frame["time"].astype(str),
        errors="raise",
    )
    frame["road_id"] = frame["road_id"].astype(str)
    return frame.sort_values(["road_id", "timestamp"], kind="stable")


@st.cache_data
def load_feature_config() -> dict:
    manifest = load_run_manifest()
    feature_config = ROOT / manifest["paths"]["report_dir"] / "m5_feature_selection.json"
    if not feature_config.exists():
        raise FileNotFoundError(f"Feature configuration not found: {feature_config}")
    return json.loads(feature_config.read_text(encoding="utf-8"))


@st.cache_data
def load_test_scoreboard() -> pd.DataFrame:
    manifest = load_run_manifest()
    path = ROOT / manifest["paths"]["report_dir"] / "m5_corrected_test_scoreboard.csv"
    if not path.exists():
        raise FileNotFoundError(f"Saved M5 test scorecard not found: {path}")
    return pd.read_csv(path)


@st.cache_resource
def load_model_bundle(task: str, horizon: int) -> dict:
    if task not in TASKS or horizon not in HORIZONS:
        raise ValueError(f"Unsupported task/horizon: {task}/{horizon}")

    manifest = load_run_manifest()
    split_file = (
        ROOT / manifest["paths"]["m5_split_dir"] / f"horizon_{horizon}.csv"
    )
    if not split_file.exists():
        raise FileNotFoundError(f"Chronological split not found: {split_file}")
    frame = pd.read_csv(split_file)
    frame["road_id"] = frame["road_id"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    features = load_feature_config()["feature_sets"][task]
    target = f"{TASKS[task]['target_prefix']}{horizon}"
    missing = sorted((set(features) | {target, "split"}) - set(frame.columns))
    if missing:
        raise ValueError(f"Split file is missing required model columns: {missing}")

    X = frame[features].copy()
    y = frame[target].copy()
    split = frame["split"]
    valid = X.notna().all(axis=1) & y.notna()
    train_mask = split.eq("train") & valid
    validation_mask = split.eq("validation") & valid
    test_mask = split.eq("test") & valid
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError(f"Incomplete chronological partitions for {task} at {horizon}m.")

    categorical = [column for column in CATEGORICAL if column in features]
    numeric = [column for column in features if column not in categorical]
    preprocessor = ColumnTransformer(
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
    preprocessor.fit(X.loc[train_mask])

    artifact = (
        ROOT / manifest["paths"]["m5_model_dir"] / f"{task}_{horizon}m.joblib"
    )
    if not artifact.exists():
        raise FileNotFoundError(f"Selected model artifact not found: {artifact}")
    saved = joblib.load(artifact)
    label_encoder = None
    saved_preprocessor = None
    model = saved
    if isinstance(saved, dict):
        if "model" not in saved:
            raise ValueError(f"Model artifact has no 'model' entry: {artifact}")
        model = saved["model"]
        label_encoder = saved.get("label_encoder")
        saved_preprocessor = saved.get("preprocessor")
    if saved_preprocessor is not None:
        preprocessor = saved_preprocessor

    model_card_path = (
        ROOT
        / manifest["paths"]["report_dir"]
        / "model_cards"
        / f"{task}_{horizon}m_model_card.json"
    )
    if not model_card_path.exists():
        raise FileNotFoundError(f"Selected model card not found: {model_card_path}")
    model_card = json.loads(model_card_path.read_text(encoding="utf-8"))
    return {
        "task": task,
        "horizon": horizon,
        "target": target,
        "features": features,
        "frame": frame,
        "X": X,
        "y": y,
        "valid": valid,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask,
        "preprocessor": preprocessor,
        "model": model,
        "label_encoder": label_encoder,
        "validation_threshold": (
            saved.get("validation_threshold")
            if isinstance(saved, dict)
            else None
        ),
        "conformal_quantile": (
            saved.get("conformal_quantile_90")
            if isinstance(saved, dict)
            else None
        ),
        "conformal_calibration_rows": (
            saved.get("conformal_calibration_rows")
            if isinstance(saved, dict)
            else None
        ),
        "conformal_calibration_period": (
            saved.get("conformal_calibration_period")
            if isinstance(saved, dict)
            else None
        ),
        "model_card": model_card,
        "artifact": artifact,
    }


def decode_predictions(bundle: dict, predictions: np.ndarray) -> np.ndarray:
    if bundle["task"] == "congestion_level" and bundle["label_encoder"] is not None:
        return bundle["label_encoder"].inverse_transform(
            np.asarray(predictions, dtype=int)
        )
    return predictions


def conformal_interval_radius(bundle: dict) -> float:
    if TASKS[bundle["task"]]["kind"] != "regression":
        raise ValueError("Residual intervals are only defined for regression models.")
    radius = bundle["conformal_quantile"]
    if radius is None or not np.isfinite(radius):
        raise ValueError(
            "Selected regression artifact has no finite split-conformal calibration quantile."
        )
    return float(radius)


def train_experiment(task: str, horizon: int, algorithm: str) -> dict:
    bundle = load_model_bundle(task, horizon)
    train_mask = bundle["train_mask"]
    validation_mask = bundle["validation_mask"]
    preprocessor = bundle["preprocessor"]
    x_train = preprocessor.transform(bundle["X"].loc[train_mask])
    x_validation = preprocessor.transform(bundle["X"].loc[validation_mask])
    y_train = bundle["y"].loc[train_mask]
    y_validation = bundle["y"].loc[validation_mask]
    label_encoder = None

    if TASKS[task]["kind"] == "regression":
        estimators = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(
                max_depth=16, min_samples_leaf=5, random_state=42
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42
            ),
            "Support Vector Regression": SVR(C=1.0, epsilon=0.1),
        }
        y_train_values = y_train.to_numpy(dtype=float)
        y_validation_values = y_validation.to_numpy(dtype=float)
    elif task == "congestion_level":
        label_encoder = LabelEncoder()
        y_train_values = label_encoder.fit_transform(y_train.astype(str))
        y_validation_values = label_encoder.transform(y_validation.astype(str))
        estimators = {
            "Decision Tree": DecisionTreeClassifier(
                max_depth=16, min_samples_leaf=5, class_weight="balanced",
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                n_jobs=-1, random_state=42,
            ),
            "Support Vector Classifier": SVC(
                class_weight="balanced", probability=True, random_state=42
            ),
        }
    else:
        y_train_values = y_train.to_numpy(dtype=int)
        y_validation_values = y_validation.to_numpy(dtype=int)
        estimators = {
            "Decision Tree": DecisionTreeClassifier(
                max_depth=16, min_samples_leaf=5, class_weight="balanced",
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                n_jobs=-1, random_state=42,
            ),
            "Support Vector Classifier": SVC(
                class_weight="balanced", probability=True, random_state=42
            ),
        }

    if algorithm not in estimators:
        raise ValueError(f"Unsupported experiment algorithm: {algorithm}")
    model = estimators[algorithm]
    model.fit(x_train, y_train_values)
    predictions = model.predict(x_validation)

    if TASKS[task]["kind"] == "regression":
        predictions = np.maximum(predictions, 0)
        metrics = {
            "MAE": float(mean_absolute_error(y_validation_values, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_validation_values, predictions))),
            "R2": float(r2_score(y_validation_values, predictions)),
        }
    else:
        metrics = {
            "accuracy": float(accuracy_score(y_validation_values, predictions)),
            "macro_f1": float(f1_score(
                y_validation_values, predictions, average="macro", zero_division=0
            )),
            "macro_precision": float(precision_score(
                y_validation_values, predictions, average="macro", zero_division=0
            )),
            "macro_recall": float(recall_score(
                y_validation_values, predictions, average="macro", zero_division=0
            )),
        }
        if task == "accident_risk":
            scores = model.predict_proba(x_validation)[:, 1]
            metrics["ROC_AUC"] = float(
                roc_auc_score(y_validation_values, scores)
            )
            metrics["PR_AUC"] = float(
                average_precision_score(y_validation_values, scores)
            )
        if label_encoder is not None:
            metrics["class_labels"] = label_encoder.classes_.tolist()

    stamp = str(clock.time_ns())
    stem = f"{task}_{horizon}m_{algorithm.lower().replace(' ', '_')}_{stamp}"
    manifest = load_run_manifest()
    experiment_dir = ROOT / manifest["paths"]["m5_model_dir"] / "experimental"
    report_dir = ROOT / manifest["paths"]["report_dir"] / "dashboard_experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = experiment_dir / f"{stem}.joblib"
    report_path = report_dir / f"{stem}.json"
    joblib.dump({
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "task": task,
        "horizon_minutes": horizon,
        "features": bundle["features"],
        "target": bundle["target"],
    }, model_path)
    result = {
        "task": task,
        "horizon_minutes": horizon,
        "algorithm": algorithm,
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_used": False,
        "metrics": metrics,
        "model_artifact": str(model_path.relative_to(ROOT)),
    }
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["report"] = str(report_path.relative_to(ROOT))
    return result


def apply_global_filters(frame: pd.DataFrame) -> pd.DataFrame:
    selected_roads = st.session_state["selected_roads"]
    start_date, end_date = st.session_state["selected_dates"]
    start_time, end_time = st.session_state["selected_times"]
    mask = (
        frame["road_id"].isin(selected_roads)
        & frame["timestamp"].dt.date.between(start_date, end_date)
        & frame["timestamp"].dt.time.between(start_time, end_time)
    )
    return frame.loc[mask].copy()


def build_selected_range_report(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# FlowCast selected-range report\n\nNo records match the selected filters.\n"

    start = frame["timestamp"].min()
    end = frame["timestamp"].max()
    lines = [
        "# FlowCast selected-range insights",
        "",
        f"- Window: {start} through {end}",
        f"- Observations: {len(frame):,}",
        f"- Road segments: {frame['road_id'].nunique():,}",
        "",
        "## Corridor summary",
        "",
    ]
    for column, label in [
        ("traffic_volume", "Mean traffic volume"),
        ("avg_speed", "Mean speed"),
        ("travel_time", "Mean travel time"),
        ("rainfall", "Mean rainfall"),
    ]:
        if column in frame:
            value = pd.to_numeric(frame[column], errors="coerce").mean()
            lines.append(f"- {label}: {value:.3f}" if pd.notna(value) else f"- {label}: unavailable")

    if "congestion_level" in frame:
        congestion = frame["congestion_level"].value_counts(normalize=True, dropna=True)
        lines.extend(["", "## Congestion distribution", ""])
        lines.extend(
            f"- {label}: {share:.1%}"
            for label, share in congestion.reindex(CONGESTION_ORDER, fill_value=0).items()
        )

    if {"road_id", "traffic_volume"}.issubset(frame.columns):
        segment_summary = (
            frame.groupby("road_id", as_index=False)["traffic_volume"]
            .mean()
            .sort_values("traffic_volume", ascending=False)
            .head(5)
        )
        lines.extend(["", "## Highest mean-volume segments", ""])
        lines.extend(
            f"- {row.road_id}: {row.traffic_volume:.3f}"
            for row in segment_summary.itertuples(index=False)
        )
    return "\n".join(lines) + "\n"


def render_live_predictions(frame: pd.DataFrame, horizon: int) -> None:
    with st.spinner("Loading persisted models and preparing latest segment inputs..."):
        latest = (
            frame.sort_values("timestamp")
            .groupby("road_id", sort=False)
            .tail(1)
            .loc[lambda rows: rows["road_id"].isin(
                st.session_state["selected_roads"]
            )]
            .copy()
        )
        if latest.empty:
            st.info("Select at least one road segment to generate forecasts.")
            return
        output = latest[["road_id", "timestamp"]].rename(
            columns={"timestamp": "as_of"}
        )
        manifest = load_run_manifest()
        output["model_run_id"] = manifest["run_id"]

        for task in TASKS:
            bundle = load_model_bundle(task, horizon)
            raw_x = latest[bundle["features"]]
            complete = raw_x.notna().all(axis=1)
            if not complete.any():
                st.warning(f"No complete latest records are available for {task}.")
                continue
            transformed = bundle["preprocessor"].transform(raw_x.loc[complete])
            predictions = bundle["model"].predict(transformed).reshape(-1)
            scores = None
            score_name = ""
            if task == "accident_risk":
                model = bundle["model"]
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(transformed)[:, 1]
                    score_name = "accident_risk_score"
                    st.caption(
                        "Accident score is the selected model's positive-class "
                        "probability; it is not independently calibrated."
                    )
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(transformed).reshape(-1)
                    score_name = "accident_risk_decision_score"
                    st.caption(
                        "This SVM exposes a decision score, not a calibrated probability."
                    )
                else:
                    raise TypeError(
                        f"Selected accident-risk model has no score method: {type(model).__name__}"
                    )
                threshold = bundle["validation_threshold"]
                if threshold is None:
                    raise ValueError(
                        "Selected accident-risk artifact has no validation threshold."
                    )
                predictions = (scores >= float(threshold)).astype(int)
            elif TASKS[task]["kind"] == "regression":
                predictions = np.maximum(predictions, 0)
            predictions = decode_predictions(bundle, predictions)
            task_output = pd.DataFrame({
                "road_id": latest.loc[complete, "road_id"].to_numpy(),
                task: predictions,
                f"{task}_model_artifact": str(
                    bundle["artifact"].relative_to(ROOT)
                ),
            })
            if (
                TASKS[task]["kind"] == "classification"
                and hasattr(bundle["model"], "predict_proba")
            ):
                task_probabilities = bundle["model"].predict_proba(transformed)
                task_output[f"{task}_confidence"] = task_probabilities.max(axis=1)
            output = output.merge(task_output, on="road_id", how="left", validate="one_to_one")

            if task == "accident_risk":
                if scores is not None:
                    score_frame = pd.DataFrame({
                        "road_id": latest.loc[complete, "road_id"].to_numpy(),
                        score_name: scores,
                    })
                    output = output.merge(
                        score_frame, on="road_id", how="left", validate="one_to_one"
                    )

            if TASKS[task]["kind"] == "regression":
                radius = conformal_interval_radius(bundle)
                interval_frame = pd.DataFrame({
                    "road_id": latest.loc[complete, "road_id"].to_numpy(),
                    f"{task}_lower_90": np.maximum(predictions - radius, 0),
                    f"{task}_upper_90": predictions + radius,
                })
                output = output.merge(
                    interval_frame, on="road_id", how="left", validate="one_to_one"
                )

    st.dataframe(output, use_container_width=True, hide_index=True)
    st.caption(
        "Latest observed records are inputs; forecasts are model outputs for the "
        "selected horizon. Regression intervals use a later validation block "
        "reserved for split-conformal calibration; time dependence and shift "
        "limit coverage guarantees. "
        "Classification confidence scores are not independently calibrated."
    )
    st.download_button(
        "Download predictions CSV",
        output.to_csv(index=False).encode("utf-8"),
        file_name=f"flowcast_predictions_{horizon}m.csv",
        mime="text/csv",
    )


def render_forecast_plot(task: str, horizon: int) -> None:
    bundle = load_model_bundle(task, horizon)
    selected = bundle["frame"].loc[bundle["test_mask"]].copy()
    selected = selected.loc[selected["road_id"].isin(st.session_state["selected_roads"])]
    start_date, end_date = st.session_state["selected_dates"]
    start_time, end_time = st.session_state["selected_times"]
    selected = selected.loc[
        selected["timestamp"].dt.date.between(start_date, end_date)
        & selected["timestamp"].dt.time.between(start_time, end_time)
    ]
    if selected.empty:
        st.info("No test records match the selected segments.")
        return
    x = bundle["X"].loc[selected.index]
    transformed = bundle["preprocessor"].transform(x)
    predictions = bundle["model"].predict(transformed).reshape(-1)
    if TASKS[task]["kind"] == "regression":
        predictions = np.maximum(predictions, 0)
    predictions = decode_predictions(bundle, predictions)
    selected["prediction"] = predictions
    selected["actual"] = bundle["y"].loc[selected.index].to_numpy()
    if len(selected) > 2500:
        selected = selected.iloc[:: max(1, len(selected) // 2500)]
    long = selected.melt(
        id_vars=["timestamp", "road_id"],
        value_vars=["actual", "prediction"],
        var_name="series",
        value_name="value",
    )
    fig = px.line(
        long,
        x="timestamp",
        y="value",
        color="series",
        facet_row="road_id" if len(st.session_state["selected_roads"]) <= 4 else None,
        title=f"{task.replace('_', ' ').title()} — {horizon}-minute held-out test",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Predicted and actual values are from the same chronological test rows.")


def main() -> None:
    st.set_page_config(
        page_title="FlowCast | Traffic Forecasting",
        page_icon="🚦",
        layout="wide",
    )
    st.title("FlowCast — Smart Traffic Flow")
    st.caption("Northline Corridor · segment-level traffic forecasting and analytics")

    try:
        data = load_features()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to load processed FlowCast data: {error}")
        st.stop()

    min_date = data["timestamp"].min().date()
    max_date = data["timestamp"].max().date()
    default_start = max(min_date, max_date - pd.Timedelta(days=30))
    with st.sidebar:
        st.header("Filters")
        selected_roads = st.multiselect(
            "Road segments",
            sorted(data["road_id"].unique()),
            default=sorted(data["road_id"].unique()),
        )
        date_range = st.date_input(
            "Date range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            st.error("Choose both a start and end date.")
            st.stop()
        selected_times = st.slider(
            "Time of day",
            min_value=time(0, 0),
            max_value=time(23, 30),
            value=(time(0, 0), time(23, 30)),
            step=timedelta(minutes=30),
            format="HH:mm",
        )
        st.session_state["selected_roads"] = selected_roads
        st.session_state["selected_dates"] = date_range
        st.session_state["selected_times"] = selected_times
        horizon = st.selectbox("Forecast horizon", HORIZONS, index=0)

    filtered = apply_global_filters(data)
    if filtered.empty:
        st.warning("No rows match the selected segment and date filters.")
    st.sidebar.download_button(
        "Download selected-range insights",
        build_selected_range_report(filtered),
        file_name="flowcast_selected_range_insights.md",
        mime="text/markdown",
    )

    pages = [
        "Live Predictions",
        "Historical Trends",
        "Congestion Heatmap",
        "Road Comparison",
        "Model Performance",
        "Feature Importance",
        "Forecast Visualization",
        "Prediction Confidence",
        "Weather vs Traffic",
        "Data Validation",
        "Model Training",
    ]
    page = st.sidebar.radio("View", pages)
    if filtered.empty and page not in {"Live Predictions", "Data Validation", "Model Training"}:
        st.info("No records match the selected segment, date, and time filters.")
        st.stop()

    if page == "Live Predictions":
        st.subheader(f"Latest predictions · {horizon} minutes")
        st.caption(
            f"Latest stored observations end at {data['timestamp'].max()}. "
            "This dataset has no live sensor feed; forecasts are based on the "
            "latest historical records."
        )
        render_live_predictions(data, horizon)

    elif page == "Historical Trends":
        st.subheader("Historical traffic patterns")
        if filtered.empty:
            st.info("No records match the selected filters.")
        else:
            segment = st.selectbox("Trend segment", sorted(filtered["road_id"].unique()))
            road = filtered.loc[filtered["road_id"].eq(segment)]
            metric = st.selectbox(
                "Measure",
                ["traffic_volume", "avg_speed", "travel_time"],
            )
            fig = px.line(
                road,
                x="timestamp",
                y=metric,
                title=f"{metric.replace('_', ' ').title()} — {segment}",
            )
            st.plotly_chart(fig, use_container_width=True)
            hourly = road.groupby("hour", as_index=False)["traffic_volume"].mean()
            st.plotly_chart(
                px.line(hourly, x="hour", y="traffic_volume", markers=True,
                        title="Mean volume by hour"),
                use_container_width=True,
            )

    elif page == "Congestion Heatmap":
        st.subheader("Observed congestion by road and time of day")
        if "congestion_level" not in filtered:
            st.error("Congestion labels are not available in the feature dataset.")
        else:
            frame = filtered.dropna(subset=["congestion_level"]).copy()
            frame["severity"] = frame["congestion_level"].map(
                {name: index for index, name in enumerate(CONGESTION_ORDER)}
            )
            frame["time_window"] = frame["timestamp"].dt.strftime("%H:%M")
            pivot = frame.pivot_table(
                index="road_id", columns="time_window", values="severity", aggfunc="mean"
            )
            fig = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale=[
                    CONGESTION_COLOR[name] for name in CONGESTION_ORDER
                ],
                zmin=0,
                zmax=3,
                labels={"x": "30-minute window", "y": "Road segment", "color": "Severity"},
                title="Mean observed congestion severity",
            )
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Road Comparison":
        st.subheader("Road segment comparison")
        comparison = (
            filtered.groupby(["road_id", "road_name"], as_index=False)
            .agg(
                mean_volume=("traffic_volume", "mean"),
                mean_speed=("avg_speed", "mean"),
                volume_std=("traffic_volume", "std"),
            )
        )
        comparison["volume_reliability"] = (
            comparison["volume_std"] / comparison["mean_volume"].replace(0, np.nan)
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.scatter(
                comparison,
                x="mean_volume",
                y="mean_speed",
                size="volume_std",
                color="road_id",
                hover_name="road_name",
                title="Mean traffic and speed by segment",
            ),
            use_container_width=True,
        )

    elif page == "Model Performance":
        st.subheader("Saved chronological test performance")
        try:
            scoreboard = load_test_scoreboard()
            st.dataframe(scoreboard, use_container_width=True, hide_index=True)
            manifest = load_run_manifest()
            m6_path = ROOT / manifest["paths"]["report_dir"] / "m6_corrected_head_to_head.csv"
            if m6_path.exists():
                st.markdown("#### Classical vs deep-learning matched comparison")
                st.dataframe(pd.read_csv(m6_path), use_container_width=True,
                             hide_index=True)
            st.download_button(
                "Download M5 test scorecard",
                scoreboard.to_csv(index=False).encode("utf-8"),
                file_name="m5_test_scoreboard.csv",
                mime="text/csv",
            )
        except (FileNotFoundError, pd.errors.ParserError) as error:
            st.error(f"Saved model metrics could not be loaded: {error}")

    elif page == "Feature Importance":
        st.subheader("Model-derived feature importance")
        task = st.selectbox(
            "Tree-model target",
            ["traffic_volume", "travel_time", "congestion_level"],
        )
        bundle = load_model_bundle(task, horizon)
        model = bundle["model"]
        if not hasattr(model, "feature_importances_"):
            st.info("The selected saved estimator does not expose tree feature importance.")
        else:
            names = bundle["preprocessor"].get_feature_names_out()
            values = np.asarray(model.feature_importances_)
            if len(names) != len(values):
                raise ValueError("Feature-importance names do not match model inputs.")
            importance = pd.DataFrame({
                "feature": names,
                "importance": values,
            }).nlargest(25, "importance")
            st.plotly_chart(
                px.bar(
                    importance.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top model-derived features",
                ),
                use_container_width=True,
            )
        st.caption("Importance is model-derived association, not causal evidence.")

    elif page == "Forecast Visualization":
        st.subheader("Held-out forecast visualization")
        task = st.selectbox("Forecast target", list(TASKS))
        render_forecast_plot(task, horizon)

    elif page == "Prediction Confidence":
        st.subheader("Split-conformal prediction intervals")
        task = st.selectbox(
            "Regression target",
            ["traffic_volume", "travel_time"],
        )
        bundle = load_model_bundle(task, horizon)
        radius = conformal_interval_radius(bundle)
        interval = bundle["model_card"]["uncertainty"]
        st.metric(
            "90% split-conformal residual quantile",
            f"{radius:.3f}",
            help=(
                f"Calibrated on {interval['calibration_rows']:,} later validation "
                "rows, separate from model selection."
            ),
        )
        st.caption(
            f"Calibration period: {interval['calibration_period']['start']} "
            f"through {interval['calibration_period']['end']}."
        )
        st.write(
            f"Observed untouched-test coverage: "
            f"{interval['observed_test_coverage']:.1%}; this is a report-only "
            "diagnostic and was not used to select or tune the model."
        )
        st.write(interval["limitations"])

    elif page == "Weather vs Traffic":
        st.subheader("Weather and observed traffic")
        if "weather_condition" not in filtered:
            st.error("Weather fields are not available in the feature dataset.")
        else:
            summary = (
                filtered.groupby("weather_condition", dropna=False, as_index=False)
                .agg(
                    mean_volume=("traffic_volume", "mean"),
                    median_volume=("traffic_volume", "median"),
                    mean_rainfall=("rainfall", "mean"),
                    mean_visibility=("visibility", "mean"),
                    observations=("traffic_volume", "count"),
                )
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.plotly_chart(
                px.scatter(
                    filtered,
                    x="rainfall",
                    y="traffic_volume",
                    color="weather_condition",
                    hover_data=["road_id", "timestamp", "visibility"],
                    title="Rainfall versus observed traffic volume",
                ),
                use_container_width=True,
            )

    elif page == "Data Validation":
        st.subheader("Validate, process, and predict from uploaded sources")
        st.caption(
            "Upload the traffic CSV. Weather and calendar CSVs are optional; "
            "when omitted, the current project source files are used. Use the "
            "expected FlowCast source filenames."
        )
        uploads = st.file_uploader(
            "Upload FlowCast source CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )
        upload_map = {}
        if uploads:
            for uploaded in uploads:
                if uploaded.name not in EXPECTED:
                    st.error(f"Unsupported source filename: {uploaded.name}")
                    continue
                if uploaded.name in upload_map:
                    st.error(f"Duplicate uploaded source filename: {uploaded.name}")
                    continue
                content = uploaded.getvalue()
                try:
                    uploaded_frame = pd.read_csv(BytesIO(content))
                except (pd.errors.ParserError, UnicodeDecodeError) as error:
                    st.error(f"Could not read {uploaded.name} as a CSV: {error}")
                    continue
                upload_map[uploaded.name] = content
                result, issues, quarantine = _validate_frame(
                    uploaded_frame, uploaded.name
                )
                st.markdown(f"#### {uploaded.name}")
                st.json(result)
                st.dataframe(
                    pd.DataFrame(issues),
                    use_container_width=True,
                    hide_index=True,
                )
                if not quarantine.empty:
                    st.download_button(
                        f"Download {Path(uploaded.name).stem} quarantine",
                        quarantine.to_csv(index=False).encode("utf-8"),
                        file_name=f"{Path(uploaded.name).stem}_quarantine.csv",
                        mime="text/csv",
                        key=f"upload-quarantine-{uploaded.name}",
                    )

        upload_task = st.selectbox(
            "Prediction target",
            list(TASKS),
            key="upload_prediction_task",
        )
        if upload_map and "traffic_sensor_log.csv" not in upload_map:
            st.warning("Upload traffic_sensor_log.csv to enable prediction processing.")
        if st.button(
            "Process uploaded data and predict",
            disabled="traffic_sensor_log.csv" not in upload_map,
            key="process_upload_prediction",
        ):
            try:
                with st.spinner(
                    "Validating and processing uploaded sources through M1–M3..."
                ):
                    processed_upload = process_uploaded_sources(upload_map)
                    uploaded_features = pd.read_csv(
                        processed_upload["feature_path"]
                    )
                    uploaded_predictions, skipped_rows = (
                        predict_latest_uploaded_rows(
                            load_model_bundle(upload_task, horizon),
                            uploaded_features,
                            upload_task,
                            horizon,
                            load_run_manifest()["run_id"],
                        )
                    )
            except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
                st.error(f"Upload processing or prediction failed: {error}")
            else:
                st.success(
                    f"Completed upload run {processed_upload['manifest']['run_id']}. "
                    f"Skipped {skipped_rows} roads with no feature-complete upload record."
                )
                st.json({
                    "processed_rows": processed_upload["m3"]["output_rows"],
                    "unmatched_weather_rows": processed_upload["m2"]["merge"][
                        "unmatched_weather_rows"
                    ],
                    "unmatched_calendar_rows": processed_upload["m2"]["merge"][
                        "unmatched_calendar_rows"
                    ],
                    "fallback_sources": (
                        processed_upload["manifest"][
                            "fallback_files_from_project_raw_data"
                        ]
                    ),
                    "feature_file": processed_upload["manifest"]["feature_file"],
                })
                st.dataframe(
                    uploaded_predictions,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download uploaded-data predictions",
                    uploaded_predictions.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        f"flowcast_uploaded_{upload_task}_{horizon}m_predictions.csv"
                    ),
                    mime="text/csv",
                )

    elif page == "Model Training":
        st.subheader("Train a non-production experiment")
        st.warning(
            "Experiments fit on the chronological training partition and are "
            "scored on validation only. The held-out test partition is not used. "
            "New artifacts are saved under this corrected run's experimental "
            "directory and do not replace selected models."
        )
        task = st.selectbox("Experiment target", list(TASKS))
        if TASKS[task]["kind"] == "regression":
            algorithms = [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Support Vector Regression",
            ]
        else:
            algorithms = [
                "Decision Tree",
                "Random Forest",
                "Support Vector Classifier",
            ]
        algorithm = st.selectbox("Algorithm", algorithms)
        if algorithm in {"Random Forest", "Support Vector Classifier",
                         "Support Vector Regression"}:
            st.caption(
                "Training time depends on the selected estimator and available CPU."
            )
        if st.button("Train and evaluate on validation", type="primary"):
            with st.spinner("Training the selected experiment..."):
                result = train_experiment(task, horizon, algorithm)
            st.success("Validation experiment completed; the test set was not used.")
            st.json(result)

    st.divider()
    st.caption(
        "FlowCast is a decision-support tool. Historical and forecast values "
        "are not instructions for direct traffic-signal control."
    )


if __name__ == "__main__":
    main()
