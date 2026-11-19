"""Export feature importance for selected models in a corrected FlowCast run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.m5_corrected_experiments import _validation_selection_calibration_masks

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "traffic_volume",
    "travel_time",
    "congestion_level",
    "accident_risk",
)
HORIZONS = (30, 60, 90, 120)


def _load_run(
    run_id: str | None,
    allow_in_progress: bool = False,
) -> tuple[str, dict, Path]:
    latest_path = ROOT / "reports/corrected_v2/latest.json"
    if run_id is None:
        if not latest_path.is_file():
            raise FileNotFoundError(
                f"Corrected-run pointer not found: {latest_path}. "
                "Run `python -m src.run_pipeline` first."
            )
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        run_id = latest["run_id"]
        manifest_path = ROOT / latest["manifest"]
    else:
        manifest_path = ROOT / "reports/corrected_v2" / run_id / "run_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Corrected-run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_statuses = {"completed"}
    if allow_in_progress:
        allowed_statuses.add("running")
    if (
        manifest.get("run_id") != run_id
        or manifest.get("status") not in allowed_statuses
    ):
        raise ValueError(f"Run {run_id!r} is not a completed corrected pipeline run.")
    report_dir = ROOT / manifest["paths"]["report_dir"]
    return run_id, manifest, report_dir


def _source_feature(encoded_name: str, features: list[str]) -> str:
    for feature in sorted(features, key=len, reverse=True):
        if encoded_name == f"numeric__{feature}":
            return feature
        if encoded_name == f"categorical__{feature}" or encoded_name.startswith(
            f"categorical__{feature}_"
        ):
            return feature
    raise ValueError(
        f"Encoded feature {encoded_name!r} does not map to a saved source feature."
    )


def _feature_importance_for_model(
    *,
    run_id: str,
    task: str,
    horizon: int,
    report_dir: Path,
    model_dir: Path,
    split_dir: Path,
) -> pd.DataFrame:
    result_path = report_dir / f"{task}_{horizon}m_results.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Selected-model report not found: {result_path}")
    selected = json.loads(result_path.read_text(encoding="utf-8"))
    artifact_path = ROOT / selected["artifact_path"]
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Selected model artifact not found: {artifact_path}")
    if not artifact_path.is_relative_to(model_dir):
        raise ValueError(
            f"Selected model artifact is outside the run model directory: {artifact_path}"
        )

    saved = joblib.load(artifact_path)
    if not isinstance(saved, dict) or not {"model", "preprocessor", "features"} <= saved.keys():
        raise ValueError(f"Selected artifact has an unexpected schema: {artifact_path}")
    model = saved["model"]
    preprocessor = saved["preprocessor"]
    features = list(saved["features"])
    importances = getattr(model, "feature_importances_", None)
    encoded_names = list(preprocessor.get_feature_names_out())
    if importances is not None:
        values = np.asarray(importances, dtype=float).reshape(-1)
        importance_method = "model-native tree importance"
    else:
        coefficients = getattr(model, "coef_", None)
        if coefficients is not None:
            values = np.abs(np.asarray(coefficients, dtype=float))
            if values.ndim == 2:
                values = values.mean(axis=0)
            values = values.reshape(-1)
            importance_method = "mean absolute linear coefficient"
        else:
            split_file = split_dir / f"horizon_{horizon}.csv"
            if not split_file.is_file():
                raise FileNotFoundError(
                    f"Validation split required for permutation importance: {split_file}"
                )
            frame = pd.read_csv(split_file)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
            valid = (
                frame["split"].eq("validation")
                & frame[features].notna().all(axis=1)
                & frame[selected["target"]].notna()
            )
            selection_mask, _ = _validation_selection_calibration_masks(
                frame,
                valid,
            )
            validation_features = preprocessor.transform(
                frame.loc[selection_mask, features]
            )
            if hasattr(validation_features, "toarray"):
                validation_features = validation_features.toarray()
            validation_target = frame.loc[selection_mask, selected["target"]]
            if task == "congestion_level":
                validation_target = saved["label_encoder"].transform(
                    validation_target.astype(str)
                )
                scoring = "f1_macro"
            elif task == "accident_risk":
                validation_target = pd.to_numeric(
                    validation_target, errors="raise"
                ).astype(int)
                scoring = "average_precision"
            else:
                validation_target = pd.to_numeric(
                    validation_target, errors="raise"
                ).to_numpy(dtype=float)
                scoring = "neg_root_mean_squared_error"
            result = permutation_importance(
                model,
                validation_features,
                validation_target,
                scoring=scoring,
                n_repeats=5,
                random_state=42,
                n_jobs=2,
            )
            values = np.abs(result.importances_mean)
            importance_method = (
                "absolute validation permutation importance "
                f"({scoring}, 5 repeats)"
            )
    if len(values) != len(encoded_names):
        raise ValueError(
            f"Feature mismatch for {artifact_path.name}: model has {len(values)} "
            f"importance values but preprocessor has {len(encoded_names)} outputs."
        )
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError(f"Invalid feature-importance values in {artifact_path}.")

    encoded = pd.DataFrame({
        "encoded_feature": encoded_names,
        "importance": values,
    })
    encoded["source_feature"] = encoded["encoded_feature"].map(
        lambda name: _source_feature(name, features)
    )
    grouped = (
        encoded.groupby("source_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False, kind="stable")
    )
    if not np.isclose(grouped["importance"].sum(), values.sum()):
        raise ValueError(f"Aggregated importance did not conserve values for {artifact_path}.")

    grouped.insert(0, "run_id", run_id)
    grouped.insert(1, "horizon_minutes", horizon)
    grouped.insert(2, "task", task)
    grouped.insert(3, "model", selected["selected_model"])
    grouped["importance_method"] = importance_method
    grouped["artifact_path"] = artifact_path.relative_to(ROOT).as_posix()
    return grouped


def generate_feature_importance(
    run_id: str | None = None,
    *,
    allow_in_progress: bool = False,
) -> tuple[Path, Path]:
    """Write CSV and Markdown summaries for selected M5 models in one run."""
    run_id, manifest, report_dir = _load_run(run_id, allow_in_progress)
    model_dir = ROOT / manifest["paths"]["m5_model_dir"]
    split_dir = ROOT / manifest["paths"]["m5_split_dir"]
    results = [
        _feature_importance_for_model(
            run_id=run_id,
            task=task,
            horizon=horizon,
            report_dir=report_dir,
            model_dir=model_dir,
            split_dir=split_dir,
        )
        for task in TASKS
        for horizon in HORIZONS
    ]
    result = pd.concat(results, ignore_index=True)
    csv_path = report_dir / "m5_feature_importance.csv"
    markdown_path = report_dir / "m5_feature_importance.md"
    result.to_csv(csv_path, index=False)

    lines = [
        "# Corrected M5 Feature Importance",
        "",
        f"- Corrected run: `{run_id}`",
        f"- CSV: `{csv_path.relative_to(ROOT).as_posix()}`",
        "- Importance values are model-specific associations, not causal effects.",
        "- Tree models use native importance, linear models use absolute coefficients, "
        "and other estimators use absolute permutation importance on validation-selection rows.",
        "- Values are not comparable across model families; absolute permutation "
        "importance omits direction.",
        "",
    ]
    for (horizon, task, model), group in result.groupby(
        ["horizon_minutes", "task", "model"], sort=True
    ):
        lines.extend([
            f"## {task.replace('_', ' ').title()} — {horizon} minutes ({model})",
            "",
            f"Importance method: {group['importance_method'].iloc[0]}.",
            "",
            "| Rank | Feature | Importance |",
            "|---:|---|---:|",
        ])
        for rank, row in enumerate(group.head(15).itertuples(index=False), 1):
            lines.append(f"| {rank} | {row.source_feature} | {row.importance:.6f} |")
        lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export feature importance from selected corrected M5 models."
    )
    parser.add_argument(
        "--run-id",
        help="Completed corrected run ID (defaults to the current latest run).",
    )
    args = parser.parse_args()
    csv_path, markdown_path = generate_feature_importance(args.run_id)
    print("Feature-importance analysis completed.")
    print(f"CSV: {csv_path.relative_to(ROOT)}")
    print(f"Markdown: {markdown_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
