"""Export tree-model feature importance for a completed corrected FlowCast run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "traffic_volume",
    "travel_time",
    "congestion_level",
    "accident_risk",
)
HORIZONS = (30, 60, 90, 120)


def _load_run(run_id: str | None) -> tuple[str, dict, Path]:
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
    if manifest.get("run_id") != run_id or manifest.get("status") != "completed":
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
    if importances is None:
        raise TypeError(
            f"Selected {selected['selected_model']} model for {task}/{horizon}m "
            "does not expose tree feature_importances_."
        )

    encoded_names = list(preprocessor.get_feature_names_out())
    values = np.asarray(importances, dtype=float).reshape(-1)
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
    grouped["artifact_path"] = artifact_path.relative_to(ROOT).as_posix()
    return grouped


def generate_feature_importance(run_id: str | None = None) -> tuple[Path, Path]:
    """Write CSV and Markdown summaries for selected tree models in one run."""
    run_id, manifest, report_dir = _load_run(run_id)
    model_dir = ROOT / manifest["paths"]["m5_model_dir"]
    results = [
        _feature_importance_for_model(
            run_id=run_id,
            task=task,
            horizon=horizon,
            report_dir=report_dir,
            model_dir=model_dir,
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
        "- Values are not directly comparable across different model families.",
        "",
    ]
    for (horizon, task, model), group in result.groupby(
        ["horizon_minutes", "task", "model"], sort=True
    ):
        lines.extend([
            f"## {task.replace('_', ' ').title()} — {horizon} minutes ({model})",
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
