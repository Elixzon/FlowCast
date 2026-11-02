import json
from pathlib import Path

horizons = [30, 60, 90, 120]
report_dir = Path("reports")
lines = [
    "# FlowCast M5 — Accident Risk Results Across All Horizons",
    "",
    "## Model comparison",
    "",
]

for horizon in horizons:
    path = report_dir / f"m5_accident_risk_{horizon}m.json"

    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text())
    lines.extend([
        f"### {horizon}-minute horizon",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for result in data["results"]:
        lines.append(
            f"| {result['model']} "
            f"| {result['roc_auc']:.4f} "
            f"| {result['pr_auc']:.4f} "
            f"| {result['precision']:.4f} "
            f"| {result['recall']:.4f} "
            f"| {result['f1']:.4f} |"
        )

    best = data["best_model"]
    lines.extend([
        "",
        f"ROC-AUC-selected model: **{best['model']}** "
        f"({best['roc_auc']:.4f})",
        "",
    ])

lines.extend([
    "## PRD evaluation",
    "",
    "Required accident-risk ROC-AUC: >= 0.75.",
    "",
    "| Horizon | Selected model | ROC-AUC | Status |",
    "|---|---|---:|---|",
])

for horizon in horizons:
    data = json.loads(
        (report_dir / f"m5_accident_risk_{horizon}m.json").read_text()
    )
    best = data["best_model"]
    status = "Achieved" if best["roc_auc"] >= 0.75 else "Not achieved"
    lines.append(
        f"| {horizon} minutes | {best['model']} "
        f"| {best['roc_auc']:.4f} | {status} |"
    )

lines.extend([
    "",
    "## Findings and limitations",
    "",
    "- Accident-risk prediction remains below the PRD ROC-AUC target.",
    "- Positive accident cases are rare, so accuracy alone is insufficient.",
    "- Precision, recall, F1, ROC-AUC and PR-AUC must be considered together.",
    "- A high ROC-AUC does not guarantee useful predictions at the default threshold.",
    "- The 120-minute Random Forest has zero recall at its default threshold.",
    "- Threshold tuning does not improve ROC-AUC.",
    "- These results are validation results; the test partition has not been used.",
    "",
    "## Source reports",
    "",
])

for horizon in horizons:
    lines.append(f"- reports/m5_accident_risk_{horizon}m.json")

output = report_dir / "m5_accident_risk_all_horizons.md"
output.write_text("\n".join(lines) + "\n")

print("Consolidated report saved:", output)
print("\nSelected models:")
for horizon in horizons:
    data = json.loads(
        (report_dir / f"m5_accident_risk_{horizon}m.json").read_text()
    )
    best = data["best_model"]
    print(
        f"{horizon}m: {best['model']} | "
        f"ROC-AUC={best['roc_auc']:.4f}"
    )
