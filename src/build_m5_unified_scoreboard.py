import json
import csv
from pathlib import Path

REPORTS = Path("reports")
HORIZONS = [30, 60, 90, 120]
rows = []

def read_json(name):
    with (REPORTS / name).open() as f:
        return json.load(f)

def metric(data, *keys):
    for key in keys:
        if key in data:
            return data[key]
    return None

def add_row(task, horizon, model, selection, values, status):
    rows.append({
        "task": task,
        "horizon_minutes": horizon,
        "selected_model": model,
        "selection_metric": selection,
        **values,
        "prd_status": status
    })

for h in HORIZONS:
    # Traffic volume
    if h == 30:
        data = read_json("m5_model_scoreboard_30m.json")
        winner = next(r for r in data if r.get("rank_by_RMSE") == 1)
        model = winner["model"]
        values = {
            "MAE": metric(winner, "MAE"),
            "RMSE": metric(winner, "RMSE"),
            "MAPE_percent": metric(winner, "MAPE_percent")
        }
    else:
        data = read_json(f"m5_traffic_volume_{h}m.json")
        winner = data["best_model"]
        model = winner["model"]
        values = {
            "MAE": metric(winner, "mae"),
            "RMSE": metric(winner, "rmse"),
            "MAPE_percent": metric(winner, "mape")
        }

    status = "PASS" if values["MAPE_percent"] <= 12 else "NOT MET"
    add_row("Traffic volume", h, model, "Lowest validation RMSE",
            values, status)

    # Travel time
    data = read_json(f"m5_travel_time_{h}m.json")
    winner = data["best_model"]
    add_row(
        "Travel time", h, winner["model"], "Lowest validation RMSE",
        {
            "MAE": metric(winner, "mae"),
            "RMSE": metric(winner, "rmse"),
            "MAPE_percent": metric(winner, "mape")
        },
        "NO PRD THRESHOLD"
    )

    # Congestion
    data = read_json(f"m5_congestion_classification_{h}m.json")
    winner = data["best_model"]
    score = metric(winner, "macro_f1")
    add_row(
        "Congestion", h, winner["model"], "Highest validation macro-F1",
        {
            "accuracy": metric(winner, "accuracy"),
            "macro_F1": score,
            "macro_precision": metric(winner, "macro_precision"),
            "macro_recall": metric(winner, "macro_recall")
        },
        "PASS" if score >= 0.80 else "NOT MET"
    )

    # Accident risk
    data = read_json(f"m5_accident_risk_{h}m.json")
    winner = data["best_model"]
    score = metric(winner, "roc_auc")
    add_row(
        "Accident risk", h, winner["model"], "Highest validation ROC-AUC",
        {
            "ROC_AUC": score,
            "PR_AUC": metric(winner, "pr_auc"),
            "precision": metric(winner, "precision"),
            "recall": metric(winner, "recall"),
            "F1": metric(winner, "f1")
        },
        "PASS" if score >= 0.75 else "NOT MET"
    )

# Write CSV
csv_path = REPORTS / "m5_unified_scoreboard.csv"
columns = list(dict.fromkeys(k for row in rows for k in row))

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

# Write Markdown
md_path = REPORTS / "m5_unified_scoreboard.md"
lines = [
    "# FlowCast M5 — Unified Model Scoreboard",
    "",
    "Validation results only. The untouched test partition was not used.",
    "",
    "| Task | Horizon | Selected model | Selection criterion | PRD status |",
    "|---|---:|---|---|---|"
]

for r in rows:
    lines.append(
        f"| {r['task']} | {r['horizon_minutes']} min "
        f"| {r['selected_model']} | {r['selection_metric']} "
        f"| {r['prd_status']} |"
    )

lines.extend([
    "",
    "## Detailed metrics",
    "",
    "| Task | Horizon | MAE | RMSE | MAPE % | Macro-F1 | ROC-AUC | PRD status |",
    "|---|---:|---:|---:|---:|---:|---:|---|"
])

for r in rows:
    def fmt(key):
        value = r.get(key)
        return f"{value:.4f}" if isinstance(value, (int, float)) else "—"

    lines.append(
        f"| {r['task']} | {r['horizon_minutes']} "
        f"| {fmt('MAE')} | {fmt('RMSE')} | {fmt('MAPE_percent')} "
        f"| {fmt('macro_F1')} | {fmt('ROC_AUC')} | {r['prd_status']} |"
    )

lines.extend([
    "",
    "## PRD acceptance summary",
    "",
    "- Traffic volume: MAPE <= 12%.",
    "- Congestion: macro-F1 >= 0.80.",
    "- Accident risk: ROC-AUC >= 0.75.",
    "- Travel time: no separate numeric acceptance threshold specified in the PRD.",
    "",
    "## Important limitations",
    "",
    "- These are validation metrics, not final test metrics.",
    "- Model winners are selected using their recorded validation criteria.",
    "- A selected model is not necessarily operationally acceptable.",
    "- PRD failures are recorded honestly; no thresholds have been relaxed.",
    "",
    "## Source",
    "",
    "Generated from the saved M5 experiment JSON reports."
])

md_path.write_text("\n".join(lines) + "\n")

print("Created:", csv_path)
print("Created:", md_path)
print("\nSelected models and PRD status:")

for r in rows:
    print(
        f"{r['task']:16} {r['horizon_minutes']:3}m "
        f"{r['selected_model']:18} {r['prd_status']}"
    )

print("\nTotal scoreboard entries:", len(rows))
