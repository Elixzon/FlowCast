from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/processed/m5_splits/horizon_30.csv"

df = pd.read_csv(FILE)
target = "accident_risk_t_plus_30"

for split in ["train", "validation"]:
    part = df[df["split"] == split].copy()

    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    print("Rows:", len(part))
    print("Future accident positives:", int(part[target].sum()))

    print("\nCurrent accident count distribution:")
    print(part["accident_count"].value_counts(dropna=False).sort_index())

    print("\nFuture accident rate by current accident count:")
    summary = (
        part.groupby("accident_count", dropna=False)[target]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    summary["future_accident_rate_pct"] = summary["mean"] * 100
    print(
        summary.drop(columns="mean").to_string(index=False)
    )

    print("\nCurrent accident count vs future target:")
    print(pd.crosstab(part["accident_count"], part[target]))

print("\nAudit complete. Test partition was not used.")
