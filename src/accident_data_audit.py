from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILE = ROOT / "data/processed/m5_splits/horizon_30.csv"
TARGET = "accident_risk_t_plus_30"

df = pd.read_csv(FILE)

if "split" not in df.columns:
    raise ValueError("Expected split column was not found.")

if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

# Audit training data only
train = df[df["split"] == "train"].copy()
train = train.dropna(subset=[TARGET])
train[TARGET] = train[TARGET].astype(int)

print("=" * 65)
print("FLOWCAST — ACCIDENT DATA AUDIT")
print("=" * 65)

print("\nTRAINING DATA")
print("Rows:", len(train))
print("Accident cases:", int(train[TARGET].sum()))
print("Non-accident cases:", int((train[TARGET] == 0).sum()))
print("Accident rate:", round(train[TARGET].mean() * 100, 4), "%")

groups = [
    "road_id",
    "hour",
    "day_of_week",
    "weather_condition",
    "rain_flag",
    "low_visibility_flag",
    "is_peak_period",
    "is_weekend",
    "event_flag",
    "roadwork_flag",
]

for column in groups:
    if column not in train.columns:
        print(f"\nSkipping {column}: column unavailable")
        continue

    summary = (
        train.groupby(column, dropna=False)[TARGET]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    summary["accident_rate_pct"] = summary["mean"] * 100
    summary = summary.drop(columns="mean")

    print(f"\n{'=' * 65}")
    print(f"ACCIDENT RATE BY {column.upper()}")
    print(summary.to_string(index=False))

# Check future accident counts if available
future_count = "accident_count_t_plus_30"

if future_count in train.columns:
    print("\nFUTURE ACCIDENT COUNT DISTRIBUTION")
    print(train[future_count].value_counts(dropna=False).sort_index())

# Check whether current accident count is present
if "accident_count" in train.columns:
    print("\nCURRENT ACCIDENT COUNT DISTRIBUTION")
    print(train["accident_count"].value_counts(dropna=False).sort_index())

print("\nAUDIT COMPLETE")
