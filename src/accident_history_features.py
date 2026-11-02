from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/processed/flowcast_m5_targets.csv"
SPLITS = ROOT / "data/processed/m5_splits/horizon_30.csv"
OUTPUT = ROOT / "data/interim/flowcast_accident_history_experiment.csv"

full = pd.read_csv(SOURCE)
splits = pd.read_csv(SPLITS)

full["timestamp"] = pd.to_datetime(full["timestamp"])
splits["timestamp"] = pd.to_datetime(splits["timestamp"])

keys = ["road_id", "timestamp"]

if full.duplicated(keys).any() or splits.duplicated(keys).any():
    raise ValueError("Duplicate road-time records found.")

if "accident_count" not in full.columns:
    raise ValueError("accident_count missing from source.")

full = full.sort_values(keys).reset_index(drop=True)

lookup = full.set_index(keys)["accident_count"]

def exact_lag(minutes):
    prior = full["timestamp"] - pd.Timedelta(minutes=minutes)
    index = pd.MultiIndex.from_arrays(
        [full["road_id"], prior],
        names=keys
    )
    return lookup.reindex(index).to_numpy()

full["accident_lag_1"] = exact_lag(30)
full["accident_lag_2"] = exact_lag(60)
full["accident_lag_48"] = exact_lag(1440)

for window in (4, 8):
    values = [
        exact_lag(minutes)
        for minutes in range(30, window * 30 + 1, 30)
    ]
    matrix = np.column_stack(values)
    full[f"accident_rolling_sum_{window}"] = (
        pd.DataFrame(matrix).sum(axis=1, min_count=window).to_numpy()
    )

grouped = full.groupby("road_id", sort=False)["accident_count"]

past_sum = grouped.transform(
    lambda s: s.shift(1).fillna(0).cumsum()
)
past_count = full.groupby("road_id", sort=False).cumcount()

full["road_historical_accident_rate"] = (
    past_sum / past_count.replace(0, np.nan)
)

history_features = [
    "accident_lag_1",
    "accident_lag_2",
    "accident_lag_48",
    "accident_rolling_sum_4",
    "accident_rolling_sum_8",
    "road_historical_accident_rate",
]

# Retain every original feature from the M5 target dataset.
# Add the existing split assignment and target labels.
split_columns = keys + [
    "split",
    "accident_risk_t_plus_30"
]

result = splits[split_columns].merge(
    full,
    on=keys,
    how="left",
    validate="one_to_one",
    suffixes=("", "_source")
)

# Keep the six newly engineered features.
for feature in history_features:
    result[feature] = full.set_index(keys)[feature].reindex(
        pd.MultiIndex.from_frame(result[keys])
    ).to_numpy()

if len(result) != len(splits):
    raise ValueError("Unexpected row count after merge.")

if result["accident_risk_t_plus_30"].isna().any():
    raise ValueError("Missing accident target labels.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("CORRECTED EXPERIMENT DATASET")
print("Rows:", len(result))
print("Columns:", len(result.columns))
print("Original features retained:", all(
    c in result.columns for c in [
        "latitude", "longitude", "road_capacity",
        "weather_condition", "traffic_volume_lag_1",
        "avg_speed_lag_1"
    ]
))
print("\nHistorical feature missing values:")
print(result[history_features].isna().sum().to_string())

print("\nRows by split:")
print(result["split"].value_counts().to_string())

print("\nPositive labels by split:")
print(result.groupby("split")["accident_risk_t_plus_30"].sum().to_string())

print("\nSaved:", OUTPUT)
