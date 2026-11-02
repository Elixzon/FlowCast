from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/interim/flowcast_accident_history_experiment.csv"

df = pd.read_csv(FILE)

base_features = [
    "road_id", "latitude", "longitude", "road_capacity",
    "weather_condition", "temperature_band", "temperature",
    "rainfall", "visibility", "public_holiday", "event_flag",
    "roadwork_flag", "hour", "day_of_week", "hour_sin",
    "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "is_weekend", "is_peak_period", "rain_flag",
    "low_visibility_flag", "holiday_peak_interaction",
    "event_proximity_flag", "traffic_volume_lag_1",
    "traffic_volume_lag_2", "traffic_volume_lag_48",
    "traffic_volume_rolling_mean_4", "traffic_volume_rolling_std_4",
    "traffic_volume_rolling_mean_8", "traffic_volume_rolling_std_8",
    "avg_speed_lag_1", "avg_speed_lag_2", "avg_speed_lag_48",
    "avg_speed_rolling_mean_4", "avg_speed_rolling_std_4",
    "avg_speed_rolling_mean_8", "avg_speed_rolling_std_8"
]

history_features = [
    "accident_lag_1",
    "accident_lag_2",
    "accident_lag_48",
    "accident_rolling_sum_4",
    "accident_rolling_sum_8",
    "road_historical_accident_rate"
]

missing_columns = [
    c for c in base_features + history_features
    if c not in df.columns
]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

base_valid = df[base_features].notna().all(axis=1)

print("FLOWCAST ACCIDENT EXPERIMENT READINESS")
print("=" * 60)

for split in ["train", "validation", "test"]:
    part = df["split"] == split
    base_part = part & base_valid

    print(f"\n{split.upper()}")
    print("Original rows:", int(part.sum()))
    print("Rows with complete original features:", int(base_part.sum()))
    print(
        "Positive labels retained:",
        int(df.loc[base_part, "accident_risk_t_plus_30"].sum())
    )

    print("Missing historical feature values:")
    print(
        df.loc[part, history_features]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "Rows with all six historical features:",
        int((part & base_valid & df[history_features].notna().all(axis=1)).sum())
    )

print("\nTEST PARTITION HAS ONLY BEEN AUDITED, NOT USED FOR TRAINING.")
