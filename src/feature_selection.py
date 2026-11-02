from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/m5_feature_selection.json"

COMMON = [
    "road_id", "latitude", "longitude", "road_capacity",
    "weather_condition", "temperature_band", "temperature",
    "rainfall", "visibility", "public_holiday", "event_flag",
    "roadwork_flag", "hour", "day_of_week", "hour_sin",
    "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "is_weekend", "is_peak_period", "rain_flag",
    "low_visibility_flag", "holiday_peak_interaction",
    "traffic_volume", "vehicle_count", "avg_speed", "occupancy",
    "congestion_level", "travel_time", "accident_count", "signal_timing",
]

HISTORICAL = [
    "traffic_volume_lag_1", "traffic_volume_lag_2",
    "traffic_volume_lag_48", "traffic_volume_rolling_mean_4",
    "traffic_volume_rolling_std_4", "traffic_volume_rolling_mean_8",
    "traffic_volume_rolling_std_8", "avg_speed_lag_1",
    "avg_speed_lag_2", "avg_speed_lag_48",
    "avg_speed_rolling_mean_4", "avg_speed_rolling_std_4",
    "avg_speed_rolling_mean_8", "avg_speed_rolling_std_8",
]

FEATURE_SETS = {
    "traffic_volume": COMMON + HISTORICAL,
    "travel_time": COMMON + HISTORICAL,
    "congestion_level": COMMON + HISTORICAL,
    "accident_risk": COMMON + HISTORICAL,
}

EXCLUDED = {
    task: [
        "all future target columns",
        "event_proximity_flag (no announcement-time evidence)",
    ]
    for task in FEATURE_SETS
}

def main():
    result = {
        "feature_sets": FEATURE_SETS,
        "excluded_features": EXCLUDED,
        "notes": [
            "Features are measured at the forecast issue timestamp; future targets are excluded.",
            "Current traffic, speed, congestion, travel-time, and incident readings are valid inputs available at issue time.",
            "Historical lag and rolling traffic features use exact past timestamps.",
            "Categorical features require encoding.",
            "Scalers must be fitted on training data only.",
            "Lag warm-up nulls must not be filled using future data.",
            (
                "event_proximity_flag excluded from operational models because "
                "the source data do not provide event-announcement timestamps."
            ),
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=4))

    print("Feature selection specification created.")
    for task, features in FEATURE_SETS.items():
        print(f"{task}: {len(features)} candidate features")
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
