from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/flowcast_m3_features.csv"
OUTPUT = ROOT / "data/processed/flowcast_m5_targets.csv"

HORIZONS = [30, 60, 90, 120]

TARGET_COLUMNS = [
    "traffic_volume",
    "travel_time",
    "congestion_level",
    "accident_count",
]


def main():
    print("Loading M3 features...")
    df = pd.read_csv(INPUT)

    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        errors="raise"
    )

    df = df.sort_values(
        ["road_id", "timestamp"]
    ).reset_index(drop=True)

    if df.duplicated(["road_id", "timestamp"]).any():
        raise ValueError("Duplicate road-time keys detected.")

    original_rows = len(df)

    for minutes in HORIZONS:
        print(f"Creating {minutes}-minute targets...")

        future = df[
            ["road_id", "timestamp"] + TARGET_COLUMNS
        ].copy()

        # Align future observations to the current timestamp.
        future["timestamp"] -= pd.Timedelta(minutes=minutes)

        future = future.rename(
            columns={
                col: f"{col}_t_plus_{minutes}"
                for col in TARGET_COLUMNS
            }
        )

        df = df.merge(
            future,
            on=["road_id", "timestamp"],
            how="left",
            validate="one_to_one"
        )

        accident_col = f"accident_count_t_plus_{minutes}"

        risk_col = f"accident_risk_t_plus_{minutes}"

        df[risk_col] = (
            df[accident_col]
            .gt(0)
            .where(df[accident_col].notna(), pd.NA)
            .astype("Int8")
        )

        matched = df[f"traffic_volume_t_plus_{minutes}"].notna()
        positives = df[risk_col].eq(1).sum()

        print(
            f"  Matched: {matched.sum():,}/{original_rows:,} "
            f"({matched.mean() * 100:.2f}%)"
        )
        print(f"  Accident-positive labels: {positives:,}")

    if len(df) != original_rows:
        raise ValueError("Row count changed during target generation.")

    if df.duplicated(["road_id", "timestamp"]).any():
        raise ValueError("Duplicate keys appeared after target generation.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print("\nM5 target generation completed.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
