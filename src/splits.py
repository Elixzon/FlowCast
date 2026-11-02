from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/flowcast_m5_targets.csv"
OUTPUT_DIR = ROOT / "data/processed/m5_splits"

HORIZONS = [30, 60, 90, 120]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10


def main():
    df = pd.read_csv(INPUT)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    timestamps = pd.DatetimeIndex(
        sorted(df["timestamp"].unique())
    )

    n = len(timestamps)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    boundaries = {
        "train": (timestamps[0], timestamps[train_end - 1]),
        "validation": (
            timestamps[train_end],
            timestamps[validation_end - 1],
        ),
        "test": (
            timestamps[validation_end],
            timestamps[-1],
        ),
    }

    print("Chronological boundaries:")
    for name, (start, end) in boundaries.items():
        print(f"{name}: {start} to {end}")

    for minutes in HORIZONS:
        target_col = f"traffic_volume_t_plus_{minutes}"

        # Future labels must remain inside their own partition.
        future_timestamp = (
            df["timestamp"] + pd.Timedelta(minutes=minutes)
        )

        parts = []

        for split_name, (start, end) in boundaries.items():
            mask = (
                df["timestamp"].between(start, end)
                & future_timestamp.le(end)
                & df[target_col].notna()
            )

            part = df.loc[mask].copy()
            part["split"] = split_name
            parts.append(part)

        result = pd.concat(parts, ignore_index=True)

        output = OUTPUT_DIR / f"horizon_{minutes}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)

        print(f"\n{minutes}-minute horizon:")
        print("Rows:", len(result))
        print(result["split"].value_counts().to_string())
        print("Saved:", output)

    print("\nM5 chronological splitting completed.")


if __name__ == "__main__":
    main()
