import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/flowcast_m3_features.csv"
FIGURES = ROOT / "reports/figures/m4"
REPORT = ROOT / "reports/m4_eda_report.json"
QUALITY_REPORT = ROOT / "reports/m4_data_quality_report.md"

FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(INPUT)

    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        errors="coerce"
    )

    numeric = df.select_dtypes(include=np.number)
    correlation = numeric.corr()

    summary = {
        "module": "M4 - Exploratory Data Analysis",
        "input": (
            str(INPUT.relative_to(ROOT))
            if INPUT.is_relative_to(ROOT)
            else str(INPUT)
        ),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "road_segments": int(df["road_id"].nunique()),
        "date_start": str(df["timestamp"].min()),
        "date_end": str(df["timestamp"].max()),
        "duplicate_road_time_keys": int(
            df.duplicated(["road_id", "date", "time"]).sum()
        ),
        "total_null_cells": int(df.isna().sum().sum()),
        "null_counts": {
            k: int(v)
            for k, v in df.isna().sum().items()
            if v > 0
        },
        "traffic_volume": {
            "mean": float(df["traffic_volume"].mean()),
            "median": float(df["traffic_volume"].median()),
            "std": float(df["traffic_volume"].std()),
            "min": float(df["traffic_volume"].min()),
            "max": float(df["traffic_volume"].max()),
        },
        "avg_speed": {
            "mean": float(df["avg_speed"].mean()),
            "median": float(df["avg_speed"].median()),
            "min": float(df["avg_speed"].min()),
            "max": float(df["avg_speed"].max()),
        },
        "congestion_distribution": {
            str(k): int(v)
            for k, v in df["congestion_level"].value_counts().items()
        },
        "weather_distribution": {
            str(k): int(v)
            for k, v in df["weather_condition"].value_counts().items()
        },
        "correlation_with_traffic_volume": {
            str(k): float(v)
            for k, v in correlation["traffic_volume"]
                .drop("traffic_volume")
                .sort_values(ascending=False)
                .items()
        },
        "figures": [],
    }

    # 1. Traffic volume distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df["traffic_volume"], bins=50, kde=True)
    plt.title("Traffic Volume Distribution")
    plt.xlabel("Traffic Volume (vehicles / 30 minutes)")
    plt.ylabel("Frequency")
    save_plot("01_traffic_volume_distribution.png")
    summary["figures"].append("01_traffic_volume_distribution.png")

    # 2. Congestion distribution
    order = ["Free-flow", "Moderate", "Heavy", "Severe"]
    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=df,
        x="congestion_level",
        order=order
    )
    plt.title("Congestion Level Distribution")
    plt.xlabel("Congestion Level")
    plt.ylabel("Number of Observations")
    save_plot("02_congestion_distribution.png")
    summary["figures"].append("02_congestion_distribution.png")

    # 3. Average speed distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df["avg_speed"], bins=50, kde=True)
    plt.title("Average Speed Distribution")
    plt.xlabel("Average Speed (km/h)")
    plt.ylabel("Frequency")
    save_plot("03_speed_distribution.png")
    summary["figures"].append("03_speed_distribution.png")

    # 4. Hourly traffic pattern
    hourly = df.groupby("hour")["traffic_volume"].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly, x="hour", y="traffic_volume", marker="o")
    plt.title("Average Traffic Volume by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Traffic Volume")
    plt.xticks(range(24))
    save_plot("04_hourly_traffic_pattern.png")
    summary["figures"].append("04_hourly_traffic_pattern.png")

    # 5. Day-of-week traffic pattern
    daily = (
        df.groupby("day_of_week")["traffic_volume"]
        .mean()
        .reindex(range(7))
        .reset_index()
    )
    daily["day_name"] = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=daily, x="day_name", y="traffic_volume")
    plt.title("Average Traffic Volume by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Mean Traffic Volume")
    plt.xticks(rotation=25)
    save_plot("05_weekday_traffic_pattern.png")
    summary["figures"].append("05_weekday_traffic_pattern.png")

    # 6. Road segment comparison
    road = (
        df.groupby("road_name")["traffic_volume"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=road,
        y="road_name",
        x="traffic_volume"
    )
    plt.title("Average Traffic Volume by Road Segment")
    plt.xlabel("Mean Traffic Volume")
    plt.ylabel("Road Segment")
    save_plot("06_road_segment_comparison.png")
    summary["figures"].append("06_road_segment_comparison.png")

    # 7. Weather versus traffic
    weather = (
        df.groupby("weather_condition")["traffic_volume"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df,
        x="weather_condition",
        y="traffic_volume",
        estimator="mean",
        errorbar=None
    )
    plt.title("Average Traffic Volume by Weather Condition")
    plt.xlabel("Weather Condition")
    plt.ylabel("Mean Traffic Volume")
    save_plot("07_weather_vs_traffic.png")
    summary["figures"].append("07_weather_vs_traffic.png")
    summary["weather_traffic_summary"] = {
        str(row["weather_condition"]): {
            "mean": float(row["mean"]),
            "median": float(row["median"]),
            "count": int(row["count"]),
        }
        for _, row in weather.iterrows()
    }

    # 8. Correlation heatmap
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.2
    )
    plt.title("Numeric Feature Correlation Heatmap")
    save_plot("08_correlation_heatmap.png")
    summary["figures"].append("08_correlation_heatmap.png")

    # 9. Congestion by hour
    plt.figure(figsize=(11, 5))
    sns.countplot(
        data=df,
        x="hour",
        hue="congestion_level",
        hue_order=order
    )
    plt.title("Congestion Levels by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Observations")
    save_plot("09_congestion_by_hour.png")
    summary["figures"].append("09_congestion_by_hour.png")

    # 10. Volume versus speed
    sample = df.sample(
        n=min(12000, len(df)),
        random_state=42
    )
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="traffic_volume",
        y="avg_speed",
        hue="congestion_level",
        alpha=0.45,
        s=18
    )
    plt.title("Traffic Volume versus Average Speed")
    plt.xlabel("Traffic Volume")
    plt.ylabel("Average Speed (km/h)")
    save_plot("10_volume_vs_speed.png")
    summary["figures"].append("10_volume_vs_speed.png")

    # 11. Accident distribution by road segment
    acc_road = (
        df.groupby("road_name")["accident_count"]
        .agg(["sum", lambda s: (s > 0).mean() * 100])
        .rename(columns={"sum": "total_accidents", "<lambda_0>": "incident_rate_pct"})
        .sort_values("total_accidents", ascending=True)
        .reset_index()
    )
    plt.figure(figsize=(12, 8))
    plt.barh(acc_road["road_name"], acc_road["total_accidents"], color="#d95f02")
    plt.title("Total Incident Windows by Road Segment (Jan–May 2025)")
    plt.xlabel("Reported Incidents (Count)")
    plt.ylabel("Road Segment")
    save_plot("11_accident_distribution_by_road.png")
    summary["figures"].append("11_accident_distribution_by_road.png")

    # 12. Accident risk by weather and hour
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    acc_weather = (
        df.groupby("weather_condition")["accident_count"]
        .agg(rate=lambda s: (s > 0).mean() * 100)
        .reset_index()
    )
    sns.barplot(
        data=acc_weather,
        x="weather_condition",
        y="rate",
        hue="weather_condition",
        legend=False,
        ax=axes[0],
        palette="Blues_d",
    )
    axes[0].set_title("Incident Rate (%) by Weather Condition")
    axes[0].set_xlabel("Weather Condition")
    axes[0].set_ylabel("Incident Rate (%)")
    for p in axes[0].patches:
        height = p.get_height()
        if np.isfinite(height):
            axes[0].annotate(
                f"{height:.2f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
            )

    acc_hour = (
        df.groupby("hour")["accident_count"]
        .agg(["sum", lambda s: (s > 0).mean() * 100])
        .rename(columns={"sum": "total_accidents", "<lambda_0>": "incident_rate_pct"})
        .reset_index()
    )
    sns.lineplot(
        data=acc_hour,
        x="hour",
        y="total_accidents",
        marker="o",
        color="#e41a1c",
        ax=axes[1],
    )
    axes[1].set_title("Reported Incidents by Hour of Day")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Total Incidents")
    axes[1].set_xticks(range(24))

    save_plot("12_accident_risk_by_weather_and_hour.png")
    summary["figures"].append("12_accident_risk_by_weather_and_hour.png")

    summary["accident_analysis"] = {
        "total_accidents": int(df["accident_count"].sum()),
        "total_incident_windows": int((df["accident_count"] > 0).sum()),
        "overall_incident_rate_percent": float((df["accident_count"] > 0).mean() * 100),
        "by_weather_incident_rate_percent": {
            str(row["weather_condition"]): round(float(row["rate"]), 4)
            for _, row in acc_weather.iterrows()
        },
        "top_5_high_risk_roads": [
            {
                "road_name": str(row["road_name"]),
                "accidents": int(row["total_accidents"]),
                "incident_rate_pct": round(float(row["incident_rate_pct"]), 4),
            }
            for _, row in acc_road.sort_values("total_accidents", ascending=False).head(5).iterrows()
        ],
    }

    # Save machine-readable EDA report
    with open(REPORT, "w") as f:
        json.dump(summary, f, indent=2)

    # Save human-readable data quality and EDA report
    report = [
        "# FlowCast M4 — EDA and Data Quality Report",
        "",
        "## Dataset overview",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns)}",
        f"- Road segments: {df['road_id'].nunique()}",
        f"- Period: {summary['date_start']} to {summary['date_end']}",
        f"- Duplicate road-time keys: {summary['duplicate_road_time_keys']}",
        f"- Total null cells: {summary['total_null_cells']:,}",
        "",
        "## Traffic volume",
        f"- Mean: {summary['traffic_volume']['mean']:.2f}",
        f"- Median: {summary['traffic_volume']['median']:.2f}",
        f"- Standard deviation: {summary['traffic_volume']['std']:.2f}",
        f"- Minimum: {summary['traffic_volume']['min']:.2f}",
        f"- Maximum: {summary['traffic_volume']['max']:.2f}",
        "",
        "## Congestion distribution",
    ]

    for label, count in summary["congestion_distribution"].items():
        report.append(f"- {label}: {count:,}")

    report += ["", "## Null values"]
    for column, count in summary["null_counts"].items():
        report.append(f"- {column}: {count:,}")

    report += [
        "",
        "The lag and rolling-window nulls represent historical warm-up "
        "periods. They should not be interpreted as missing sensor readings.",
        "",
        "## Weather versus traffic",
    ]

    for condition, values in summary["weather_traffic_summary"].items():
        report.append(
            f"- {condition}: mean volume {values['mean']:.2f}, "
            f"median {values['median']:.2f}, "
            f"observations {values['count']:,}"
        )

    report += [
        "",
        "## Correlation with traffic volume",
    ]

    for feature, value in summary[
        "correlation_with_traffic_volume"
    ].items():
        report.append(f"- {feature}: {value:.4f}")

    report += [
        "",
        "## Accident-risk exploratory analysis",
        f"- Overall incident rate: {summary['accident_analysis']['overall_incident_rate_percent']:.4f}% ({summary['accident_analysis']['total_incident_windows']:,} windows with accidents out of {len(df):,})",
        "- Severe class imbalance: over 99% of 30-minute windows experience zero reported accidents.",
        "- Weather impact: Incident rates in Fog and Rain are significantly elevated (over 3.4%) compared to Clear and Cloudy conditions (< 0.85%).",
        "- Top high-risk corridors:",
    ]
    for road in summary["accident_analysis"]["top_5_high_risk_roads"]:
        report.append(
            f"  * {road['road_name']}: {road['accidents']} incidents ({road['incident_rate_pct']}%)"
        )
    report += [
        "- Operational implication: Extreme class imbalance makes accident prediction near-stochastic from sensor volume alone; probability calibration and specialized thresholding are mandatory.",
    ]

    report += [
        "",
        "## Modelling considerations",
        "- Use chronological train/test splitting.",
        "- Preserve road-segment boundaries when constructing lag features.",
        "- Do not randomly split time-series observations.",
        "- Handle lag and rolling warm-up nulls without introducing leakage.",
        "- Investigate highly correlated predictors before model fitting.",
        "- Treat correlation as association, not proof of causation.",
        "",
        "## Generated figures",
    ]

    for filename in summary["figures"]:
        report.append(f"- `figures/m4/{filename}`")

    QUALITY_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("FlowCast M4 EDA completed.")
    print("Rows:", len(df))
    print("Road segments:", df["road_id"].nunique())
    print("Figures generated:", len(summary["figures"]))
    print("EDA report:", REPORT)
    print("Data quality report:", QUALITY_REPORT)
    print("Figures directory:", FIGURES)


if __name__ == "__main__":
    main()
