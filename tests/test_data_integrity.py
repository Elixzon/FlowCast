import numpy as np
import pandas as pd
import pytest

from src import clean
from src.clean import fill_segment_gaps
from src.features import add_lag_and_rolling_features
from src.ingest import _validate_frame
from src.data_quality import _iqr_counts, write_quality_report


def test_grouped_outlier_statistics_ignore_nan_and_both_infinities():
    frame = pd.DataFrame({
        "road_id": ["R1"] * 5 + ["R1"] * 3 + ["R2"] * 3,
        "measurement": [
            1.0, 2.0, 3.0, 4.0, 5.0,
            np.nan, np.inf, -np.inf,
            np.nan, np.inf, -np.inf,
        ],
    })

    report = _iqr_counts(frame, "measurement", "road_id")

    assert report["valid_rows"] == 5
    assert report["non_finite_numeric_rows"] == 6
    assert report["iqr_outlier_rows"] == 0
    assert report["zscore_outlier_rows"] == 0
    assert len(report["group_fences"]) == 1
    assert report["group_fences"][0]["valid_rows"] == 5
    assert all(
        np.isfinite(value)
        for key, value in report["group_fences"][0].items()
        if key in {"q1", "q3", "lower_fence", "upper_fence"}
    )


def _traffic_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "road_id": ["R1", "R1"],
        "road_name": ["Road 1", "Road 1"],
        "latitude": [28.0, 28.0],
        "longitude": [77.0, 77.0],
        "weather_station_id": ["W1", "W1"],
        "date": ["2025-01-01", "2025-01-01"],
        "time": ["00:00", "00:00"],
        "traffic_volume": [100, -1],
        "vehicle_count": [100, 100],
        "vehicle_type_dist": ["{}", "{}"],
        "avg_speed": [40, "not-a-number"],
        "occupancy": [45, 45],
        "congestion_level": ["Free-flow", "Free-flow"],
        "travel_time": [2.0, 2.0],
        "accident_count": [0, 0],
        "signal_timing": [30, 30],
        "road_capacity": [1000, 1000],
    })


def test_ingestion_flags_bad_numbers_dates_and_duplicate_keys():
    result, issues, quarantine = _validate_frame(
        _traffic_frame(), "traffic_sensor_log.csv"
    )

    issue_names = {item["issue"] for item in issues}
    assert "impossible_traffic_volume" in issue_names
    assert "invalid_numeric_avg_speed" in issue_names
    assert "duplicate_key" in issue_names
    assert result["quarantined_rows"] == 2
    assert set(quarantine["source_row"]) == {2, 3}


def test_weather_validation_accepts_documented_date_format_and_normalizes_labels():
    frame = pd.DataFrame({
        "station_id": ["W1", "W1"],
        "date": ["01/01/2025", "01/01/2025"],
        "time": ["00:00", "01:00"],
        "weather_condition": [" rainy ", "FOG"],
        "temperature": [10.0, 11.0],
        "rainfall": [1.0, 0.0],
        "visibility": [1000.0, 800.0],
    })

    result, issues, quarantine = _validate_frame(
        frame, "weather_observations.csv"
    )

    issue_names = {item["issue"] for item in issues}
    assert "invalid_date" not in issue_names
    assert "unknown_weather_condition" not in issue_names
    assert result["weather_label_counts_normalized"] == {"Rain": 1, "Fog": 1}
    assert quarantine.empty


def test_calendar_blank_names_are_allowed_only_when_corresponding_flag_is_clear():
    frame = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "public_holiday": [0, 0],
        "holiday_name": [None, None],
        "event_flag": [0, 1],
        "event_name": [None, None],
        "roadwork_flag": [0, 0],
    })

    result, issues, quarantine = _validate_frame(
        frame, "calendar_events.csv"
    )

    assert result["quarantined_rows"] == 1
    assert quarantine["rejection_reasons"].iloc[0] == (
        "missing_event_name_for_flagged_date"
    )
    assert any(item["issue"] == "missing_event_name_for_flagged_date" for item in issues)


def test_segment_gap_fill_uses_past_values_only_and_is_limited():
    frame = pd.DataFrame({
        "road_id": ["R1"] * 4,
        "_timestamp": pd.date_range("2025-01-01", periods=4, freq="30min"),
        "time": ["00:00", "00:30", "01:00", "01:30"],
        "measurement": [None, 10.0, None, None],
    })

    result = fill_segment_gaps(frame, "measurement")

    assert pd.isna(result.loc[0, "measurement"])
    assert result.loc[2, "measurement"] == 10.0
    assert result.loc[3, "measurement"] == 10.0


def test_segment_gap_fill_does_not_cross_a_long_timestamp_gap():
    frame = pd.DataFrame({
        "road_id": ["R1", "R1"],
        "_timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 02:00"]),
        "time": ["00:00", "02:00"],
        "measurement": [10.0, None],
    })

    result = fill_segment_gaps(frame, "measurement")

    assert pd.isna(result.loc[1, "measurement"])


def test_traffic_cleaning_does_not_impute_missing_target_from_future(tmp_path, monkeypatch):
    source = pd.DataFrame({
        "road_id": ["R1", "R1", "R1"],
        "road_name": ["Road 1"] * 3,
        "latitude": [28.0] * 3,
        "longitude": [77.0] * 3,
        "weather_station_id": ["W1"] * 3,
        "date": ["2025-01-01"] * 3,
        "time": ["00:00", "00:30", "01:00"],
        "traffic_volume": [100.0, None, 300.0],
        "vehicle_count": [100, 0, 300],
        "vehicle_type_dist": ["{}"] * 3,
        "avg_speed": [40.0, None, 50.0],
        "occupancy": [30.0, None, 40.0],
        "congestion_level": ["Free-flow", None, "Moderate"],
        "travel_time": [2.0, None, 3.0],
        "accident_count": [0, None, 0],
        "signal_timing": [30, 30, 30],
        "road_capacity": [400, 400, 400],
    })
    source_path = tmp_path / "traffic.csv"
    source.to_csv(source_path, index=False)
    monkeypatch.setattr(clean, "TRAFFIC_FILE", source_path)
    monkeypatch.setattr(clean, "TRAFFIC_OUTPUT", tmp_path / "clean.csv")
    monkeypatch.setattr(clean, "QUARANTINE", tmp_path / "quarantine")

    output, report = clean.clean_traffic()

    assert len(output) == len(source)
    assert pd.isna(output.loc[1, "traffic_volume"])
    assert output.loc[1, "avg_speed"] == 40.0
    assert output.loc[1, "occupancy"] == 30.0
    assert pd.isna(output.loc[1, "travel_time"])
    assert pd.isna(output.loc[1, "accident_count"])
    assert report["rows_with_unresolved_model_fields"] == 1


def test_lags_and_rollups_use_exact_prior_timestamps_and_keep_segment_boundaries():
    timestamps = pd.to_datetime([
        "2025-01-01 00:00",
        "2025-01-01 00:30",
        "2025-01-01 01:30",
        "2025-01-01 02:00",
        "2025-01-01 00:00",
        "2025-01-01 00:30",
    ])
    frame = pd.DataFrame({
        "road_id": ["R1", "R1", "R1", "R1", "R2", "R2"],
        "_timestamp": timestamps,
        "traffic_volume": [10.0, 20.0, 40.0, 50.0, 100.0, 200.0],
        "avg_speed": [30.0, 31.0, 33.0, 34.0, 50.0, 51.0],
    })

    result = add_lag_and_rolling_features(frame)

    assert pd.isna(result.loc[2, "traffic_volume_lag_1"])
    assert result.loc[3, "traffic_volume_lag_1"] == 40.0
    assert result.loc[1, "traffic_volume_lag_1"] == 10.0
    assert result.loc[5, "traffic_volume_lag_1"] == 100.0


def test_lag_features_reject_unsorted_rows():
    frame = pd.DataFrame({
        "road_id": ["R1", "R1"],
        "_timestamp": pd.to_datetime(["2025-01-01 00:30", "2025-01-01 00:00"]),
        "traffic_volume": [20.0, 10.0],
        "avg_speed": [31.0, 30.0],
    })

    with pytest.raises(ValueError, match="sorted by road segment and timestamp"):
        add_lag_and_rolling_features(frame)


def test_source_quality_report_flags_iqr_candidates_without_modifying_raw_inputs(
    tmp_path,
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    traffic = pd.DataFrame({
        "road_id": ["R1"] * 8,
        "date": ["2025-01-01"] * 8,
        "time": [f"{hour:02d}:00" for hour in range(8)],
        "traffic_volume": [100, 100, 100, 100, 100, 100, 100, 5000],
        "vehicle_count": [100] * 8,
        "avg_speed": [40] * 8,
        "occupancy": [50] * 8,
        "travel_time": [2] * 8,
        "accident_count": [0] * 8,
        "signal_timing": [30] * 8,
        "road_capacity": [400] * 8,
    })
    traffic.loc[0, "traffic_volume"] = -1
    traffic.to_csv(raw_dir / "traffic_sensor_log.csv", index=False)
    pd.DataFrame({
        "station_id": ["W1"] * 4,
        "date": ["01/01/2025"] * 4,
        "time": ["00:00", "01:00", "02:00", "03:00"],
        "temperature": [10, 11, 12, 13],
        "rainfall": [0, 0, 0, 0],
        "visibility": [1000, 1000, 1000, 1000],
    }).to_csv(raw_dir / "weather_observations.csv", index=False)
    pd.DataFrame({
        "date": ["2025-01-01"],
        "public_holiday": [0],
        "event_flag": [0],
        "roadwork_flag": [0],
    }).to_csv(raw_dir / "calendar_events.csv", index=False)
    original_bytes = {
        path.name: path.read_bytes() for path in raw_dir.iterdir()
    }

    report = write_quality_report(
        raw_dir,
        tmp_path / "report" / "quality.json",
        tmp_path / "report" / "quality.md",
    )

    traffic_report = report["tables"]["traffic_sensor_log.csv"]
    assert traffic_report["domain_invalid_counts"]["traffic_volume"] == 1
    assert traffic_report["statistical_iqr_flags"]["traffic_volume"][
        "iqr_outlier_rows"
    ] >= 1
    assert traffic_report["duplicate_key_rows_to_remove"] == 0
    assert {
        path.name: path.read_bytes() for path in raw_dir.iterdir()
    } == original_bytes
    assert "no statistical outlier is automatically" in traffic_report[
        "statistical_outlier_policy"
    ]


def test_eda_generates_accident_risk_analysis_and_figures(tmp_path):
    from src import eda

    report_dir = tmp_path / "reports"
    figures_dir = report_dir / "figures" / "m4"
    report_json = report_dir / "m4_eda_report.json"
    quality_md = report_dir / "m4_data_quality_report.md"

    # Create synthetic feature dataset with accidents
    features_csv = tmp_path / "features.csv"
    timestamps = pd.date_range("2025-01-01", periods=50, freq="30min")
    df = pd.DataFrame({
        "road_id": ["NL-001"] * 25 + ["NL-002"] * 25,
        "road_name": ["Northline Express"] * 25 + ["South Corridor"] * 25,
        "date": timestamps.strftime("%Y-%m-%d"),
        "time": timestamps.strftime("%H:%M"),
        "traffic_volume": np.random.randint(100, 500, size=50),
        "avg_speed": np.random.uniform(30, 80, size=50),
        "congestion_level": ["Free-flow"] * 40 + ["Moderate"] * 10,
        "weather_condition": ["Clear"] * 30 + ["Rain"] * 10 + ["Fog"] * 10,
        "accident_count": [0] * 45 + [1] * 5,
        "hour": timestamps.hour,
        "day_of_week": timestamps.dayofweek,
    })
    df.to_csv(features_csv, index=False)

    old_input = eda.INPUT
    old_figures = eda.FIGURES
    old_report = eda.REPORT
    old_quality = eda.QUALITY_REPORT

    try:
        eda.INPUT = features_csv
        eda.FIGURES = figures_dir
        eda.REPORT = report_json
        eda.QUALITY_REPORT = quality_md
        figures_dir.mkdir(parents=True, exist_ok=True)

        eda.main()

        assert report_json.is_file()
        assert quality_md.is_file()
        assert (figures_dir / "11_accident_distribution_by_road.png").is_file()
        assert (figures_dir / "12_accident_risk_by_weather_and_hour.png").is_file()

        import json
        with open(report_json) as f:
            summary = json.load(f)
        assert "accident_analysis" in summary
        assert summary["accident_analysis"]["total_incident_windows"] == 5
        assert summary["accident_analysis"]["overall_incident_rate_percent"] == 10.0
        assert "11_accident_distribution_by_road.png" in summary["figures"]
        assert "12_accident_risk_by_weather_and_hour.png" in summary["figures"]

        md_content = quality_md.read_text(encoding="utf-8")
        assert "Accident-risk exploratory analysis" in md_content
    finally:
        eda.INPUT = old_input
        eda.FIGURES = old_figures
        eda.REPORT = old_report
        eda.QUALITY_REPORT = old_quality
