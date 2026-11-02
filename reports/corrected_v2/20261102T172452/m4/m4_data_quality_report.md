# FlowCast M4 — EDA and Data Quality Report

## Dataset overview
- Rows: 176,701
- Columns: 60
- Road segments: 25
- Period: 2025-01-01 00:00:00 to 2025-05-31 23:30:00
- Duplicate road-time keys: 0
- Total null cells: 341,132

## Traffic volume
- Mean: 431.29
- Median: 362.00
- Standard deviation: 283.41
- Minimum: 41.00
- Maximum: 2090.00

## Congestion distribution
- Free-flow: 108,139
- Moderate: 41,905
- Heavy: 16,257
- Severe: 9,719

## Null values
- traffic_volume: 4,587
- avg_speed: 10
- occupancy: 11
- congestion_level: 681
- traffic_volume_lag_1: 8,889
- traffic_volume_lag_2: 8,893
- traffic_volume_lag_48: 9,956
- traffic_volume_rolling_mean_4: 32,958
- traffic_volume_rolling_std_4: 32,958
- traffic_volume_rolling_mean_8: 59,892
- traffic_volume_rolling_std_8: 59,892
- avg_speed_lag_1: 4,420
- avg_speed_lag_2: 4,448
- avg_speed_lag_48: 5,517
- avg_speed_rolling_mean_4: 16,996
- avg_speed_rolling_std_4: 16,996
- avg_speed_rolling_mean_8: 32,427
- avg_speed_rolling_std_8: 32,427
- volume_capacity_ratio: 4,587
- capacity_headroom: 4,587

The lag and rolling-window nulls represent historical warm-up periods. They should not be interpreted as missing sensor readings.

## Weather versus traffic
- Clear: mean volume 437.27, median 367.00, observations 129,292
- Cloudy: mean volume 428.89, median 360.00, observations 34,833
- Fog: mean volume 217.88, median 181.00, observations 1,215
- Rain: mean volume 367.83, median 331.00, observations 6,774

## Correlation with traffic volume
- vehicle_count: 1.0000
- volume_capacity_ratio: 0.9488
- occupancy: 0.9380
- traffic_volume_lag_1: 0.9338
- traffic_volume_lag_48: 0.8668
- traffic_volume_lag_2: 0.8415
- traffic_volume_rolling_mean_4: 0.8001
- signal_timing: 0.7609
- is_peak_period: 0.7165
- traffic_volume_rolling_std_4: 0.6428
- vehicle_share_car: 0.5943
- traffic_volume_rolling_mean_8: 0.5717
- travel_time: 0.5360
- traffic_volume_rolling_std_8: 0.5271
- avg_speed_rolling_std_4: 0.5181
- avg_speed_rolling_std_8: 0.4440
- temperature: 0.3850
- road_capacity: 0.2650
- hour: 0.2196
- day_of_week_sin: 0.1606
- event_flag: 0.1322
- event_proximity_flag: 0.0823
- visibility: 0.0727
- longitude: 0.0705
- accident_count: 0.0350
- latitude: 0.0253
- vehicle_share_lcv: -0.0006
- holiday_peak_interaction: -0.0229
- day_of_week_cos: -0.0352
- rainfall: -0.0381
- rain_flag: -0.0465
- low_visibility_flag: -0.0589
- roadwork_flag: -0.1107
- hour_sin: -0.1555
- public_holiday: -0.1562
- day_of_week: -0.1800
- is_weekend: -0.2290
- avg_speed_rolling_mean_8: -0.4530
- vehicle_share_hcv: -0.5170
- hour_cos: -0.5288
- vehicle_share_2w: -0.5405
- avg_speed_rolling_mean_4: -0.6917
- avg_speed_lag_2: -0.7211
- avg_speed_lag_48: -0.7591
- avg_speed_lag_1: -0.8211
- capacity_headroom: -0.8362
- avg_speed: -0.8961

## Modelling considerations
- Use chronological train/test splitting.
- Preserve road-segment boundaries when constructing lag features.
- Do not randomly split time-series observations.
- Handle lag and rolling warm-up nulls without introducing leakage.
- Investigate highly correlated predictors before model fitting.
- Treat correlation as association, not proof of causation.

## Generated figures
- `figures/m4/01_traffic_volume_distribution.png`
- `figures/m4/02_congestion_distribution.png`
- `figures/m4/03_speed_distribution.png`
- `figures/m4/04_hourly_traffic_pattern.png`
- `figures/m4/05_weekday_traffic_pattern.png`
- `figures/m4/06_road_segment_comparison.png`
- `figures/m4/07_weather_vs_traffic.png`
- `figures/m4/08_correlation_heatmap.png`
- `figures/m4/09_congestion_by_hour.png`
- `figures/m4/10_volume_vs_speed.png`