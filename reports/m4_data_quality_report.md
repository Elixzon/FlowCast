# FlowCast M4 — EDA and Data Quality Report

## Dataset overview
- Rows: 176,701
- Columns: 60
- Road segments: 25
- Period: 2025-01-01 00:00:00 to 2025-05-31 23:30:00
- Duplicate road-time keys: 0
- Total null cells: 3,750

## Traffic volume
- Mean: 431.44
- Median: 362.00
- Standard deviation: 283.21
- Minimum: 41.00
- Maximum: 2090.00

## Congestion distribution
- Free-flow: 108,544
- Moderate: 42,098
- Heavy: 16,311
- Severe: 9,748

## Null values
- traffic_volume_lag_1: 25
- traffic_volume_lag_2: 50
- traffic_volume_lag_48: 1,200
- traffic_volume_rolling_mean_4: 100
- traffic_volume_rolling_std_4: 100
- traffic_volume_rolling_mean_8: 200
- traffic_volume_rolling_std_8: 200
- avg_speed_lag_1: 25
- avg_speed_lag_2: 50
- avg_speed_lag_48: 1,200
- avg_speed_rolling_mean_4: 100
- avg_speed_rolling_std_4: 100
- avg_speed_rolling_mean_8: 200
- avg_speed_rolling_std_8: 200

The lag and rolling-window nulls represent historical warm-up periods. They should not be interpreted as missing sensor readings.

## Weather versus traffic
- Clear: mean volume 437.39, median 367.00, observations 132,715
- Cloudy: mean volume 429.01, median 361.00, observations 35,779
- Fog: mean volume 219.41, median 181.50, observations 1,248
- Rain: mean volume 368.54, median 332.00, observations 6,959

## Correlation with traffic volume
- vehicle_count: 0.9993
- volume_capacity_ratio: 0.9487
- occupancy: 0.9381
- traffic_volume_lag_1: 0.9328
- traffic_volume_lag_2: 0.8354
- traffic_volume_rolling_mean_4: 0.7930
- traffic_volume_lag_48: 0.7902
- signal_timing: 0.7607
- is_peak_period: 0.7162
- traffic_volume_rolling_std_4: 0.6351
- vehicle_share_car: 0.5944
- traffic_volume_rolling_mean_8: 0.5614
- travel_time: 0.5350
- avg_speed_rolling_std_4: 0.5199
- traffic_volume_rolling_std_8: 0.5171
- avg_speed_rolling_std_8: 0.4390
- temperature: 0.3866
- road_capacity: 0.2658
- hour: 0.2202
- day_of_week_sin: 0.1607
- event_flag: 0.1332
- event_proximity_flag: 0.0831
- visibility: 0.0722
- longitude: 0.0711
- accident_count: 0.0349
- latitude: 0.0250
- vehicle_share_lcv: -0.0014
- holiday_peak_interaction: -0.0228
- day_of_week_cos: -0.0356
- rainfall: -0.0378
- rain_flag: -0.0461
- low_visibility_flag: -0.0585
- roadwork_flag: -0.1107
- hour_sin: -0.1557
- public_holiday: -0.1562
- day_of_week: -0.1800
- is_weekend: -0.2291
- avg_speed_rolling_mean_8: -0.4471
- vehicle_share_hcv: -0.5168
- hour_cos: -0.5291
- vehicle_share_2w: -0.5407
- avg_speed_lag_48: -0.6794
- avg_speed_rolling_mean_4: -0.6888
- avg_speed_lag_2: -0.7189
- avg_speed_lag_1: -0.8230
- capacity_headroom: -0.8359
- avg_speed: -0.8963

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