# Corrected M5 Feature Importance

- Corrected run: `20261102T172452`
- CSV: `reports/corrected_v2/20261102T172452/m5_feature_importance.csv`
- Importance values are model-specific associations, not causal effects.
- Values are not directly comparable across different model families.

## Accident Risk — 30 minutes (XGBoost (shallow))

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | road_id | 0.118944 |
| 2 | occupancy | 0.075744 |
| 3 | rain_flag | 0.068912 |
| 4 | visibility | 0.054979 |
| 5 | weather_condition | 0.039003 |
| 6 | avg_speed | 0.036183 |
| 7 | traffic_volume | 0.027267 |
| 8 | rainfall | 0.026452 |
| 9 | travel_time | 0.025276 |
| 10 | traffic_volume_lag_2 | 0.023078 |
| 11 | hour_cos | 0.023009 |
| 12 | traffic_volume_lag_1 | 0.022925 |
| 13 | day_of_week | 0.022686 |
| 14 | roadwork_flag | 0.022657 |
| 15 | traffic_volume_rolling_std_8 | 0.020222 |

## Congestion Level — 30 minutes (Random Forest)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | occupancy | 0.096483 |
| 2 | congestion_level | 0.094991 |
| 3 | avg_speed | 0.071372 |
| 4 | vehicle_count | 0.056815 |
| 5 | traffic_volume | 0.054983 |
| 6 | hour_cos | 0.054512 |
| 7 | is_peak_period | 0.041631 |
| 8 | traffic_volume_lag_1 | 0.038162 |
| 9 | hour | 0.036572 |
| 10 | hour_sin | 0.029217 |
| 11 | avg_speed_rolling_mean_8 | 0.027020 |
| 12 | avg_speed_rolling_mean_4 | 0.026215 |
| 13 | traffic_volume_lag_48 | 0.026114 |
| 14 | avg_speed_lag_1 | 0.023716 |
| 15 | signal_timing | 0.023332 |

## Traffic Volume — 30 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | congestion_level | 0.357464 |
| 2 | vehicle_count | 0.302461 |
| 3 | traffic_volume | 0.169977 |
| 4 | occupancy | 0.033896 |
| 5 | is_peak_period | 0.011904 |
| 6 | hour_cos | 0.011093 |
| 7 | is_weekend | 0.010839 |
| 8 | road_id | 0.010549 |
| 9 | avg_speed_rolling_mean_4 | 0.009983 |
| 10 | hour | 0.009159 |
| 11 | avg_speed_rolling_mean_8 | 0.008407 |
| 12 | road_capacity | 0.007888 |
| 13 | hour_sin | 0.007133 |
| 14 | traffic_volume_lag_48 | 0.005786 |
| 15 | event_flag | 0.005616 |

## Travel Time — 30 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | travel_time | 0.189540 |
| 2 | road_id | 0.128951 |
| 3 | congestion_level | 0.099113 |
| 4 | event_flag | 0.061373 |
| 5 | is_peak_period | 0.047935 |
| 6 | hour | 0.046643 |
| 7 | hour_sin | 0.039694 |
| 8 | occupancy | 0.037400 |
| 9 | road_capacity | 0.032688 |
| 10 | latitude | 0.026985 |
| 11 | longitude | 0.025240 |
| 12 | is_weekend | 0.024856 |
| 13 | avg_speed_rolling_mean_8 | 0.019006 |
| 14 | avg_speed_rolling_mean_4 | 0.018316 |
| 15 | day_of_week | 0.017831 |

## Accident Risk — 60 minutes (XGBoost (shallow))

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | road_id | 0.189073 |
| 2 | rain_flag | 0.053990 |
| 3 | visibility | 0.038524 |
| 4 | rainfall | 0.032825 |
| 5 | occupancy | 0.028275 |
| 6 | day_of_week_cos | 0.028052 |
| 7 | avg_speed | 0.026664 |
| 8 | hour_cos | 0.026377 |
| 9 | hour | 0.023847 |
| 10 | vehicle_count | 0.023592 |
| 11 | roadwork_flag | 0.022732 |
| 12 | traffic_volume_lag_1 | 0.021884 |
| 13 | day_of_week | 0.020984 |
| 14 | public_holiday | 0.020261 |
| 15 | avg_speed_lag_1 | 0.019654 |

## Congestion Level — 60 minutes (Random Forest)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_cos | 0.098112 |
| 2 | occupancy | 0.082061 |
| 3 | hour | 0.057286 |
| 4 | traffic_volume | 0.050498 |
| 5 | congestion_level | 0.047514 |
| 6 | hour_sin | 0.047385 |
| 7 | vehicle_count | 0.041691 |
| 8 | avg_speed | 0.040605 |
| 9 | avg_speed_rolling_mean_8 | 0.039540 |
| 10 | avg_speed_rolling_mean_4 | 0.031972 |
| 11 | traffic_volume_lag_1 | 0.029520 |
| 12 | is_peak_period | 0.028154 |
| 13 | traffic_volume_rolling_std_8 | 0.024055 |
| 14 | traffic_volume_lag_48 | 0.023474 |
| 15 | day_of_week | 0.023449 |

## Traffic Volume — 60 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | vehicle_count | 0.304280 |
| 2 | hour_cos | 0.197461 |
| 3 | traffic_volume | 0.174557 |
| 4 | avg_speed_rolling_mean_4 | 0.039368 |
| 5 | hour_sin | 0.034202 |
| 6 | hour | 0.030940 |
| 7 | road_id | 0.020266 |
| 8 | day_of_week | 0.019984 |
| 9 | road_capacity | 0.017894 |
| 10 | is_weekend | 0.016121 |
| 11 | roadwork_flag | 0.015655 |
| 12 | avg_speed_rolling_mean_8 | 0.014602 |
| 13 | event_flag | 0.014586 |
| 14 | occupancy | 0.013849 |
| 15 | traffic_volume_rolling_std_8 | 0.011296 |

## Travel Time — 60 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | is_peak_period | 0.162925 |
| 2 | congestion_level | 0.123235 |
| 3 | travel_time | 0.112963 |
| 4 | road_id | 0.088311 |
| 5 | hour_sin | 0.075422 |
| 6 | event_flag | 0.058788 |
| 7 | is_weekend | 0.038066 |
| 8 | hour | 0.035538 |
| 9 | day_of_week | 0.030810 |
| 10 | avg_speed_rolling_mean_8 | 0.020515 |
| 11 | road_capacity | 0.020264 |
| 12 | weather_condition | 0.019914 |
| 13 | latitude | 0.019663 |
| 14 | avg_speed_rolling_mean_4 | 0.018313 |
| 15 | hour_cos | 0.018250 |

## Accident Risk — 90 minutes (Random Forest)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | traffic_volume_rolling_std_8 | 0.041752 |
| 2 | avg_speed_rolling_std_4 | 0.039432 |
| 3 | avg_speed | 0.038916 |
| 4 | travel_time | 0.038805 |
| 5 | occupancy | 0.038360 |
| 6 | avg_speed_rolling_std_8 | 0.038310 |
| 7 | avg_speed_rolling_mean_8 | 0.037111 |
| 8 | traffic_volume_rolling_mean_8 | 0.036931 |
| 9 | traffic_volume_rolling_std_4 | 0.036695 |
| 10 | avg_speed_rolling_mean_4 | 0.036363 |
| 11 | traffic_volume_rolling_mean_4 | 0.036331 |
| 12 | avg_speed_lag_1 | 0.036271 |
| 13 | avg_speed_lag_48 | 0.035987 |
| 14 | avg_speed_lag_2 | 0.035856 |
| 15 | temperature | 0.035456 |

## Congestion Level — 90 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_cos | 0.171126 |
| 2 | congestion_level | 0.142276 |
| 3 | road_id | 0.079275 |
| 4 | hour_sin | 0.060161 |
| 5 | hour | 0.051594 |
| 6 | is_peak_period | 0.045273 |
| 7 | is_weekend | 0.040442 |
| 8 | occupancy | 0.037749 |
| 9 | roadwork_flag | 0.036902 |
| 10 | day_of_week | 0.036740 |
| 11 | public_holiday | 0.032361 |
| 12 | event_flag | 0.027042 |
| 13 | holiday_peak_interaction | 0.022261 |
| 14 | day_of_week_sin | 0.020385 |
| 15 | vehicle_count | 0.018049 |

## Traffic Volume — 90 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_cos | 0.601522 |
| 2 | vehicle_count | 0.043798 |
| 3 | hour | 0.038974 |
| 4 | hour_sin | 0.037866 |
| 5 | traffic_volume | 0.036781 |
| 6 | road_capacity | 0.029866 |
| 7 | day_of_week | 0.027088 |
| 8 | holiday_peak_interaction | 0.022588 |
| 9 | roadwork_flag | 0.022117 |
| 10 | road_id | 0.019308 |
| 11 | event_flag | 0.019006 |
| 12 | avg_speed_rolling_mean_4 | 0.015946 |
| 13 | public_holiday | 0.012138 |
| 14 | day_of_week_sin | 0.011835 |
| 15 | occupancy | 0.008271 |

## Travel Time — 90 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | congestion_level | 0.102927 |
| 2 | travel_time | 0.096710 |
| 3 | hour_cos | 0.089820 |
| 4 | hour_sin | 0.087745 |
| 5 | road_id | 0.079272 |
| 6 | event_flag | 0.067471 |
| 7 | vehicle_count | 0.046191 |
| 8 | hour | 0.038726 |
| 9 | weather_condition | 0.037174 |
| 10 | day_of_week | 0.035498 |
| 11 | is_weekend | 0.032434 |
| 12 | latitude | 0.026780 |
| 13 | avg_speed_rolling_mean_4 | 0.020240 |
| 14 | day_of_week_sin | 0.019707 |
| 15 | roadwork_flag | 0.019275 |

## Accident Risk — 120 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | road_id | 0.316783 |
| 2 | congestion_level | 0.043683 |
| 3 | weather_condition | 0.027681 |
| 4 | hour_sin | 0.019601 |
| 5 | visibility | 0.019488 |
| 6 | hour | 0.018333 |
| 7 | is_peak_period | 0.018136 |
| 8 | traffic_volume | 0.016679 |
| 9 | rain_flag | 0.016446 |
| 10 | roadwork_flag | 0.016324 |
| 11 | vehicle_count | 0.016315 |
| 12 | traffic_volume_rolling_mean_8 | 0.016237 |
| 13 | occupancy | 0.016231 |
| 14 | avg_speed_rolling_std_8 | 0.016215 |
| 15 | rainfall | 0.016209 |

## Congestion Level — 120 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_cos | 0.150167 |
| 2 | congestion_level | 0.122081 |
| 3 | road_id | 0.090507 |
| 4 | hour | 0.065663 |
| 5 | hour_sin | 0.065152 |
| 6 | day_of_week | 0.047110 |
| 7 | is_weekend | 0.042683 |
| 8 | roadwork_flag | 0.041971 |
| 9 | public_holiday | 0.040191 |
| 10 | event_flag | 0.031743 |
| 11 | occupancy | 0.026581 |
| 12 | is_peak_period | 0.025196 |
| 13 | day_of_week_sin | 0.025185 |
| 14 | avg_speed_rolling_mean_8 | 0.021477 |
| 15 | avg_speed_rolling_std_8 | 0.015787 |

## Traffic Volume — 120 minutes (Random Forest)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_cos | 0.450851 |
| 2 | hour_sin | 0.085693 |
| 3 | traffic_volume_rolling_mean_8 | 0.075033 |
| 4 | hour | 0.071297 |
| 5 | road_capacity | 0.064923 |
| 6 | traffic_volume_rolling_std_8 | 0.047409 |
| 7 | avg_speed_rolling_mean_8 | 0.028962 |
| 8 | is_weekend | 0.019747 |
| 9 | event_flag | 0.019523 |
| 10 | roadwork_flag | 0.018061 |
| 11 | day_of_week | 0.017608 |
| 12 | vehicle_count | 0.014578 |
| 13 | day_of_week_sin | 0.014250 |
| 14 | traffic_volume | 0.013738 |
| 15 | public_holiday | 0.010289 |

## Travel Time — 120 minutes (XGBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_sin | 0.120283 |
| 2 | road_id | 0.114348 |
| 3 | travel_time | 0.094504 |
| 4 | congestion_level | 0.075303 |
| 5 | event_flag | 0.070771 |
| 6 | hour_cos | 0.061629 |
| 7 | hour | 0.050968 |
| 8 | day_of_week | 0.044545 |
| 9 | is_weekend | 0.035460 |
| 10 | latitude | 0.033207 |
| 11 | roadwork_flag | 0.025330 |
| 12 | day_of_week_sin | 0.024377 |
| 13 | road_capacity | 0.022866 |
| 14 | avg_speed_rolling_mean_4 | 0.018447 |
| 15 | traffic_volume_rolling_std_8 | 0.017223 |
