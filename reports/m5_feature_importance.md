# M5 Feature Importance

Tree-based model feature importance, aggregated to original source features.
Importance values are model-specific and should not be interpreted as causal effects.

## Congestion Level — 30 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | is_peak_period | 0.431880 |
| 2 | hour_cos | 0.077640 |
| 3 | holiday_peak_interaction | 0.066862 |
| 4 | public_holiday | 0.060086 |
| 5 | road_id | 0.054970 |
| 6 | is_weekend | 0.040142 |
| 7 | roadwork_flag | 0.037775 |
| 8 | hour | 0.030533 |
| 9 | day_of_week | 0.028425 |
| 10 | traffic_volume_lag_1 | 0.027810 |
| 11 | day_of_week_sin | 0.017885 |
| 12 | event_flag | 0.015066 |
| 13 | hour_sin | 0.014141 |
| 14 | weather_condition | 0.009812 |
| 15 | avg_speed_lag_1 | 0.009706 |

## Traffic Volume — 30 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | is_peak_period | 0.423524 |
| 2 | traffic_volume_lag_1 | 0.187461 |
| 3 | hour_cos | 0.129903 |
| 4 | hour | 0.037078 |
| 5 | road_capacity | 0.027289 |
| 6 | hour_sin | 0.020696 |
| 7 | roadwork_flag | 0.019634 |
| 8 | event_flag | 0.019493 |
| 9 | day_of_week | 0.018004 |
| 10 | road_id | 0.017854 |
| 11 | day_of_week_sin | 0.017196 |
| 12 | traffic_volume_rolling_std_4 | 0.014781 |
| 13 | avg_speed_rolling_mean_8 | 0.009507 |
| 14 | is_weekend | 0.007534 |
| 15 | traffic_volume_rolling_mean_4 | 0.007101 |

## Travel Time — 30 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | road_id | 0.267131 |
| 2 | is_peak_period | 0.187975 |
| 3 | event_flag | 0.076558 |
| 4 | is_weekend | 0.049802 |
| 5 | road_capacity | 0.048434 |
| 6 | longitude | 0.043527 |
| 7 | latitude | 0.042708 |
| 8 | hour_cos | 0.035486 |
| 9 | day_of_week_sin | 0.033355 |
| 10 | day_of_week | 0.027791 |
| 11 | hour | 0.026510 |
| 12 | hour_sin | 0.017304 |
| 13 | event_proximity_flag | 0.016318 |
| 14 | avg_speed_rolling_std_8 | 0.015241 |
| 15 | roadwork_flag | 0.012204 |

## Congestion Level — 60 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | is_peak_period | 0.307308 |
| 2 | hour_cos | 0.153628 |
| 3 | public_holiday | 0.066600 |
| 4 | road_id | 0.060117 |
| 5 | is_weekend | 0.051380 |
| 6 | holiday_peak_interaction | 0.049624 |
| 7 | roadwork_flag | 0.047391 |
| 8 | hour | 0.040201 |
| 9 | day_of_week | 0.036341 |
| 10 | hour_sin | 0.028553 |
| 11 | day_of_week_sin | 0.024581 |
| 12 | event_flag | 0.015924 |
| 13 | traffic_volume_lag_1 | 0.013130 |
| 14 | avg_speed_rolling_mean_4 | 0.009569 |
| 15 | weather_condition | 0.009377 |

## Traffic Volume — 60 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | is_peak_period | 0.350816 |
| 2 | hour_cos | 0.349738 |
| 3 | road_capacity | 0.035888 |
| 4 | roadwork_flag | 0.031076 |
| 5 | traffic_volume_lag_1 | 0.028916 |
| 6 | day_of_week_sin | 0.026411 |
| 7 | day_of_week | 0.023188 |
| 8 | event_flag | 0.022234 |
| 9 | hour_sin | 0.021483 |
| 10 | hour | 0.020532 |
| 11 | holiday_peak_interaction | 0.014890 |
| 12 | road_id | 0.014319 |
| 13 | public_holiday | 0.011610 |
| 14 | is_weekend | 0.008902 |
| 15 | traffic_volume_rolling_std_4 | 0.004947 |

## Travel Time — 60 minutes (random_forest)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | is_peak_period | 0.204035 |
| 2 | latitude | 0.130935 |
| 3 | longitude | 0.101480 |
| 4 | road_capacity | 0.078884 |
| 5 | hour | 0.078007 |
| 6 | event_flag | 0.076862 |
| 7 | hour_cos | 0.054721 |
| 8 | is_weekend | 0.039645 |
| 9 | hour_sin | 0.033503 |
| 10 | avg_speed_rolling_mean_8 | 0.033322 |
| 11 | day_of_week | 0.030738 |
| 12 | road_id | 0.027037 |
| 13 | day_of_week_sin | 0.025483 |
| 14 | avg_speed_rolling_mean_4 | 0.009056 |
| 15 | roadwork_flag | 0.008713 |

## Congestion Level — 90 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_cos | 0.215261 |
| 2 | is_peak_period | 0.121290 |
| 3 | holiday_peak_interaction | 0.078284 |
| 4 | public_holiday | 0.077972 |
| 5 | road_id | 0.073390 |
| 6 | hour | 0.055762 |
| 7 | is_weekend | 0.052632 |
| 8 | roadwork_flag | 0.052004 |
| 9 | hour_sin | 0.050845 |
| 10 | day_of_week | 0.039985 |
| 11 | day_of_week_sin | 0.027703 |
| 12 | event_flag | 0.018644 |
| 13 | avg_speed_rolling_mean_8 | 0.012162 |
| 14 | avg_speed_rolling_mean_4 | 0.011923 |
| 15 | weather_condition | 0.011339 |

## Traffic Volume — 90 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_cos | 0.457851 |
| 2 | hour_sin | 0.057691 |
| 3 | road_capacity | 0.052701 |
| 4 | traffic_volume_lag_1 | 0.047101 |
| 5 | roadwork_flag | 0.046565 |
| 6 | hour | 0.045353 |
| 7 | holiday_peak_interaction | 0.044525 |
| 8 | day_of_week_sin | 0.042753 |
| 9 | event_flag | 0.035646 |
| 10 | road_id | 0.029498 |
| 11 | day_of_week | 0.029157 |
| 12 | public_holiday | 0.015897 |
| 13 | avg_speed_rolling_mean_4 | 0.015805 |
| 14 | traffic_volume_rolling_mean_8 | 0.012030 |
| 15 | traffic_volume_rolling_mean_4 | 0.011376 |

## Travel Time — 90 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | road_id | 0.281828 |
| 2 | hour_cos | 0.093896 |
| 3 | road_capacity | 0.062481 |
| 4 | event_flag | 0.058595 |
| 5 | latitude | 0.055201 |
| 6 | longitude | 0.052147 |
| 7 | hour | 0.047776 |
| 8 | is_peak_period | 0.043212 |
| 9 | day_of_week_sin | 0.041003 |
| 10 | day_of_week | 0.038608 |
| 11 | event_proximity_flag | 0.029496 |
| 12 | hour_sin | 0.024874 |
| 13 | is_weekend | 0.023986 |
| 14 | avg_speed_rolling_mean_4 | 0.019435 |
| 15 | roadwork_flag | 0.016329 |

## Accident Risk — 120 minutes (random_forest)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | traffic_volume_rolling_std_8 | 0.050308 |
| 2 | avg_speed_rolling_mean_8 | 0.050110 |
| 3 | avg_speed_rolling_std_8 | 0.050047 |
| 4 | avg_speed_lag_1 | 0.048257 |
| 5 | avg_speed_rolling_std_4 | 0.048215 |
| 6 | traffic_volume_rolling_std_4 | 0.047613 |
| 7 | temperature | 0.047308 |
| 8 | traffic_volume_rolling_mean_8 | 0.047275 |
| 9 | traffic_volume_lag_48 | 0.046893 |
| 10 | avg_speed_rolling_mean_4 | 0.046251 |
| 11 | traffic_volume_rolling_mean_4 | 0.046127 |
| 12 | avg_speed_lag_48 | 0.046074 |
| 13 | traffic_volume_lag_1 | 0.045825 |
| 14 | avg_speed_lag_2 | 0.045412 |
| 15 | traffic_volume_lag_2 | 0.044433 |

## Congestion Level — 120 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_cos | 0.192843 |
| 2 | hour | 0.098844 |
| 3 | road_id | 0.089012 |
| 4 | public_holiday | 0.081775 |
| 5 | is_weekend | 0.069698 |
| 6 | hour_sin | 0.064387 |
| 7 | roadwork_flag | 0.057961 |
| 8 | day_of_week | 0.057607 |
| 9 | holiday_peak_interaction | 0.042208 |
| 10 | day_of_week_sin | 0.037117 |
| 11 | avg_speed_rolling_mean_8 | 0.024502 |
| 12 | event_flag | 0.020960 |
| 13 | traffic_volume_rolling_std_8 | 0.016245 |
| 14 | traffic_volume_lag_1 | 0.016160 |
| 15 | latitude | 0.015233 |

## Traffic Volume — 120 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | hour_cos | 0.373647 |
| 2 | hour | 0.111131 |
| 3 | hour_sin | 0.080802 |
| 4 | road_capacity | 0.048527 |
| 5 | event_flag | 0.040493 |
| 6 | traffic_volume_rolling_mean_8 | 0.040044 |
| 7 | day_of_week | 0.039485 |
| 8 | traffic_volume_lag_1 | 0.038767 |
| 9 | day_of_week_sin | 0.037222 |
| 10 | road_id | 0.032352 |
| 11 | roadwork_flag | 0.032100 |
| 12 | public_holiday | 0.023956 |
| 13 | avg_speed_rolling_mean_8 | 0.018348 |
| 14 | traffic_volume_rolling_mean_4 | 0.012824 |
| 15 | is_weekend | 0.011739 |

## Travel Time — 120 minutes (xgboost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | road_id | 0.283914 |
| 2 | event_flag | 0.074344 |
| 3 | hour_sin | 0.068916 |
| 4 | road_capacity | 0.057946 |
| 5 | hour | 0.055440 |
| 6 | latitude | 0.054822 |
| 7 | longitude | 0.050244 |
| 8 | day_of_week_sin | 0.049567 |
| 9 | hour_cos | 0.048776 |
| 10 | is_weekend | 0.040725 |
| 11 | day_of_week | 0.039670 |
| 12 | event_proximity_flag | 0.028952 |
| 13 | roadwork_flag | 0.017662 |
| 14 | is_peak_period | 0.017269 |
| 15 | avg_speed_rolling_mean_4 | 0.014248 |
