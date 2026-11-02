# FlowCast — Initial Raw Data Audit

This is an initial inspection of the supplied files. It is not the final M2 cleaning report.

## Dataset inventory
| Source | Rows | Columns |
|---|---:|---:|
| traffic_sensor_log.csv | 178,468 | 17 |
| weather_observations.csv | 10,872 | 7 |
| calendar_events.csv | 151 | 6 |

Traffic spans 2025-01-01 through 2025-05-31 and contains 25 distinct road IDs.
Weather contains three stations. Calendar covers every date in the same 151-day period.

## Observed traffic defects
| Finding | Count |
|---|---:|
| Rows in duplicate road_id + date + time groups | 3,534 |
| Excess duplicate rows (keep-first comparison) | 1,767 |
| Missing traffic_volume | 4,387 |
| Missing avg_speed | 4,382 |
| Missing occupancy | 4,383 |
| Missing congestion_level | 26,883 |
| Negative traffic_volume | 241 |
| avg_speed > 200 km/h | 237 |
| occupancy > 100% | 234 |

## Observed weather defects
| Finding | Count |
|---|---:|
| Missing temperature | 167 |
| Missing visibility | 111 |
| Distinct raw labels (after trim/lowercase) | Clear, cloudy, overcast, rain, rainy, fog, foggy |

## Integrity observations
- Calendar dates have no duplicate keys and cover the full 2025-01-01 to 2025-05-31 period.
- Weather has no duplicate station + date + hour keys.
- Raw files are retained unchanged in `data/raw/`.

## Important interpretation
The M1 validator reports defects; it does not clean or discard records. M2 must apply
the PRD's ordered cleaning pipeline, retain an audit trail, and quarantine rows that
cannot safely be recovered. The 3,534 count is the number of rows belonging to
duplicate-key groups; 1,767 is the number of excess rows if one row per key is kept.
