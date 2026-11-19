# FlowCast Project Audit — Corrected Python 3.11 Run

## Audit scope and evidence

This audit covers the approved FlowCast PRD, source code, raw inputs, corrected

pipeline outputs, model evaluations/cards, dashboard behavior, and automated

tests. It supersedes earlier audit conclusions based on legacy metrics. The

November 4 M5 recalibration run has now regenerated all 12 task/horizon

models using the revised accident-risk validation protocol. Run B remains the

raw-data pipeline evaluation record; the recalibration is a separate M5-only

run and does not establish an uninterrupted end-to-end pipeline execution.

The latest evaluation run is `prd-compliance-20261103-b`. It started from the

supplied raw CSVs in Python 3.11.18 and completed all ten registered stages.

The initial invocation was externally interrupted after M6 comparison; the

same run was safely resumed at the sequence-sweep stage. Consequently, full

stage completion and raw-data reproduction are verified, but a single

uninterrupted invocation of the current ten-stage version remains unverified.

All three raw-input SHA-256 values match `data/raw/`; earlier experiment

outputs remain separately versioned.

Evidence: [run manifest](./corrected_v2/prd-compliance-20261103-b/run_manifest.json),

[M5 scoreboard](./corrected_v2/prd-compliance-20261103-b/m5_corrected_test_scoreboard.csv),

[M6 comparison](./corrected_v2/prd-compliance-20261103-b/m6_corrected_head_to_head.md),

[sequence sweep](./corrected_v2/prd-compliance-20261103-b/sequence_length_sweep/m6_sequence_length_sweep.md),

and [model cards](./corrected_v2/prd-compliance-20261103-b/model_cards/).

Status definitions: **PASS** means verified; **PARTIAL** means some required

behavior or evidence remains incomplete; **FAIL** means a stated acceptance

threshold was not met; **NOT VERIFIED** means the available environment or

evidence cannot establish the requirement.

## Verified results

- Corrected M1–M6 pipeline stages completed under Python 3.11.18 with raw-file

  hashes unchanged; the run processed 176,701 merged traffic records on a

  16-GiB workstation.

- Run B's M5 evaluated 16 selected task/horizon models on chronological,

  untouched test partitions. Its preprocessing/model selection used training

  and early validation data; classifier calibration used a later validation

  block. In that run, accident thresholds were selected from the earlier

  validation block after fitting calibration on the later block. This was not

  test leakage, but the temporal order was not strictly forward. Source now

  defines three ordered accident-risk validation blocks; Run B results do not

  reflect that revised protocol. A separate M5 recalibration was completed

  on November 4; its protocol, artifacts, and results are summarized below.

- M6 trained LSTM, GRU, and Bidirectional LSTM models for all four horizons.

  Classical/deep selection and comparisons use identical matched validation

  and test records per horizon: 2,673 (30m), 2,602 (60m), 2,522 (90m), and

  2,447 (120m) for the frozen test evaluation.

- Generated 32 model cards (16 M5, 12 primary M6, and four sequence-sweep

  winners), 40 training-history CSVs, and 40 loss-curve figures from actual

  training. The selected-model feature-importance report contains 720 finite,

  non-negative rows across all 16 M5 task/horizon groups.

- Automated tests: **45 passed in 56.29 seconds** on Python 3.11.18,

  including the AppTest exercising all 11 dashboard routes against the corrected

  evaluation data and dedicated M4 accident-risk EDA tests. This is the latest

  full-suite result recorded in the project.

- Streamlit AppTest exercised all 11 dashboard routes with no exceptions.

  Live Predictions completed in **23.67 seconds**, below the PRD's 30-second

  batch target.

- Historical traffic-only upload run

  `upload_20261103T175330107104Z_ccb360caec54` used the raw weather/calendar

  fallback, completed M1–M3, and generated all 16 task/horizon predictions

  for all 25 roads with no skipped roads. Those predictions were transient

  execution output; no prediction CSV or reproducibility manifest was retained

  for that run. Source now persists predictions and run evidence for future

  uploads, but a real upload has not yet been rerun to verify the new artifacts.

- Independent Python 3.11.18 runs `prd-compliance-20261103-a` and

  `prd-compliance-20261103-b` produced identical selected M5/M6 model choices

  and test metrics that agree to reported precision, including the 24/48/96

  sequence sweep; tiny floating-point differences remain in stored M6 values.

- Nominal 90% split-conformal intervals were calibrated on later validation

  residuals. Test coverage is reported below; time dependence means nominal

  coverage is not guaranteed.

- November 4 M5 recalibration: 12 task/horizon result JSON files and 12

  selected model artifacts were verified, plus four traffic-volume candidate

  artifacts. All four accident-risk artifacts contain a persisted

  `CalibratedClassifierCV` sigmoid calibrator, calibration period/row metadata,

  and a separately selected operating threshold. The accident protocol uses

  chronological model-selection, calibration, and threshold-selection blocks.

  Calibration rows were 1,842–1,844; threshold-selection rows were 2,022–2,040.

  All four confusion matrices, combined ROC–PR plots, and reliability diagrams

  are present. The run is stored separately under

  `models/corrected_v2/m5-recalibration-20261104/` and

  `reports/corrected_v2/m5-recalibration-20261104/`.

- The recalibrated M5 metrics remain below the congestion and accident-risk

  PRD thresholds. Recalibration protocol and persistence are verified; improved

  predictive performance is not claimed.

## PRD functional-requirement acceptance

| Requirement | Status | Evidence / remaining limitation |

|---|---|---|

| FR-01 — Ingest raw traffic, weather, and calendar data | PASS | Corrected M1–M6 run completed from the project raw inputs; raw hashes are preserved in the manifest. |

| FR-02 — Validate schema/types/ranges and record rejected rows | PASS | M1 report, issue register, and row-level quarantine outputs were generated. |

| FR-03 — Resolve missing readings, duplicates, and sensor faults | PARTIAL | Corrected M2 removed 1,767 duplicate rows and retained invalid measurements as missing. A read-only IQR/z-score review now flags outliers, but statistical flags are not adjudicated as sensor faults or automatically removed. |

| FR-04 — Merge sources by segment and time | PASS | M2 merged 176,701 traffic rows with weather/calendar data; no unmatched weather/calendar keys. |

| FR-05 — Generate lag, rolling, time, and weather features | PASS | Corrected M3 produced 59 features with exact-time, road-specific lags/rolling windows. `event_proximity_flag` is excluded from operational predictors because event-announcement times are unavailable. |

| FR-06 — Persist versioned data and quality reports | PASS | Corrected run artifacts and quality reports are isolated under the versioned run ID. |

| FR-07 — Train required regression families | PASS | Corrected M5 candidate results include Linear Regression, Decision Tree, Random Forest, XGBoost, and NumPy Gradient Descent across the requested regression targets/horizons. |

| FR-08 — Train congestion/accident classifiers including SVM | PASS | Corrected M5 candidate records include Linear SVM and tree/boosting classifiers for the classification targets. |

| FR-09 — Train recurrent models from scratch | PASS | LSTM/GRU and comparison-only Bidirectional LSTM were trained for all four horizons without pretrained weights. |

| FR-10 — Produce the four forecast targets by segment/window | PASS | Corrected M5 covers all 16 target/horizon combinations; dashboard inference emits segment-level outputs. |

| FR-11 — Attach uncertainty/confidence to each forecast | PARTIAL | M5/M6 regression intervals use later-validation split-conformal calibration. M5 test coverage was 0.917–0.987; some selected M6 models were below 0.90 (0.867–0.890). Classifier probabilities use held-out sigmoid calibration. For accident risk, the November 4 models use three chronological validation blocks (selection, calibration, then threshold selection); congestion and regression retain their earlier two-block protocol. Time dependence/shift means nominal interval coverage is not guaranteed. |

| FR-12 — Persist and reload trained models | PASS | M5 and M6 estimators are saved and loaded by the dashboard without retraining. |

| FR-13 — Show live/near-term segment predictions | PARTIAL | Near-term predictions render, but are based on the latest stored historical observations; no live sensor feed is configured. |

| FR-14 — Historical trends, congestion heatmaps, road comparisons | PASS | All three requested views rendered on corrected outputs in AppTest. |

| FR-15 — Performance, feature importance, forecast visualizations | PASS | All requested views rendered on corrected outputs in AppTest. |

| FR-16 — Weather-versus-traffic view | PASS | View rendered on corrected outputs in AppTest. |

| FR-17 — Upload a new file and trigger a prediction run | PASS | Upload `upload_20261103T175330107104Z_ccb360caec54` was validated, cleaned, and feature-engineered through M1–M3; current saved M5 models generated all 16 target/horizon outputs for all 25 roads. Fallbacks and `as_of` were recorded, regression bounds were finite and ordered, and raw files were unchanged. |

| FR-18 — Export selected-range insights | PASS | Selected-range report export is implemented and covered by automated tests. |

## PRD success criteria and non-functional checks

| Criterion | Status | Verified result |

|---|---|---|

| Volume forecast MAPE ≤ 12% on hold-out | PASS | 9.8569%, 10.1204%, 10.4251%, and 10.3468% at 30/60/90/120m. |

| Congestion macro-F1 ≥ 0.80 | FAIL | 0.7592, 0.7461, 0.7571, and 0.7332; none meets the threshold. |

| Accident-risk ROC-AUC ≥ 0.75 | FAIL | 0.3876, 0.4006, 0.5061, and 0.5578; all remain below target. |

| Sequence model beats best classical volume RMSE | FAIL | Validation-selected GRU/LSTM RMSE is worse than the matched classical model by 1.6032, 7.1501, 9.0527, and 15.2188 at 30/60/90/120m. |

| Full pipeline reproduction from raw data | PARTIAL | Run `prd-compliance-20261103-b` completed all ten current stages from unchanged raw inputs, but the process was resumed at `m6_sequence_sweep` after an external session interruption. A single uninterrupted invocation remains unverified. |

| All nine required dashboard views render | PASS | All nine analytics views plus Data Validation and Model Training (11 routes) passed AppTest. |

| Full-corridor inference ≤ 30 seconds | PASS | Live Predictions including model loading and batch inference completed in 23.67 seconds. |

| Python 3.11 compatibility | PASS | Python 3.11.18 and `pip check` pass; all 41 tests and the dashboard route test pass. |

| Event-feature as-of availability | PARTIAL | No event-announcement timestamps are supplied, so event proximity is deliberately excluded from operational predictors. |

## PRD non-functional requirements

| Requirement | Status | Evidence / limitation |

|---|---|---|

| Performance ≤ 30 seconds | PASS | Live Predictions completed in 23.67 seconds. |

| Scalability ≥ 150,000 rows on 16 GB | PASS | Full 176,701-row pipeline completed on a 16-GiB machine; peak RSS was not separately profiled. |

| Deterministic clean rerun | PASS | Two Python 3.11.18 from-raw runs produced identical selected model choices and recorded M5/M6 test metrics; wall-clock timings and artifact bytes were not compared. |

| Modular/documented, PEP 8 | PARTIAL | Pipeline modules and documentation exist; syntax/compile and tests pass, but no dedicated style linter was run. |

| ≤ 3 clicks to any dashboard view | PASS | All views are reachable through the flat navigation radio. |

| Quarantine bad rows; no silent drops | PASS | M1 quarantine/issue reports and M2 cleaning reports were generated; upload parsing and processing errors are surfaced. |

| Windows/macOS/Linux portability | PARTIAL | Pinned requirements and setup are documented; clean execution was verified on macOS only. |

| Full prediction lineage | PASS | Run/raw hashes, model cards/artifact paths, model-run IDs, and upload `as_of` timestamps are retained. |

## Model comparison outcome

M6 classical and operational deep candidates were selected using the same

matched early-validation records; the later validation block was reserved for

interval calibration. Each frozen model was evaluated on the same held-out

test rows at its horizon. The raw issue-time predictor columns match across

families; each family uses training-fitted, model-appropriate encoding and

scaling, while M6 additionally receives a 48-step historical sequence.

Validation selection did not use the test set. Recurrent models did not

outperform their matched classical comparators, including the Bidirectional

LSTM, so the sequence-learning success criterion remains unmet.

## Overall acceptance decision

**Overall: PARTIAL — not PRD acceptance-complete.** The Python 3.11

reproduction stages, upload-to-prediction path, dashboard, tests, and

artifacts are verified. Dedicated accident-risk EDA has now been completed

in M4 (`src/eda.py` and `notebooks/FlowCast_M4_EDA.ipynb`), generating figures 11

and 12 and quantifying incident rates across roads, weather, and hours.

Acceptance remains blocked by failed congestion, accident-risk, and

deep-versus-classical thresholds, with definitive sensor fault adjudication,

event timing, interval coverage guarantees, repeatability, style linting, and

cross-platform verification explicitly partial above. No historical metric is

used as a verified final result.

## Final release checklist — FlowCast v1.0-rc1

Review date: 2026-11-04. Technical completion is distinguished from PRD

performance acceptance. The November 4 M5-only recalibration is additional to

the November 3 raw-data pipeline run.

### Technical release checks

- [x] **Python 3.11 environment:** Python 3.11.18 was used; `pip check`

  reported no broken requirements.

- [x] **Automated tests and dashboard:** The full suite passed: **41 passed

  in 43.72 seconds**, including the dashboard AppTest exercising all 11 routes.

- [x] **Dashboard on persisted models and data:** All 11 routes rendered in

  AppTest without exceptions against the corrected run and its real processed

  data.

- [ ] **Uninterrupted one-command reproduction:** Run

  `prd-compliance-20261103-b` started from raw data and completed all ten

  stages, but it required resuming at `m6_sequence_sweep` after an external

  session interruption. Do not claim this strict criterion passed.

- [x] **Models, cards, curves, and reports:** The run contains 20 M5 joblib

  artifacts, 40 M6 Keras artifacts, 32 model cards (28 primary cards plus

  four sequence-sweep cards), 40 training-history CSVs, and 40 actual

  loss-curve figures. All 92 artifact paths referenced by the model cards

  resolve. Corrected M4, M5, M6, interval, quality, and sequence sweep

  reports are present.

- [x] **Feature importance:** The selected-model CSV/Markdown outputs contain

  720 finite, non-negative source-feature rows, 45 per each of the 16 M5

  task/horizon combinations. See the

  [CSV](./corrected_v2/prd-compliance-20261103-b/m5_feature_importance.csv)

  and

  [Markdown report](./corrected_v2/prd-compliance-20261103-b/m5_feature_importance.md).

- [x] **Real upload-to-prediction integration:** Upload run

  `upload_20261103T175330107104Z_ccb360caec54` processed 176,701 traffic rows

  through M1–M3 with explicit weather/calendar fallbacks. Current M5 models

  produced all 16 target/horizon outputs for all 25 roads, with zero skipped

  roads and retained `as_of`/model-run lineage.

- [x] **Intervals and calibrated classifier outputs:** Upload predictions

  returned finite ordered split-conformal intervals for both regression

  targets and held-out sigmoid-calibrated classifier scores. This verifies

  execution, not nominal coverage under time dependence or distribution

  shift.

- [x] **Raw-data integrity:** All three raw-file SHA-256 values match the

  completed run manifest after the pipeline and upload checks.

- [x] **Repository hygiene and baseline preservation:** Work remains on

  `feature/prd-compliance-improvements` from baseline `dd3d20a`; no commit,

  push, or merge was made. Raw/generated data, trained binaries, row

  quarantines, controlled source documents, executed notebooks, and

  environments remain ignored.

### PRD acceptance gates that remain open

- [ ] **Congestion macro-F1 ≥ 0.80:** unmet at all horizons

  (0.7332–0.7592).

- [ ] **Accident-risk ROC-AUC ≥ 0.75:** unmet at all horizons

  (0.3876–0.5578).

- [ ] **Deep model beats classical RMSE:** unmet on matched untouched test

  rows at all horizons; no test metric was used for model selection.

- [ ] **Overall PRD acceptance:** remains **PARTIAL**. Uninterrupted

  one-command reproduction, interval coverage guarantees for dependent time

  series, definitive sensor-fault adjudication, event timestamps, live feeds,

  style linting, and Windows/Linux verification remain incomplete or

  unavailable. Dedicated accident-risk EDA has now been completed and verified.

**Release decision:** FlowCast v1.0-rc1 remains a technical release candidate

for controlled evaluation, not a PRD-accepted production release. The

congestion, accident-risk, and deep-versus-classical performance targets

remain unmet and are reported without modification. Keep overall PRD

acceptance **PARTIAL** until failed targets are resolved or formally waived

and all remaining checks are complete.