# FlowCast — Final Technical Report

## 1. Executive summary

This report presents the corrected raw-data evaluation run `prd-compliance-20261103-b`

and a separate M5-only recalibration completed on November 4, 2026. Run B was

generated from raw inputs under Python 3.11.18; its versioned M1–M6 pipeline

completed and M5 evaluated all 16 target/horizon combinations on untouched

chronological test data. The November 4 run regenerated all 12 M5 task/horizon

models using a revised accident-risk validation protocol. M6 trained and compared

LSTM, GRU, and Bidirectional LSTM models on identical matched records. The run

generated 32 model cards, 40 actual training histories/loss curves, and

validation-selected 24/48/96-step sequence-sweep results.

The selected model choices and recorded test metrics match the independent

`prd-compliance-20261103-a` run. Run B completed all ten stages after a safe

resume at the sequence-sweep stage following an external interruption; an

uninterrupted single-invocation run of the current ten-stage pipeline remains

unverified.

The project is **not PRD acceptance-complete**. Traffic-volume MAPE passes its

threshold at all horizons, but congestion macro-F1 and accident-risk ROC-AUC

miss their targets. The validation-selected recurrent model loses to the

classical model on test RMSE at every horizon. Dashboard routes, inference

timing, and automated tests passed; remaining functional and environment gaps

are listed in Section 10 and in the

[final project audit](./final_project_audit.md).

## 2. Problem and scope

FlowCast provides segment-level forecasts for traffic volume, congestion

level, travel time, and accident risk on the Northline Corridor. M5 forecasts

each target 30, 60, 90, and 120 minutes ahead. M6 is a traffic-volume sequence

benchmark using 48 historical half-hour steps. The system is decision support,

not a traffic-signal controller.

## 3. Data and corrected preprocessing

### 3.1 Sources

The raw inputs contain 178,468 traffic rows, 10,872 hourly weather rows, and

151 calendar dates, covering 25 road segments from January 1 through May 31,

2025. The verified run records SHA-256 values for all three files; recomputing

the hashes after the pipeline and upload integration confirmed raw inputs were

unchanged.

### 3.2 Validation, cleaning, and integration

M1 generated a machine-readable validation report, issue register, and

row-level quarantine outputs. The input review found duplicate traffic

road/time keys, missing sensor readings, blank congestion labels, and

out-of-range volume/speed/occupancy measurements. M2 removed 1,767 duplicate

traffic rows and retained 176,701 traffic rows. Invalid measurements were

marked missing rather than preserved as valid measurements. It retained 4,587

missing volume targets, reported 4,588 rows with unresolved model fields, and

matched all traffic rows to weather/calendar keys (zero unmatched keys).

Missing future targets were not synthesized. Historical features use exact

road/timestamp lookup, so a missing half-hour observation does not become a

false adjacent lag. The model pipeline excludes `event_proximity_flag` because

the source has no event-announcement timestamp with which to prove

point-in-time availability.

The original clean-run smoke test exposed that the shared preprocessing stage

treated the newly enabled current congestion observation as numeric. It was

added to the categorical feature list; the failed run was retained, and the

final verification was restarted from raw data.

### 3.3 Feature engineering and EDA

Corrected M3 produced 59 features over 176,701 rows. It includes cyclic time

encodings, weekend/peak/weather indicators, road-specific lags and rolling

statistics, capacity features, and vehicle/weather inputs. Exact-time lag and

rolling warm-up gaps remain null rather than being fabricated. M4 generated a

corrected 176,701-row, 60-column EDA report and ten figures for distributions,

time patterns, road comparisons, weather relationships, congestion, and

correlations. A separate raw-data quality report applies per-road/station IQR

and absolute-z-score flags to selected fields. These are review-only

statistical candidates, not an adjudicated sensor-fault detector; no

statistical outlier is automatically removed. Dedicated accident-risk target

EDA remains a gap.

## 4. Evaluation design

### 4.1 Classical models (M5)

M5 evaluates Linear Regression, NumPy Gradient Descent, Decision Tree, Random

Forest, XGBoost, and Linear SVM candidates as appropriate to each task.

Preprocessing is fitted on training rows only. Model and threshold selection

uses only chronological validation data; the test partition is not used for

selection. For regression and congestion, the existing later validation block

is reserved for interval/probability calibration. In the November 4 accident-

risk recalibration, validation is split into three ordered blocks: the first

60% for model selection, the next 20% for sigmoid calibration, and the final

20% for operating-threshold selection. This prevents threshold selection from

preceding calibration in time. The four recalibrated accident-risk artifacts

persist the calibrator, selected threshold, row counts, and date ranges.

For the 30-minute M5 split, the training period ends April 16, 2025 at 15:30,

validation ends May 1 at 18:00, and test begins May 1 at 19:00. Rows removed

for target/feature availability explain why test counts vary slightly by

horizon; all test periods remain later than their train/validation periods.

### 4.2 Sequence models (M6)

M6 trained 12 models from scratch: LSTM, GRU, and Bidirectional LSTM for each

of four horizons. Sequences contain 48 consecutive half-hour records, preserve

road boundaries, and do not cross split boundaries. Numeric imputation,

scaling, and one-hot encoding are fitted on training data only; target values

are not imputed. The raw M5/M6 issue-time predictor columns match, while each

family uses its own training-fitted transformation. The Bidirectional LSTM is

included as a historical comparison, not selected for operational forecasts.

Classical and operational deep candidates were selected using the same

matched early-validation records at each horizon. The final chronological

validation block was used only for conformal calibration. Every classical,

LSTM, GRU, and Bidirectional LSTM comparison uses the same frozen test rows;

metadata marks test records as unused for training or selection.

## 5. Corrected M5 test results

The traffic-volume and congestion metrics below, and the accident-risk

ranking metrics, are from the untouched chronological test partitions used

for Run B and reproduced by the November 4 M5 recalibration. The November 4

recalibration used the revised three-block accident-risk validation protocol;

its test evaluation did not use test data for model or threshold selection.

The independent Run A recorded the same earlier selected models and test

metrics.

Rows per horizon are 20,302 (30m), 20,297 (60m), and 20,274 (90m and 120m).

| Target | Horizon | Selected model | Test metrics |

|---|---:|---|---|

| Traffic volume | 30m | XGBoost | MAE 40.7033; RMSE 59.2581; MAPE 9.8569%; R² 0.9574 |

| Traffic volume | 60m | XGBoost | MAE 41.5231; RMSE 60.0879; MAPE 10.1204%; R² 0.9563 |

| Traffic volume | 90m | XGBoost | MAE 42.3389; RMSE 61.1300; MAPE 10.4251%; R² 0.9550 |

| Traffic volume | 120m | Random Forest | MAE 42.5399; RMSE 62.3832; MAPE 10.3468%; R² 0.9529 |

| Travel time | 30m | XGBoost | MAE 0.4476; RMSE 1.0718; MAPE 9.5962%; R² 0.8238 |

| Travel time | 60m | XGBoost | MAE 0.4503; RMSE 1.0881; MAPE 9.6324%; R² 0.8207 |

| Travel time | 90m | XGBoost | MAE 0.4574; RMSE 1.0743; MAPE 9.9909%; R² 0.8268 |

| Travel time | 120m | XGBoost | MAE 0.4675; RMSE 1.0842; MAPE 10.4657%; R² 0.8203 |

| Congestion | 30m | Extra Trees | Accuracy 0.8618; macro-F1 0.7592 |

| Congestion | 60m | Random Forest | Accuracy 0.8583; macro-F1 0.7461 |

| Congestion | 90m | Extra Trees | Accuracy 0.8624; macro-F1 0.7571 |

| Congestion | 120m | XGBoost | Accuracy 0.8508; macro-F1 0.7332 |

| Accident risk | 30m | XGBoost (shallow) | ROC-AUC 0.3876; PR-AUC 0.00813; positive F1 0.0093 |

| Accident risk | 60m | XGBoost (shallow) | ROC-AUC 0.4006; PR-AUC 0.00910; positive F1 0.0064 |

| Accident risk | 90m | Extra Trees | ROC-AUC 0.5061; PR-AUC 0.01178; positive F1 0.0281 |

| Accident risk | 120m | XGBoost | ROC-AUC 0.5578; PR-AUC 0.01282; positive F1 0.0284 |

Traffic-volume MAPE meets the PRD's ≤12% criterion at every horizon. All four

congestion macro-F1 scores are below 0.80; all four accident-risk ROC-AUC

scores are below 0.75. Class weighting and multiple tree/boosting candidates

were selected with training and validation data only. Despite these

imbalance-aware experiments, accident ranking and positive-class F1 remain

too weak for operational incident predictions.
5.1 November 4 accident-risk recalibration
The M5-only recalibration regenerated all four accident-risk horizon models
from the existing Run B chronological split files. Its three non-overlapping
validation blocks were used in forward order: model selection, sigmoid
probability calibration, and operating-threshold selection. The persisted
calibration block contained 1,842, 1,844, 1,843, and 1,843 rows at 30/60/90/120
minutes respectively. The threshold-selection block contained 2,040, 2,027,
2,022, and 2,026 rows.
Horizon	Selected threshold	Threshold-selection period
30m	0.0106923	2025-04-28 18:00 to 2025-05-01 18:00
60m	0.0090091	2025-04-28 17:30 to 2025-05-01 17:30
90m	0.0103449	2025-04-28 17:00 to 2025-05-01 17:00
120m	0.0172529	2025-04-28 16:30 to 2025-05-01 16:30


All four serialized accident-risk models contain a
`CalibratedClassifierCV` sigmoid calibrator and the corresponding threshold,
period, and row-count metadata. The report directory contains the four
confusion matrices, four combined ROC–PR figures, and four reliability
diagrams. These checks verify protocol execution, persistence, and diagnostic
generation; they do not imply useful accident-risk discrimination. ROC-AUC
remains 0.3876–0.5578, below the PRD target of 0.75, and positive-class
precision/recall remain poor under the severe class imbalance.

## 6. Corrected M6 matched comparison

Classical candidates and LSTM/GRU winners were selected on the same matched

early-validation records at each horizon. Every model below was evaluated on

the identical held-out test rows. The Bidirectional LSTM remains a

historical-only comparison.

| Horizon | Test rows | Model | Test RMSE | 90% interval coverage |

|---:|---:|---|---:|---:|

| 30m | 2,673 | Classical XGBoost | 59.7493 | 0.9300 |

| 30m | 2,673 | LSTM | 70.9926 | 0.9095 |

| 30m | 2,673 | GRU (selected deep) | 61.3525 | 0.9226 |

| 30m | 2,673 | Bidirectional LSTM | 60.7992 | 0.9379 |

| 60m | 2,602 | Classical XGBoost | 58.2218 | 0.9500 |

| 60m | 2,602 | LSTM (selected deep) | 65.3720 | 0.8909 |

| 60m | 2,602 | GRU | 64.7338 | 0.9101 |

| 60m | 2,602 | Bidirectional LSTM | 62.6850 | 0.9385 |

| 90m | 2,522 | Classical Random Forest | 61.0058 | 0.9766 |

| 90m | 2,522 | LSTM (selected deep) | 70.0585 | 0.8894 |

| 90m | 2,522 | GRU | 67.3656 | 0.9227 |

| 90m | 2,522 | Bidirectional LSTM | 68.7919 | 0.9314 |

| 120m | 2,447 | Classical XGBoost | 61.7110 | 0.9787 |

| 120m | 2,447 | LSTM (selected deep) | 76.9298 | 0.8672 |

| 120m | 2,447 | GRU | 72.9168 | 0.9109 |

| 120m | 2,447 | Bidirectional LSTM | 66.7733 | 0.9424 |

The selected deep-model RMSE exceeds matched classical RMSE by 1.6032, 7.1501,

9.0527, and 15.2188 at 30/60/90/120 minutes. Neither unidirectional nor

bidirectional recurrent models beat the comparator. All model families use

the same 45 raw issue-time predictors and same matched rows; the classical

model uses tabular features, while deep models additionally encode the prior

48-step sequence. Available complete sequences and matching reduce test

counts to 2,447–2,673 records. Results do not support a deep-learning

advantage.

### 6.1 Sequence-length sweep

The additional 24/48/96-step sweep selected the recurrent architecture and

sequence length using the shared early-validation block. For each horizon,

the classical and deep candidates in the sweep were evaluated on the same

matched test keys. The common-cohort counts are smaller than the primary

48-step M6 comparison and are reported separately; they do not replace it.

| Horizon | Matched test rows | Selected deep model | Deep RMSE | Classical RMSE | Deep minus classical |

|---:|---:|---|---:|---:|---:|

| 30m | 247 | LSTM · 24 steps | 67.1585 | 64.6889 | +2.4696 |

| 60m | 241 | GRU · 24 steps | 72.8820 | 62.9036 | +9.9784 |

| 90m | 235 | LSTM · 24 steps | 64.7793 | 60.2595 | +4.5198 |

| 120m | 229 | LSTM · 96 steps | 151.8561 | 72.3582 | +79.4979 |

No swept recurrent model beats the matched classical candidate. The very

small 120-minute sweep result is a test outcome, not a basis for further

selection or tuning.

## 7. Model cards, uncertainty, and reproducibility artifacts

The corrected run contains 32 model cards: 16 for selected M5 task/horizon

models, 12 for the primary M6 architectures/horizons, and four for the

sequence-sweep winners. Cards record model purpose, features, chronological

periods, validation/test metrics, hyperparameters, artifact paths, and

limitations. The run also generated 40 training-history CSVs and 40

loss-curve figures from actual fitting runs, including sequence-sweep

candidates. The standalone

`python -m src.feature_importance` command loads these corrected run artifacts

and their persisted preprocessors and writes feature-level CSV/Markdown

reports beside the run manifest. It generated 720 rows across 16 selected

task/horizon groups; grouped importance values were finite, non-negative,

and positive in every group. The historical root-level reports were not

overwritten.

Regression intervals use the finite-sample split-conformal order statistic of

absolute residuals from a later chronological validation block, after model

selection. M5 observed test coverage was 0.9269–0.9873 for volume and

0.9172–0.9431 for travel time. M6 coverages are listed in Section 6; selected

deep models fell below 0.90 at 60/90/120 minutes. Coverage is report-only and

was not used for model selection. The finite-sample guarantee requires

exchangeability; serial dependence and distribution shift mean that nominal

coverage is not guaranteed for these time series. Classifier probabilities use held-out sigmoid calibration. In the November

4 accident-risk recalibration, the calibration block follows model selection

and precedes the separate threshold-selection block. Test-set calibration

diagnostics are evaluation-only; they were not used for model or threshold

selection.

The runner records input checksums, outputs, stage status, and checkpoints.

All ten registered stages are done for run `prd-compliance-20261103-b`.

Artifacts are under `reports/corrected_v2/prd-compliance-20261103-b/`,

`data/processed/corrected_v2/prd-compliance-20261103-b/`, and

`models/*/corrected_v2/prd-compliance-20261103-b/`. The pipeline started from

raw data, but its initial invocation was externally interrupted after M6

comparison and safely resumed at `m6_sequence_sweep`. An uninterrupted

single-invocation run of the current ten-stage source is therefore not

verified. Setup and resume behavior are documented in the

[README](../README.md).

## 8. Dashboard and automated testing

Streamlit AppTest exercised all 11 routes: Live Predictions, Historical

Trends, Congestion Heatmap, Road Comparison, Model Performance, Feature

Importance, Forecast Visualization, Prediction Confidence, Weather vs Traffic,

Data Validation, and Model Training. All completed without exceptions.

The Live Predictions page completed, including model loading and batch

inference, in **23.67 seconds** on Python 3.11.18. The output exposes

model-run/artifact lineage, regression intervals, and classifier scores.

The final release review reran the automated suite on Python 3.11.18:

**41 passed in 43.72 seconds**, including the dashboard AppTest exercising

all 11 routes against the latest run. A separate instrumented AppTest also

rendered all 11 routes without exceptions. A real traffic-only upload under

`upload_20261103T175330107104Z_ccb360caec54` processed 176,701 rows through

M1–M3, using the project weather/calendar files as recorded fallbacks. The

current persisted M5 models generated all 16 task/horizon outputs for all 25

roads, with zero skipped roads and model-run/`as_of` lineage. Regression

outputs contained finite, ordered split-conformal bounds around the point

forecasts; classifier outputs included held-out sigmoid-calibrated scores. This is the
older upload execution check; a fresh real upload has not yet verified the
newly added prediction-persistence artifacts.

The three raw-file hashes remained unchanged.

The dashboard still uses historical observations because no live feed is

configured. Upload prediction does not retrain models; it uses the latest

persisted M5 run, as required for inference without on-the-fly training.

## 9. Environment and limitations

The clean verified environment was Python 3.11.18, TensorFlow 2.16.2 (CPU),

NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1, and XGBoost 2.1.4, installed

from the pinned `requirements.txt`. In the final review, a fresh Python 3.11.18

environment passed `pip check` and ran the full test suite and dashboard

AppTests. Two independent from-raw runs produced identical selected model

choices and recorded M5/M6 test metrics; wall-clock timings and artifact bytes

were not compared. The full pipeline run completed on a 16-GiB macOS

workstation. Windows/Linux compatibility is not verified. The work remains

on `feature/prd-compliance-improvements` from baseline `dd3d20a`; no commit,

push, or merge was made.

Other material limitations:

- Congestion and accident-risk acceptance thresholds are not met.

- Sequence models do not improve matched test RMSE over classical models.

- The read-only IQR/z-score review flags candidate outliers but does not

  adjudicate sensor faults or automatically remove valid peaks. Dedicated

  accident-risk EDA has now been completed in M4 (`src/eda.py` and

  `notebooks/FlowCast_M4_EDA.ipynb`), generating figures 11 and 12 and

  quantifying incident rates across corridor segments, weather conditions,

  and time-of-day.

- Classifier probabilities use held-out sigmoid calibration, but accident

  discrimination remains poor. Some deep-model test interval

  coverage fell below the nominal 90%; time-series conformal coverage is not

  guaranteed.

- No live traffic/weather feed or event-publication timestamp is supplied.

## 10. PRD acceptance summary

| Criterion | Status | Result |

|---|---|---|

| Volume MAPE ≤ 12% | PASS | 9.8569%–10.4251% across four horizons. |

| Congestion macro-F1 ≥ 0.80 | FAIL | 0.7332–0.7592; all horizons miss. |

| Accident-risk ROC-AUC ≥ 0.75 | FAIL | 0.3876–0.5578; all horizons miss. |

| Deep sequence model beats best classical volume RMSE | FAIL | The selected recurrent model loses by 1.6032–15.2188 RMSE on matched test rows. |

| Uninterrupted one-command full reproduction from raw data | PARTIAL | Run B completed all ten stages from raw data, but was resumed at `m6_sequence_sweep` after an external interruption. |

| All nine required dashboard views | PASS | All 11 dashboard routes passed AppTest. |

| Batch inference ≤ 30 seconds | PASS | Live Predictions completed in 23.67 seconds. |

| Upload file and trigger predictions | PASS | Traffic-only upload processed through M1–M3 and generated all four targets for 25 roads using persisted models. |

| Validation-based calibrated prediction intervals | PARTIAL | Split-conformal intervals are implemented; test coverage varies, and serial dependence prevents a nominal guarantee. |

| Python 3.11 runtime compatibility | PASS | Python 3.11.18, `pip check`, all 41 tests, and all 11 dashboard routes passed. |

**Technical release readiness: PASS for FlowCast v1.0-rc1 controlled

evaluation. PRD acceptance: PARTIAL — do not claim overall acceptance.**

The final Python 3.11 test rerun, all dashboard routes, persisted models,

model cards/curves, corrected feature-importance export, and actual upload

predictions are verified. An uninterrupted current-code single-invocation

reproduction remains unverified. Congestion, accident-risk, and deep-versus-classical

performance thresholds remain unmet; definitive sensor-fault adjudication,

time-series coverage guarantees, repeat-run determinism, style linting, live

feed availability, and cross-platform verification remain partial or

unavailable. See the

[detailed audit](./final_project_audit.md) for the release checklist and

FR-01–FR-18/NFR status.