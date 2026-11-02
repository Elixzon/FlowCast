# FlowCast — Final Technical Report

## 1. Executive summary

This report presents results from corrected run `20261102T172452`, executed
from raw inputs with one command in a clean Python 3.11.18 environment. The
versioned M1–M6 pipeline completed; M5 evaluated all 16 target/horizon
combinations on untouched chronological test data, and M6 trained and compared
LSTM, GRU, and Bidirectional LSTM models on identical matched records. The run
generated 28 model cards and 12 actual training histories/loss curves.

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
correlations.

The EDA does not provide a complete statistical/IQR outlier analysis or a
dedicated accident-risk target analysis. Those remain gaps rather than
assumed-complete analyses.

## 4. Evaluation design

### 4.1 Classical models (M5)

M5 evaluates Linear Regression, NumPy Gradient Descent, Decision Tree, Random
Forest, XGBoost, and Linear SVM candidates as appropriate to each task.
Preprocessing is fitted on training rows only. Model and threshold selection
uses only the early chronological validation block; a later validation block
is reserved for regression interval calibration. All 16 model cards mark the
test partition unused for selection.

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

All metrics below come from run `20261102T172452`'s untouched test partitions.
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
| Congestion | 30m | Random Forest | Accuracy 0.8645; macro-F1 0.7651 |
| Congestion | 60m | Random Forest | Accuracy 0.8548; macro-F1 0.7463 |
| Congestion | 90m | XGBoost | Accuracy 0.8550; macro-F1 0.7535 |
| Congestion | 120m | XGBoost | Accuracy 0.8407; macro-F1 0.7326 |
| Accident risk | 30m | XGBoost (shallow) | ROC-AUC 0.6124; PR-AUC 0.02025; positive F1 0.0564 |
| Accident risk | 60m | XGBoost (shallow) | ROC-AUC 0.5994; PR-AUC 0.02082; positive F1 0.0378 |
| Accident risk | 90m | Random Forest | ROC-AUC 0.5500; PR-AUC 0.01906; positive F1 0.0308 |
| Accident risk | 120m | XGBoost | ROC-AUC 0.5578; PR-AUC 0.01282; positive F1 0.0000 |

Traffic-volume MAPE meets the PRD's ≤12% criterion at every horizon. All four
congestion macro-F1 scores are below 0.80; all four accident-risk ROC-AUC
scores are below 0.75. Incident labels are highly imbalanced: for 30m, the
valid training rows contain 651 positives among 71,492 records, and the
validation-selection block contains 79 among 10,195. Class weighting and
multiple XGBoost weight/depth candidates were used with training/validation
data only. The resulting ranking and positive-class F1 remain too weak for
operational incident predictions.

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

## 7. Model cards, uncertainty, and reproducibility artifacts

The corrected run contains 28 model cards: 16 for selected M5 task/horizon
models and 12 for M6 architectures/horizons. Cards record model purpose,
features, chronological periods, validation/test metrics, hyperparameters,
artifact paths, and limitations. The run also generated 12 training-history
CSVs and 12 loss-curve figures from actual fitting runs. The standalone
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
coverage is not guaranteed for these time series. Classification probabilities
and SVM decision scores are not independently calibrated.

The runner records input checksums, outputs, stage status, and checkpoints.
All eight stages are done for run `20261102T172452`. Artifacts are under
`reports/corrected_v2/20261102T172452/`,
`data/processed/corrected_v2/20261102T172452/`, and
`models/*/corrected_v2/20261102T172452/`. One clean `python -m src.run_pipeline`
command completed from raw data. Setup and resume behavior are documented in
the [README](../README.md).
The final release review verified the completed manifest and reproduction
CLI in a fresh Python 3.11 environment; it did not repeat the full training
run, preserving the frozen results and artifacts.

## 8. Dashboard and automated testing

Streamlit AppTest exercised all 11 routes: Live Predictions, Historical
Trends, Congestion Heatmap, Road Comparison, Model Performance, Feature
Importance, Forecast Visualization, Prediction Confidence, Weather vs Traffic,
Data Validation, and Model Training. All completed without exceptions.

The Live Predictions page completed, including model loading and batch
inference, in **14.410 seconds** on Python 3.11.18. The output exposes
model-run/artifact lineage, regression intervals, and classifier scores.

The final release review reran the automated suite on Python 3.11.18:
**26 passed in 144.26 seconds**, including AppTest of all 11 dashboard routes.
It also reran a real upload integration using
only `traffic_sensor_log.csv` under upload run
`upload_20261102T130747549983Z_ccb360caec54`. M1–M3 processed 176,701 rows;
project weather/calendar files were recorded as fallbacks. Saved M5 models
produced all four targets for all 25 roads, with zero skipped roads and
model-run/`as_of` lineage. Both regression targets returned finite conformal
bounds containing their point forecasts. The three raw-file hashes remained
unchanged.

The dashboard still uses historical observations because no live feed is
configured. Upload prediction does not retrain models; it uses the latest
persisted M5 run, as required for inference without on-the-fly training.

## 9. Environment and limitations

The clean verified environment was Python 3.11.18, TensorFlow 2.16.2 (CPU),
NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1, and XGBoost 2.1.4, installed
from the pinned `requirements.txt`. In the final review, a fresh Python 3.11.18
environment passed `pip check` and ran the full test suite and dashboard
AppTests. The full pipeline run completed on a 16-GiB macOS workstation.
Windows/Linux compatibility and a second independent run for metric-by-metric
determinism were not tested. The handover initialized a local `main` branch,
but no commit or remote was created; run manifests and raw hashes remain the
artifact provenance.

Other material limitations:

- Congestion and accident-risk acceptance thresholds are not met.
- Sequence models do not improve matched test RMSE over classical models.
- Statistical/IQR outlier analysis and dedicated accident-risk EDA are absent.
- Classifier probabilities are not calibrated. Some deep-model test interval
  coverage fell below the nominal 90%; time-series conformal coverage is not
  guaranteed.
- No live traffic/weather feed or event-publication timestamp is supplied.

## 10. PRD acceptance summary

| Criterion | Status | Result |
|---|---|---|
| Volume MAPE ≤ 12% | PASS | 9.8569%–10.4251% across four horizons. |
| Congestion macro-F1 ≥ 0.80 | FAIL | 0.7326–0.7651; all horizons miss. |
| Accident-risk ROC-AUC ≥ 0.75 | FAIL | 0.5500–0.6124; all horizons miss. |
| Deep sequence model beats best classical volume RMSE | FAIL | The selected recurrent model loses by 1.6032–15.2188 RMSE on matched test rows. |
| One-command full reproduction from raw data | PASS | Eight stages completed in a clean Python 3.11.18 environment. |
| All nine required dashboard views | PASS | All 11 dashboard routes passed AppTest. |
| Batch inference ≤ 30 seconds | PASS | Live Predictions completed in 14.410 seconds. |
| Upload file and trigger predictions | PASS | Traffic-only upload processed through M1–M3 and generated all four targets for 25 roads using persisted models. |
| Validation-based calibrated prediction intervals | PARTIAL | Split-conformal intervals are implemented; test coverage varies, and serial dependence prevents a nominal guarantee. |
| Python 3.11 runtime compatibility | PASS | Pinned environment, full pipeline, 26 tests, and dashboard route tests passed. |

**Technical release readiness: PASS for FlowCast v1.0-rc1 controlled
evaluation. PRD acceptance: PARTIAL — do not claim overall acceptance.**
The clean reproduction evidence, final Python 3.11 test rerun, all dashboard
routes, persisted models, model cards/curves, corrected feature-importance
export, and actual upload predictions are verified. Congestion, accident-risk,
and deep-versus-classical performance thresholds remain unmet; outlier
analysis, time-series coverage guarantees, repeat-run determinism, style
linting, live feed availability, and cross-platform verification remain
partial or unavailable. See the
[detailed audit](./final_project_audit.md) for the release checklist and
FR-01–FR-18/NFR status.
