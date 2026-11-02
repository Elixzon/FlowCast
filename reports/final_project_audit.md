# FlowCast Project Audit — Corrected Python 3.11 Run

## Audit scope and evidence

This audit covers the approved FlowCast PRD, source code, raw inputs, corrected
pipeline outputs, model evaluations/cards, dashboard behavior, and automated
tests. It supersedes earlier audit conclusions based on legacy metrics.

The verified run is `20261102T172452`. It started from the supplied raw CSVs
with one `python -m src.run_pipeline` invocation in a clean Python 3.11.18
virtual environment. Its manifest records every M1–M6 stage as complete; all
three raw-input SHA-256 values still match `data/raw/`. Corrected outputs are
versioned separately from historical artifacts. An earlier fresh-run attempt
exposed a categorical-preprocessing bug at M5; that failed run is preserved,
the bug was fixed, and the verified run was restarted from raw data.

Evidence: [run manifest](./corrected_v2/20261102T172452/run_manifest.json),
[M5 scoreboard](./corrected_v2/20261102T172452/m5_corrected_test_scoreboard.csv),
[M6 comparison](./corrected_v2/20261102T172452/m6_corrected_head_to_head.md),
and [model cards](./corrected_v2/20261102T172452/model_cards/).

Status definitions: **PASS** means verified; **PARTIAL** means some required
behavior or evidence remains incomplete; **FAIL** means a stated acceptance
threshold was not met; **NOT VERIFIED** means the available environment or
evidence cannot establish the requirement.

## Verified results

- Corrected M1–M6 pipeline stages completed under Python 3.11.18 with raw-file
  hashes unchanged; the run processed 176,701 merged traffic records on a
  16-GiB workstation.
- M5 evaluated 16 selected task/horizon models on chronological, untouched
  test partitions. Selection and thresholds used only early validation data;
  regression intervals were calibrated on a later validation block.
- M6 trained LSTM, GRU, and Bidirectional LSTM models for all four horizons.
  Classical/deep selection and comparisons use identical matched validation
  and test records per horizon: 2,673 (30m), 2,602 (60m), 2,522 (90m), and
  2,447 (120m) for the frozen test evaluation.
- Generated 28 model cards (16 M5 and 12 M6), 12 training-history CSVs, and
  12 loss-curve figures from actual training.
- Automated tests: **26 passed in 144.26 seconds** on Python 3.11.18 after
  adding a durable AppTest covering all 11 dashboard routes.
- Streamlit AppTest exercised all 11 dashboard routes with no exceptions.
  Live Predictions completed in **14.410 seconds**, below the PRD's 30-second
  batch target.
- An actual traffic-only upload used the raw weather/calendar fallback,
  completed M1–M3, and generated predictions for all 25 roads for all four
  M5 targets; raw inputs remained unchanged.
- Nominal 90% split-conformal intervals were calibrated on later validation
  residuals. Test coverage is reported below; time dependence means nominal
  coverage is not guaranteed.

## PRD functional-requirement acceptance

| Requirement | Status | Evidence / remaining limitation |
|---|---|---|
| FR-01 — Ingest raw traffic, weather, and calendar data | PASS | Corrected M1–M6 run completed from the project raw inputs; raw hashes are preserved in the manifest. |
| FR-02 — Validate schema/types/ranges and record rejected rows | PASS | M1 report, issue register, and row-level quarantine outputs were generated. |
| FR-03 — Resolve missing readings, duplicates, and sensor faults | PARTIAL | Corrected M2 removed 1,767 duplicate rows, kept invalid measurements missing, and reported unresolved fields. A separate statistical/IQR sensor-fault outlier analysis is not evidenced. |
| FR-04 — Merge sources by segment and time | PASS | M2 merged 176,701 traffic rows with weather/calendar data; no unmatched weather/calendar keys. |
| FR-05 — Generate lag, rolling, time, and weather features | PASS | Corrected M3 produced 59 features with exact-time, road-specific lags/rolling windows. `event_proximity_flag` is excluded from operational predictors because event-announcement times are unavailable. |
| FR-06 — Persist versioned data and quality reports | PASS | Corrected run artifacts and quality reports are isolated under the versioned run ID. |
| FR-07 — Train required regression families | PASS | Corrected M5 candidate results include Linear Regression, Decision Tree, Random Forest, XGBoost, and NumPy Gradient Descent across the requested regression targets/horizons. |
| FR-08 — Train congestion/accident classifiers including SVM | PASS | Corrected M5 candidate records include Linear SVM and tree/boosting classifiers for the classification targets. |
| FR-09 — Train recurrent models from scratch | PASS | LSTM/GRU and comparison-only Bidirectional LSTM were trained for all four horizons without pretrained weights. |
| FR-10 — Produce the four forecast targets by segment/window | PASS | Corrected M5 covers all 16 target/horizon combinations; dashboard inference emits segment-level outputs. |
| FR-11 — Attach uncertainty/confidence to each forecast | PARTIAL | Regression models have split-conformal 90% intervals; all M5 regression test coverages were 0.917–0.987. Some M6 test coverages fell below 0.90 (0.867–0.890). Classifier scores are not independently calibrated, and temporal dependence/shift invalidate a distribution-free coverage claim. |
| FR-12 — Persist and reload trained models | PASS | M5 and M6 estimators are saved and loaded by the dashboard without retraining. |
| FR-13 — Show live/near-term segment predictions | PARTIAL | Near-term predictions render, but are based on the latest stored historical observations; no live sensor feed is configured. |
| FR-14 — Historical trends, congestion heatmaps, road comparisons | PASS | All three requested views rendered on corrected outputs in AppTest. |
| FR-15 — Performance, feature importance, forecast visualizations | PASS | All requested views rendered on corrected outputs in AppTest. |
| FR-16 — Weather-versus-traffic view | PASS | View rendered on corrected outputs in AppTest. |
| FR-17 — Upload a new file and trigger a prediction run | PASS | A real traffic-only upload was validated, cleaned, and feature-engineered through M1–M3; saved M5 models generated volume, travel-time, congestion, and accident-risk outputs for all 25 roads. Weather/calendar fallbacks were recorded, `as_of` was retained, and raw files were unchanged. |
| FR-18 — Export selected-range insights | PASS | Selected-range report export is implemented and covered by automated tests. |

## PRD success criteria and non-functional checks

| Criterion | Status | Verified result |
|---|---|---|
| Volume forecast MAPE ≤ 12% on hold-out | PASS | 9.8569%, 10.1204%, 10.4251%, and 10.3468% at 30/60/90/120m. |
| Congestion macro-F1 ≥ 0.80 | FAIL | 0.7651, 0.7463, 0.7535, and 0.7326; none meets the threshold. |
| Accident-risk ROC-AUC ≥ 0.75 | FAIL | 0.6124, 0.5994, 0.5500, and 0.5578; all remain below target. |
| Sequence model beats best classical volume RMSE | FAIL | Validation-selected GRU/LSTM RMSE is worse than the matched classical model by 1.6032, 7.1501, 9.0527, and 15.2188 at 30/60/90/120m. |
| Full pipeline reproduction from raw data | PASS | One clean Python 3.11.18 invocation completed all eight stages; manifest and raw hashes verified. |
| All nine required dashboard views render | PASS | All nine analytics views plus Data Validation and Model Training (11 routes) passed AppTest. |
| Full-corridor inference ≤ 30 seconds | PASS | Live Predictions including model loading and batch inference completed in 14.410 seconds. |
| Python 3.11 compatibility | PASS | Clean Python 3.11.18 environment installed the pinned requirements, passed all 26 tests, and completed the full pipeline. |
| Event-feature as-of availability | PARTIAL | No event-announcement timestamps are supplied, so event proximity is deliberately excluded from operational predictors. |

## PRD non-functional requirements

| Requirement | Status | Evidence / limitation |
|---|---|---|
| Performance ≤ 30 seconds | PASS | Live Predictions completed in 14.410 seconds. |
| Scalability ≥ 150,000 rows on 16 GB | PASS | Full 176,701-row pipeline completed on a 16-GiB machine; peak RSS was not separately profiled. |
| Deterministic clean rerun | PARTIAL | Seeds are recorded and the clean from-raw run completed. A second independent run was not performed to compare metric reproducibility. |
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

**Overall: PARTIAL — not PRD acceptance-complete.** The clean Python 3.11
reproduction, upload-to-prediction path, dashboard, tests, and artifacts are
verified. Acceptance is still blocked by failed congestion, accident-risk,
and deep-versus-classical thresholds, with outlier handling, probabilistic
calibration, event timing, repeatability, style linting, and cross-platform
verification explicitly partial above. No historical metric is used as a
verified final result.

## Final release checklist — FlowCast v1.0-rc1

Review date: 2026-11-02. This checklist separates technical readiness from
PRD performance acceptance.

### Technical release checks

- [x] **Python 3.11 environment:** A fresh Python 3.11.18 environment was
  installed from `requirements.txt`; `pip check` reported no broken
  requirements, and the reproduction CLI help ran successfully. The
  workspace `.venv` is Python 3.12.13 and was not used for these checks.
- [x] **Automated tests:** `python -m pytest -q` in the clean Python 3.11
  environment: **26 passed in 144.26 seconds**, including the 11-route
  dashboard AppTest.
- [x] **Dashboard AppTests:** All 11 routes passed Streamlit AppTest without
  exceptions, including Live Predictions, Feature Importance, Prediction
  Confidence, and Data Validation, using the current persisted run and its
  real processed data.
- [x] **One-command reproduction evidence:** Run
  `20261102T172452` has all eight M1–M6 pipeline stages marked `done`, uses
  Python 3.11.18, and is the current completed-run pointer. The clean-runtime
  CLI was verified in this review. The full training pipeline was not rerun
  for this checklist; the existing frozen run and results were preserved.
- [x] **Model and report inventory:** All 16 selected M5 model artifacts,
  four M5 validation-candidate artifacts, 12 M6 model artifacts, 28 model
  cards, 12 training-history CSVs, and 12 training-curve figures exist.
  Corrected M4 EDA, M5 scoreboard, M6 matched-comparison reports, and run
  manifest are present. The dashboard Feature Importance view uses the
  persisted corrected model and passed AppTest.
- [x] **Standalone feature-importance exporter:** Uses each selected
  corrected M5 artifact and its saved training-fitted preprocessor. Generated
  run-versioned CSV/Markdown outputs with 720 source-feature rows (45 per
  each of 16 task/horizon combinations); importance values are finite,
  non-negative, and nonzero per group. Existing historical root-level
  feature-importance reports were left unchanged. See the
  [CSV](./corrected_v2/20261102T172452/m5_feature_importance.csv) and
  [Markdown report](./corrected_v2/20261102T172452/m5_feature_importance.md).
- [x] **Real upload-to-prediction integration:** Review upload run
  `upload_20261102T130747549983Z_ccb360caec54` processed 176,701 traffic rows
  through M1–M3, using the project weather/calendar files as explicit
  fallbacks. Each of the four M5 targets produced predictions for all 25
  roads, with zero skipped roads and retained `as_of`/model-run lineage.
- [x] **Prediction intervals:** The live upload test verified finite
  split-conformal lower/upper bounds that contain each point forecast for
  both regression targets. This verifies interval execution, not nominal
  coverage under time dependence or distribution shift.
- [x] **Raw-data integrity:** SHA-256 hashes for all three source CSVs match
  the completed-run manifest after the upload test; no raw dataset changed.
- [x] **GitHub-ready hygiene:** A local `main` branch is initialized.
  `.gitignore` checks confirmed raw/generated data, trained binaries, row
  quarantines, controlled PRD documents, executed notebooks, and environments
  are excluded, while final and historical reports remain includable. No
  commit or remote was created.

### PRD acceptance gates that remain open

- [ ] **Congestion macro-F1 ≥ 0.80:** unmet at all horizons (0.7326–0.7651).
- [ ] **Accident-risk ROC-AUC ≥ 0.75:** unmet at all horizons (0.5500–0.6124).
- [ ] **Deep model beats classical RMSE:** unmet on matched untouched test
  rows at all horizons; no metrics were tuned or changed during this review.
- [ ] **Overall PRD acceptance:** remains **PARTIAL**. Classifier
  calibration, time-series interval coverage guarantees, outlier analysis,
  event timestamps, live feeds, repeat-run determinism, style linting, and
  Windows/Linux verification remain incomplete or unavailable.

**Release decision:** The executable technical checks pass, so
**FlowCast v1.0-rc1 is prepared as a technical release candidate for
controlled evaluation** on the verified Python 3.11/macOS setup. This is not
PRD acceptance or a claim that the model-performance targets pass. Keep the
overall acceptance status **PARTIAL** until the failed thresholds and listed
gaps are resolved or explicitly waived.
