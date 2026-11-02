# FlowCast

FlowCast is a corridor-level traffic forecasting and decision-support project.
It cleans traffic, weather, and calendar observations, creates issue-time and
historical features, trains classical and recurrent models, and presents
segment-level forecasts in a Streamlit dashboard. It is not a real-time feed
or a traffic-signal controller.

The supplied PRD and data dictionary are the sources of truth. They are
intentionally excluded from this public-ready tree because they may be
controlled project documents; obtain permission before sharing them.

## Release status

**FlowCast v1.0-rc1 is a technical release candidate for controlled
evaluation, not a PRD-accepted production release.** Verified results,
acceptance gaps, and release evidence are maintained in
[the final audit](reports/final_project_audit.md) and
[the technical report](reports/final_technical_report.md).

The corrected final run is `20261102T172452`. It completed all eight pipeline
stages from raw inputs in a clean Python 3.11.18 environment. The model
evaluation used chronological train/validation/test partitions; candidate
selection and calibration did not use the test set. The test metrics below
are the frozen results from that run, not values recalculated during this
handover.

| Target / criterion | Horizon results (30 / 60 / 90 / 120 minutes) | PRD status |
|---|---|---|
| Traffic-volume MAPE, target ≤ 12% | 9.8569% / 10.1204% / 10.4251% / 10.3468% | PASS |
| Congestion macro-F1, target ≥ 0.80 | 0.7651 / 0.7463 / 0.7535 / 0.7326 | FAIL |
| Accident-risk ROC-AUC, target ≥ 0.75 | 0.6124 / 0.5994 / 0.5500 / 0.5578 | FAIL |
| Deep vs. classical matched test RMSE | Deep model is worse by 1.6032 / 7.1501 / 9.0527 / 15.2188 | FAIL |

The M6 comparisons use identical held-out test records for classical and
sequence models. All 11 dashboard routes and 26 automated tests passed in the
release review. The dashboard renders stored historical observations because
the supplied data has no live sensor feed. Accident-risk scores are weak and
not calibrated; do not use them as dependable incident predictions.

## Architecture

```text
Raw traffic / weather / calendar CSVs
                 |
                 v
 M1 validation and quarantine reports
                 |
                 v
 M2 cleaning, deduplication, source joins
                 |
                 v
 M3 issue-time, temporal, weather and lag features
                 |
          +------+------+
          |             |
          v             v
 M4 EDA/reporting   M5 chronological splits,
                    fitted preprocessing, classical models
                          |
                          v
                    M6 48-step LSTM/GRU/
                    BiLSTM comparison
                          |
                          v
       Versioned reports, model cards, curves and artifacts
                          |
                          v
              Streamlit dashboard / upload inference
```

### Repository map

```text
dashboard/      Streamlit application
data/raw/       Required local source CSVs; ignored by Git
data/interim/   Generated cleaned and upload data; ignored by Git
data/processed/ Versioned features, splits and sequences; ignored by Git
models/         Generated model binaries; ignored by Git
notebooks/      EDA source notebook (executed output copy is ignored)
reports/        Audits, versioned metrics, model cards and training curves
references/     Controlled PRD/data-dictionary documents; ignored by Git
src/            Pipeline stages, modeling, inference and feature-importance tool
tests/          Automated data-integrity and corrected-pipeline tests
```

The canonical pipeline is `src/run_pipeline.py`; it creates isolated
`corrected_v2/<run-id>` outputs and updates `reports/corrected_v2/latest.json`
only after a run succeeds. The latest pointer provides paths to the run
manifest, processed data, models, and reports. Legacy experiment scripts and
reports remain in the tree for historical context; do not use their metrics as
the corrected final evaluation.

## Requirements and installation

- Python **3.11** (the verified runtime was 3.11.18)
- macOS or Linux for the verified setup; Windows has not been tested
- About 16 GB RAM for the complete pipeline; training runtime depends on CPU
- The three supplied raw CSV files (not included in a public GitHub repository)

From the project root on macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python --version
python -m pip check
```

The verified setup is macOS/Linux. Windows compatibility and installation
commands are not verified.

Dependencies are pinned in [`requirements.txt`](requirements.txt), including
CPU TensorFlow. CUDA is not required.

## Configuration and input data

`config.yaml` records project metadata, the time zone, source/interim/processed
directories, join keys, and the intended chronological validation policy.
The cleaning and feature stages read the configured time zone and base
directories; the corrected runner redirects generated files into per-run
folders. The corrected split implementation uses its own chronological
partition constants, so editing the descriptive validation values in
`config.yaml` alone does not change train/validation/test boundaries. Keep the
documented source filenames and required schemas unchanged unless updating
ingestion, tests, and handover documentation together.

Place these files locally in `data/raw/`:

- `traffic_sensor_log.csv`
- `weather_observations.csv`
- `calendar_events.csv`

Raw files are read-only inputs. The `.gitignore` excludes all files under
`data/raw/`, generated intermediate/processed data, model binaries, secrets,
and row-level quarantine data. Historical experiment summaries and corrected
versioned model cards/metrics remain available under `reports/`. Before
publishing, verify the Git staging area does not contain private inputs or
controlled source documents.

## Reproduce the corrected pipeline

From the project root with the Python 3.11 environment active:

```bash
python -m src.run_pipeline
```

This runs M1, M2, M3, M4, M5, M6 sequence generation, M6 training, and the
matched M6 comparison. It starts from the three local raw files, took about
46 minutes on the verified 16-GB Mac, and writes a new timestamped run rather
than overwriting previous results. Runtime depends on hardware. The release
evidence for the completed run is in `reports/corrected_v2/20261102T172452/`.

The run manifest records input hashes and stage status. If a stage fails,
inspect its manifest/checkpoint and error first. After correcting the cause,
resume that run using its exact ID and failed stage:

```bash
python -m src.run_pipeline --run-id <run-id> --start-at <stage>
```

Supported stages are `m1`, `m2`, `m3`, `m4`, `m5`, `m6_sequences`,
`m6_train`, and `m6_compare`. Resume requires unchanged raw-input hashes.
Use a new run ID for a fresh reproduction; an existing completed run is not
overwritten.

Generate tree-based feature-importance reports for the latest completed
corrected run:

```bash
python -m src.feature_importance
```

The CSV and Markdown outputs are written into that run's versioned report
directory. To regenerate for a specific completed corrected run:

```bash
python -m src.feature_importance --run-id 20261102T172452
```

Feature importance is a model-specific association, not a causal explanation.
The tool uses the selected saved model and its training-fitted preprocessor;
it does not fit models or use the test partition.

## Tests

The suite includes unit/integration checks and an AppTest route check for all
11 dashboard pages. Run it after the corrected pipeline has completed:

```bash
python -m pytest -q
```

In a fresh clone without a completed run, the dashboard route test is skipped;
that skip is not evidence that the dashboard works. The release review
verified all 11 routes against the persisted corrected run. Final handover
validation completed with **26 passed in 144.26 seconds** on Python 3.11.18.

## Start the dashboard

After a successful corrected pipeline run:

```bash
streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit. The dashboard reads only the latest
successfully completed corrected run and does not retrain selected production
models. The Data Validation view accepts `traffic_sensor_log.csv`, with
optional weather/calendar files. If either is omitted, the current project
raw source is copied into an isolated upload run as a clearly recorded
fallback. Upload processing runs M1–M3 and uses saved M5 models to predict
each road's latest feature-complete record. The result includes its `as_of`
time and model-run lineage. Upload inference does not change raw inputs or
retrain models.

Regression forecasts include 90% split-conformal residual intervals
calibrated on a later validation block. Their nominal coverage is not
guaranteed for temporally dependent or shifted data; some deep-model held-out
coverage was below 90%. Classification scores are not independently
calibrated.

## Troubleshooting

| Symptom | Check / action |
|---|---|
| `Required immutable raw inputs are missing` | Obtain the three approved source CSVs and place them in `data/raw/`; do not commit them. |
| Dashboard says corrected run not found | Activate the Python 3.11 environment and complete `python -m src.run_pipeline`; check `reports/corrected_v2/latest.json`. |
| Pipeline reports an existing run ID | Use a new ID for a fresh run, or resume the same failed run with its manifest's failed stage. Never delete or overwrite a historical run to clear this error. |
| Upload skips a road | Supply aligned weather/calendar data or confirm project fallbacks cover the upload; inspect the reported `as_of` and feature-completeness details. |
| Upload parser/schema errors | Use the documented CSV filenames and required columns; read the Data Validation issue/quarantine reports. |
| Out-of-memory or slow training | Close other memory-heavy applications and rerun on a machine with sufficient RAM; retain the run ID and manifest if a stage must be resumed. |
| Import or binary dependency errors | Confirm Python 3.11 is active, then run `python -m pip install -r requirements.txt` and `python -m pip check`. |

## Handover and responsible use

Review the [final project audit](reports/final_project_audit.md) before
interpreting acceptance. The remaining congestion, accident-risk, and
deep-versus-classical performance requirements are **unmet**. Do not describe
the current model as PRD-complete, operationally validated, or ready to
control traffic. Preserve each versioned run and its historical results.

The handover initialized a local `main` branch but did not create a commit or
configure a remote. Review the staging area and repository visibility before
publishing.
