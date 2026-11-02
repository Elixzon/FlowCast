# FlowCast Leakage Audit

## Scope and disposition

Reviewed M2 cleaning, M3 feature generation, M5 target generation/splitting/
preprocessing, M6 sequence generation/training, and the saved test comparisons.
The findings below distinguish code inspection from rerun evidence. No previous
experiment artifacts have been overwritten. Since the feature pipeline changed,
old model metrics are retained as historical outputs but are not certified as
results from the corrected pipeline.

## Findings

| ID | Severity | Finding and impact | Repair/status |
|---|---|---|---|
| L-01 | HIGH | The previous M2 `fill_segment_gaps` linearly interpolated traffic volume, speed, and occupancy using both earlier and later readings, then fell back to medians calculated over the complete road/time series. That exposed future observations to earlier feature rows and could synthesize traffic-volume target values. | Fixed in `src/clean.py`: target measurements are no longer imputed; short fills for speed/occupancy use only earlier values from the same segment; unresolved rows remain present. Pipeline outputs and all dependent experiments still need regeneration. |
| L-02 | HIGH | The previous M2 cleaner replaced missing `travel_time` and `accident_count` values with full-segment medians. These fields are targets or target-label inputs; imputation fabricates outcomes and can change accident-risk labels. | Fixed in `src/clean.py`: target columns are no longer median-imputed. New outputs and labels have not been generated. |
| L-03 | HIGH | The previous M3 `groupby(...).shift()` implementation treated the prior available row as `t-1` even if an entire 30-minute record was absent. The Data Dictionary explicitly says sensor windows can be absent, so requested `t-1`, `t-2`, `t-48`, and rolling windows could refer to the wrong times. | Fixed in `src/features.py`: lag and rolling values are looked up at exact segment timestamps offset by 30-minute multiples; incomplete windows remain null. Tests and regenerated features are pending. |
| L-04 | MEDIUM | The previous M6 code used `pd.get_dummies` on all partitions before training, allowing validation/test category vocabularies to affect the encoded feature space. | Fixed in `src/m6_prepare_sequences.py`: `OneHotEncoder` is fitted on training rows only, ignores unknown categories, and is persisted with the training-fitted scaler. Existing sequence arrays and deep-learning models are stale until regenerated/retrained. |
| L-05 | NOT VERIFIED | M3 event proximity reads event dates from the full supplied calendar and marks the preceding and following day. If those events were not known to operators at forecast time, this feature uses future information. The files have no event-announcement/publication timestamps, so the as-of availability assumption cannot be tested. | Not resolved by inference. Restrict event proximity to events documented as scheduled/known by the prediction timestamp, or add availability metadata and test it. Until then, exclude this feature from operational claims. |
| L-06 | MEDIUM | M2 previously dropped unrecoverable rows after cleaning. Removing an absent observation before row-based lag calculation can compress time and further misalign lags. | Fixed alongside L-03: unresolved timestamp rows are retained; M3 uses exact timestamp joins. Validate row counts and split coverage after rerunning. |

## Controls observed

- M5 target generation aligns future outcomes by road ID and exact target
  timestamp and checks one-to-one keys.
- M5 chronological splits constrain each future target timestamp to the current
  partition end.
- M5 feature preprocessing fits its scaler/encoder on training rows only and
  preserves feature warm-up nulls by excluding incomplete samples rather than
  filling them with zero.
- M5 feature sets exclude current traffic measurements, target columns, and
  target-derived capacity ratio/headroom.
- M6 constructs each sequence inside one road and one split, checks 30-minute
  continuity, and stores target road/timestamp metadata.
- The existing M6 comparison matched XGBoost, LSTM, and GRU on the same
  9,573 records. That matching evidence does not remove the need to rerun after
  the M2/M3/M6 corrections.

## Required reruns before certifying metrics

1. Run M1 and M2 against the immutable raw files and inspect validation,
   quarantine, unresolved-null, and unmatched-merge counts.
2. Regenerate M3 features and verify exact-time lag/rolling assertions.
3. Regenerate future targets, chronological partitions, and train-only
   preprocessors for all four horizons.
4. Retrain/reselect M5 models using validation only, then evaluate once on the
   untouched test partition.
5. Regenerate M6 sequences with train-only encoders, retrain LSTM/GRU, and
   recreate the matched test comparison on identical records.
6. Resolve L-05 with documented event availability or remove that feature from
   operational forecasts, then repeat impacted experiments.

Until these reruns finish, the saved M5/M6 scores are **legacy results**, not
validated performance claims for the corrected pipeline.

## Validation completed during this audit

- M1 was run against all three raw files; its updated report and row-level
  quarantine CSVs are persisted under `reports/`. Previous M1 report/register
  files were preserved as `*_pre_audit` copies.
- Revised M2/M3 code was run over the full raw dataset using disposable
  temporary outputs: 176,701 deduplicated traffic rows, 1,767 duplicate rows
  quarantined, 4,587 missing traffic-volume targets left unfilled, zero
  unmatched weather/calendar keys, and 176,701 M3 rows with 59 columns.
- Corrected targets, four chronological horizon splits, and all 16
  train-fitted M5 preprocessing combinations were executed against those
  temporary M3 outputs. These output files were removed after validation and
  did not replace legacy processed data or models.
- Eight focused pytest tests pass. Streamlit AppTest exercised all dashboard
  routes without exceptions.

No corrected M5 estimator was trained or evaluated on the held-out test
partition during this audit. No corrected M6 sequence/model experiment was run.
