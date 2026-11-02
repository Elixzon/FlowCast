# FlowCast M5 — Accident Risk Threshold Analysis

## Method
- Model: Linear SVM
- C: 0.01
- Features: 38 baseline features
- Validation: Three chronological expanding-window folds
- Threshold selection: Top-scoring 0.5%, 1%, 2%, and 5% alerts per fold
- Test partition: Not used

## Average Results

| Alert Rate | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.5% | 0.0500 | 0.0265 | 0.0346 |
| 1% | 0.0468 | 0.0499 | 0.0482 |
| 2% | 0.0343 | 0.0727 | 0.0465 |
| 5% | 0.0251 | 0.1330 | 0.0422 |

## Findings
- The 1% alert rate achieved the highest average F1.
- The 5% alert rate achieved the highest recall but generated many false positives.
- Accident-risk discrimination remains weak.
- Threshold adjustment changes the precision-recall trade-off but does not improve ROC-AUC.
- The PRD ROC-AUC target of 0.75 has not been achieved.

## Decision
Retain the baseline Linear SVM with C=0.01 as an experimental reference.
Do not claim accident-risk prediction meets the PRD target.
Do not use the test partition for further threshold selection.

## Output
reports/m5_accident_threshold_cv_30m.csv
