# FlowCast M5 — Accident Risk Results Across All Horizons

## Model comparison

### 30-minute horizon

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.5838 | 0.0131 | 0.0120 | 0.4733 | 0.0233 |
| Random Forest | 0.5644 | 0.0148 | 0.0439 | 0.0333 | 0.0379 |
| XGBoost | 0.5525 | 0.0127 | 0.0148 | 0.1800 | 0.0273 |
| Decision Tree | 0.5483 | 0.0113 | 0.0131 | 0.4133 | 0.0254 |

ROC-AUC-selected model: **Linear SVM** (0.5838)

### 60-minute horizon

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.5735 | 0.0122 | 0.0112 | 0.5302 | 0.0220 |
| Random Forest | 0.5661 | 0.0137 | 0.0307 | 0.0336 | 0.0321 |
| XGBoost | 0.5500 | 0.0149 | 0.0151 | 0.1745 | 0.0277 |
| Decision Tree | 0.5162 | 0.0097 | 0.0104 | 0.3624 | 0.0203 |

ROC-AUC-selected model: **Linear SVM** (0.5735)

### 90-minute horizon

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.5623 | 0.0107 | 0.0105 | 0.5068 | 0.0206 |
| Random Forest | 0.5564 | 0.0137 | 0.0392 | 0.0135 | 0.0201 |
| Decision Tree | 0.5466 | 0.0102 | 0.0098 | 0.3784 | 0.0190 |
| XGBoost | 0.5456 | 0.0126 | 0.0174 | 0.2095 | 0.0321 |

ROC-AUC-selected model: **Linear SVM** (0.5623)

### 120-minute horizon

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.5780 | 0.0124 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | 0.5627 | 0.0132 | 0.0139 | 0.1409 | 0.0253 |
| Linear SVM | 0.5601 | 0.0106 | 0.0115 | 0.5369 | 0.0226 |
| Decision Tree | 0.5234 | 0.0096 | 0.0104 | 0.3289 | 0.0201 |

ROC-AUC-selected model: **Random Forest** (0.5780)

## PRD evaluation

Required accident-risk ROC-AUC: >= 0.75.

| Horizon | Selected model | ROC-AUC | Status |
|---|---|---:|---|
| 30 minutes | Linear SVM | 0.5838 | Not achieved |
| 60 minutes | Linear SVM | 0.5735 | Not achieved |
| 90 minutes | Linear SVM | 0.5623 | Not achieved |
| 120 minutes | Random Forest | 0.5780 | Not achieved |

## Findings and limitations

- Accident-risk prediction remains below the PRD ROC-AUC target.
- Positive accident cases are rare, so accuracy alone is insufficient.
- Precision, recall, F1, ROC-AUC and PR-AUC must be considered together.
- A high ROC-AUC does not guarantee useful predictions at the default threshold.
- The 120-minute Random Forest has zero recall at its default threshold.
- Threshold tuning does not improve ROC-AUC.
- These results are validation results; the test partition has not been used.

## Source reports

- reports/m5_accident_risk_30m.json
- reports/m5_accident_risk_60m.json
- reports/m5_accident_risk_90m.json
- reports/m5_accident_risk_120m.json
