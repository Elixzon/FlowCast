# FlowCast M5 — Accident Risk Model Comparison

## 1. Objective
Predict whether an accident will occur in the next 30 or 60 minutes for each road segment.

## 2. Model Comparison

### 30-Minute Horizon

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.5838 | 0.0131 | 0.0120 | 0.4733 | 0.0233 |
| Random Forest | 0.5644 | 0.0148 | 0.0439 | 0.0333 | 0.0379 |
| XGBoost | 0.5525 | 0.0127 | 0.0148 | 0.1800 | 0.0273 |
| Decision Tree | 0.5483 | 0.0113 | 0.0131 | 0.4133 | 0.0254 |

### 60-Minute Horizon

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.5735 | 0.0122 | 0.0112 | 0.5302 | 0.0220 |
| Random Forest | 0.5661 | 0.0137 | 0.0307 | 0.0336 | 0.0321 |
| XGBoost | 0.5500 | 0.0149 | 0.0151 | 0.1745 | 0.0277 |
| Decision Tree | 0.5162 | 0.0097 | 0.0104 | 0.3624 | 0.0203 |

## 3. Selected Models

Selection criterion: validation ROC-AUC.

- 30-minute: Linear SVM, ROC-AUC 0.5838.
- 60-minute: Linear SVM, ROC-AUC 0.5735.

XGBoost achieved the highest PR-AUC at both horizons, so it remains a useful alternative for further investigation.

## 4. Key Findings

- Accident labels are highly imbalanced, with fewer than 1% positive cases.
- Accuracy is not a reliable standalone metric for this task.
- Linear SVM achieved the highest ROC-AUC at both horizons.
- Precision remains very low, indicating a substantial false-alert problem.
- Historical accident features did not consistently improve cross-validation results.
- Threshold tuning changed the precision-recall trade-off but did not improve ROC-AUC.
- The 30-minute and 60-minute models both remain below the PRD target.

## 5. PRD Evaluation

Required ROC-AUC: >= 0.75.

| Horizon | Best ROC-AUC | Status |
|---|---:|---|
| 30 minutes | 0.5838 | Not achieved |
| 60 minutes | 0.5735 | Not achieved |

## 6. Limitations

- Rare positive accident labels.
- Weak discrimination between positive and negative cases.
- Low precision and substantial false-positive alerts.
- Threshold selection does not improve ranking performance.

## 7. Conclusion

Linear SVM is the selected experimental reference model for both horizons based on validation ROC-AUC. However, accident-risk prediction does not satisfy the PRD acceptance criterion.

The models must be reported as experimental and below target. No test partition was used for model selection.

## 8. Source Reports

- reports/m5_accident_risk_30m.json
- reports/m5_accident_risk_60m.json
- reports/m5_accident_threshold_cv_30m.csv
- reports/m5_accident_threshold_findings.md
