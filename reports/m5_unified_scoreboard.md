# FlowCast M5 — Unified Model Scoreboard

Validation results only. The untouched test partition was not used.

| Task | Horizon | Selected model | Selection criterion | PRD status |
|---|---:|---|---|---|
| Traffic volume | 30 min | XGBoost | Lowest validation RMSE | PASS |
| Travel time | 30 min | XGBoost | Lowest validation RMSE | NO PRD THRESHOLD |
| Congestion | 30 min | XGBoost | Highest validation macro-F1 | NOT MET |
| Accident risk | 30 min | Linear SVM | Highest validation ROC-AUC | NOT MET |
| Traffic volume | 60 min | XGBoost | Lowest validation RMSE | NOT MET |
| Travel time | 60 min | Random Forest | Lowest validation RMSE | NO PRD THRESHOLD |
| Congestion | 60 min | XGBoost | Highest validation macro-F1 | NOT MET |
| Accident risk | 60 min | Linear SVM | Highest validation ROC-AUC | NOT MET |
| Traffic volume | 90 min | XGBoost | Lowest validation RMSE | NOT MET |
| Travel time | 90 min | XGBoost | Lowest validation RMSE | NO PRD THRESHOLD |
| Congestion | 90 min | XGBoost | Highest validation macro-F1 | NOT MET |
| Accident risk | 90 min | Linear SVM | Highest validation ROC-AUC | NOT MET |
| Traffic volume | 120 min | XGBoost | Lowest validation RMSE | NOT MET |
| Travel time | 120 min | XGBoost | Lowest validation RMSE | NO PRD THRESHOLD |
| Congestion | 120 min | XGBoost | Highest validation macro-F1 | NOT MET |
| Accident risk | 120 min | Random Forest | Highest validation ROC-AUC | NOT MET |

## Detailed metrics

| Task | Horizon | MAE | RMSE | MAPE % | Macro-F1 | ROC-AUC | PRD status |
|---|---:|---:|---:|---:|---:|---:|---|
| Traffic volume | 30 | 43.9959 | 65.7338 | 11.1032 | — | — | PASS |
| Travel time | 30 | 0.4744 | 1.1427 | 10.7469 | — | — | NO PRD THRESHOLD |
| Congestion | 30 | — | — | — | 0.7379 | — | NOT MET |
| Accident risk | 30 | — | — | — | — | 0.5838 | NOT MET |
| Traffic volume | 60 | 47.2704 | 73.1304 | 12.0620 | — | — | NOT MET |
| Travel time | 60 | 0.4814 | 1.1909 | 10.8988 | — | — | NO PRD THRESHOLD |
| Congestion | 60 | — | — | — | 0.7163 | — | NOT MET |
| Accident risk | 60 | — | — | — | — | 0.5735 | NOT MET |
| Traffic volume | 90 | 49.9744 | 79.1501 | 13.1817 | — | — | NOT MET |
| Travel time | 90 | 0.5146 | 1.2289 | 12.3450 | — | — | NO PRD THRESHOLD |
| Congestion | 90 | — | — | — | 0.7305 | — | NOT MET |
| Accident risk | 90 | — | — | — | — | 0.5623 | NOT MET |
| Traffic volume | 120 | 51.6613 | 80.9760 | 14.0889 | — | — | NOT MET |
| Travel time | 120 | 0.5236 | 1.2280 | 12.5667 | — | — | NO PRD THRESHOLD |
| Congestion | 120 | — | — | — | 0.7043 | — | NOT MET |
| Accident risk | 120 | — | — | — | — | 0.5780 | NOT MET |

## PRD acceptance summary

- Traffic volume: MAPE <= 12%.
- Congestion: macro-F1 >= 0.80.
- Accident risk: ROC-AUC >= 0.75.
- Travel time: no separate numeric acceptance threshold specified in the PRD.

## Important limitations

- These are validation metrics, not final test metrics.
- Model winners are selected using their recorded validation criteria.
- A selected model is not necessarily operationally acceptable.
- PRD failures are recorded honestly; no thresholds have been relaxed.

## Source

Generated from the saved M5 experiment JSON reports.
