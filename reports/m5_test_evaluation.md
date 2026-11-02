# M5 Final Test Evaluation

Evaluation uses the untouched chronological test partition.
Models and configurations were selected using validation results.
No retraining or threshold tuning was performed on test data.

## Traffic Volume — 30 minutes

- Model: `xgboost_regression_30m.joblib`
- Test rows: 34416
- MAE: 40.595044
- RMSE: 59.306570
- MAPE_percent: 9.707989
- R2: 0.956713

## Traffic Volume — 60 minutes

- Model: `xgboost_regression_60m.joblib`
- Test rows: 34389
- MAE: 41.634393
- RMSE: 60.427322
- MAPE_percent: 9.977747
- R2: 0.955080

## Traffic Volume — 90 minutes

- Model: `xgboost_regression_90m.joblib`
- Test rows: 34367
- MAE: 42.362604
- RMSE: 61.487853
- MAPE_percent: 10.220590
- R2: 0.953545

## Traffic Volume — 120 minutes

- Model: `xgboost_regression_120m.joblib`
- Test rows: 34350
- MAE: 43.111512
- RMSE: 62.613886
- MAPE_percent: 10.458606
- R2: 0.951867

## Travel Time — 30 minutes

- Model: `xgboost_travel_time_30m.joblib`
- Test rows: 34416
- MAE: 0.438887
- RMSE: 1.059383
- MAPE_percent: 9.455154
- R2: 0.824447

## Travel Time — 60 minutes

- Model: `random_forest_travel_time_60m.joblib`
- Test rows: 34389
- MAE: 0.430852
- RMSE: 1.074421
- MAPE_percent: 8.992344
- R2: 0.820229

## Travel Time — 90 minutes

- Model: `xgboost_travel_time_90m.joblib`
- Test rows: 34367
- MAE: 0.448584
- RMSE: 1.069327
- MAPE_percent: 9.778008
- R2: 0.821772

## Travel Time — 120 minutes

- Model: `xgboost_travel_time_120m.joblib`
- Test rows: 34350
- MAE: 0.484638
- RMSE: 1.114788
- MAPE_percent: 10.931748
- R2: 0.805885

## Congestion Level — 30 minutes

- Model: `xgboost_congestion_30m.joblib`
- Test rows: 34416
- accuracy: 0.873547
- macro_precision: 0.770208
- macro_recall: 0.761545
- macro_f1: 0.763977
- confusion_matrix: `[[19773, 0, 996, 0], [0, 2308, 604, 471], [738, 722, 6825, 15], [0, 795, 11, 1158]]`
- class_labels: `["Free-flow", "Heavy", "Moderate", "Severe"]`

## Congestion Level — 60 minutes

- Model: `xgboost_congestion_60m.joblib`
- Test rows: 34389
- accuracy: 0.863939
- macro_precision: 0.755779
- macro_recall: 0.736883
- macro_f1: 0.744976
- confusion_matrix: `[[19737, 1, 1036, 0], [0, 1995, 914, 492], [859, 537, 6836, 22], [0, 798, 20, 1142]]`
- class_labels: `["Free-flow", "Heavy", "Moderate", "Severe"]`

## Congestion Level — 90 minutes

- Model: `xgboost_congestion_90m.joblib`
- Test rows: 34367
- accuracy: 0.868828
- macro_precision: 0.763714
- macro_recall: 0.756633
- macro_f1: 0.757525
- confusion_matrix: `[[19718, 0, 1043, 0], [0, 2375, 546, 473], [777, 809, 6644, 19], [0, 830, 11, 1122]]`
- class_labels: `["Free-flow", "Heavy", "Moderate", "Severe"]`

## Congestion Level — 120 minutes

- Model: `xgboost_congestion_120m.joblib`
- Test rows: 34350
- accuracy: 0.860757
- macro_precision: 0.748607
- macro_recall: 0.729719
- macro_f1: 0.737658
- confusion_matrix: `[[19670, 1, 1079, 0], [0, 1796, 1054, 534], [832, 472, 6914, 24], [0, 761, 26, 1187]]`
- class_labels: `["Free-flow", "Heavy", "Moderate", "Severe"]`

## Accident Risk — 30 minutes

- Model: `linear_svm_accident_risk_30m.joblib`
- Test rows: 34416
- accuracy: 0.697670
- macro_precision: 0.502947
- macro_recall: 0.565281
- macro_f1: 0.423807
- confusion_matrix: `[[23869, 10217], [188, 142]]`
- class_labels: `["0", "1"]`
- positive_precision: 0.013708
- positive_recall: 0.430303
- positive_f1: 0.026569
- ROC_AUC: 0.587900
- PR_AUC: 0.017019

## Accident Risk — 60 minutes

- Model: `linear_svm_accident_risk_60m.joblib`
- Test rows: 34389
- accuracy: 0.610050
- macro_precision: 0.503312
- macro_recall: 0.581277
- macro_f1: 0.391578
- confusion_matrix: `[[20793, 13259], [151, 186]]`
- class_labels: `["0", "1"]`
- positive_precision: 0.013834
- positive_recall: 0.551929
- positive_f1: 0.026992
- ROC_AUC: 0.609474
- PR_AUC: 0.017956

## Accident Risk — 90 minutes

- Model: `linear_svm_accident_risk_90m.joblib`
- Test rows: 34367
- accuracy: 0.572235
- macro_precision: 0.502689
- macro_recall: 0.568247
- macro_f1: 0.375538
- confusion_matrix: `[[19477, 14555], [146, 189]]`
- class_labels: `["0", "1"]`
- positive_precision: 0.012819
- positive_recall: 0.564179
- positive_f1: 0.025068
- ROC_AUC: 0.588229
- PR_AUC: 0.015368

## Accident Risk — 120 minutes

- Model: `random_forest_accident_risk_120m.joblib`
- Test rows: 34350
- accuracy: 0.990160
- macro_precision: 0.495195
- macro_recall: 0.499882
- macro_f1: 0.497528
- confusion_matrix: `[[34012, 8], [330, 0]]`
- class_labels: `["0", "1"]`
- positive_precision: 0.000000
- positive_recall: 0.000000
- positive_f1: 0.000000
- ROC_AUC: 0.555689
- PR_AUC: 0.012944
