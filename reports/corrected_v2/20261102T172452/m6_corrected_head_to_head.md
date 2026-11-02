# Corrected M6 / M5 Matched Comparison

Classical and deep models are selected on the same matched chronological
validation-selection records, calibrated on a later validation block, and
evaluated once on exactly the same held-out test records. Issue-time
features are identical; M6 additionally uses the prior 47 contiguous
half-hour feature vectors. All preprocessing is fitted on training data.

| Horizon | Model | Validation RMSE | Test rows | Test RMSE | Test MAPE (%) | 90% interval coverage | Δ RMSE vs classical |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | Classical XGBoost | 63.2756 | 2673 | 59.7493 | 9.8337 | 0.9300 | +0.0000 |
| 30 | LSTM | 72.0148 | 2673 | 70.9926 | 15.0891 | 0.9095 | +11.2433 |
| 30 | GRU | 66.0937 | 2673 | 61.3525 | 10.9788 | 0.9226 | +1.6032 |
| 30 | Bidirectional LSTM | 62.8159 | 2673 | 60.7992 | 10.6420 | 0.9379 | +1.0499 |
| 60 | Classical XGBoost | 65.2537 | 2602 | 58.2218 | 10.0379 | 0.9500 | +0.0000 |
| 60 | LSTM | 65.5077 | 2602 | 65.3720 | 12.7806 | 0.8909 | +7.1501 |
| 60 | GRU | 68.8411 | 2602 | 64.7338 | 11.6031 | 0.9101 | +6.5119 |
| 60 | Bidirectional LSTM | 64.5577 | 2602 | 62.6850 | 11.7571 | 0.9385 | +4.4632 |
| 90 | Classical Random Forest | 63.7372 | 2522 | 61.0058 | 10.1995 | 0.9766 | +0.0000 |
| 90 | LSTM | 67.5762 | 2522 | 70.0585 | 13.5966 | 0.8894 | +9.0527 |
| 90 | GRU | 70.2899 | 2522 | 67.3656 | 12.5810 | 0.9227 | +6.3598 |
| 90 | Bidirectional LSTM | 68.4384 | 2522 | 68.7919 | 13.4179 | 0.9314 | +7.7861 |
| 120 | Classical XGBoost | 67.5905 | 2447 | 61.7110 | 10.5252 | 0.9787 | +0.0000 |
| 120 | LSTM | 69.1613 | 2447 | 76.9298 | 15.0473 | 0.8672 | +15.2188 |
| 120 | GRU | 72.7361 | 2447 | 72.9168 | 14.2078 | 0.9109 | +11.2059 |
| 120 | Bidirectional LSTM | 67.6336 | 2447 | 66.7733 | 12.8913 | 0.9424 | +5.0623 |