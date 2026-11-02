# M6 Final Test Comparison

- Horizon: 30 minutes
- Identical test samples: 9573
- Models: XGBoost, LSTM, GRU
- Preprocessing fitted on training data only
- Test data used only for final evaluation

| Model | Samples | MAE | RMSE | MAPE (%) | R² |
|---|---:|---:|---:|---:|---:|
| LSTM | 9573 | 39.7690 | 58.7231 | 9.4401 | 0.9584 |
| XGBoost | 9573 | 40.2339 | 59.0495 | 9.6024 | 0.9579 |
| GRU | 9573 | 40.2883 | 59.2894 | 10.1146 | 0.9576 |

Best model by RMSE: **LSTM**
Best model by MAE: **LSTM**

All models were evaluated on the same matched test samples.