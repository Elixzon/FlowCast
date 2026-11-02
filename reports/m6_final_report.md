# M6 — Deep Learning Models and Final Evaluation

## 1. Objective

The objective of M6 was to develop and evaluate deep learning models for
30-minute-ahead traffic volume prediction using historical traffic,
weather, road and calendar information.

The models implemented were:
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

Their performance was compared against the selected classical machine
learning baseline, XGBoost.

## 2. Dataset and sequence preparation

The M6 sequence preparation used the M5 chronological train, validation
and test partitions.

| Parameter | Configuration |
|---|---|
| Prediction target | traffic_volume_t_plus_30 |
| Prediction horizon | 30 minutes |
| Sequence length | 48 time steps |
| Historical input duration | 24 hours |
| Road segments | 25 |
| Features per time step | 67 |
| Scaling | StandardScaler |
| Scaler fitting | Training partition only |

Each sequence contains 48 consecutive half-hour observations for one
road segment. Sequences do not cross road-segment boundaries or
train/validation/test boundaries. Windows with timestamp gaps or
incomplete input values are excluded.

### Sequence counts

| Partition | Sequences | Input shape |
|---|---:|---|
| Training | 37,459 | 37,459 × 48 × 67 |
| Validation | 4,814 | 4,814 × 48 × 67 |
| Test | 9,573 | 9,573 × 48 × 67 |

No non-finite input values or target values were found in the saved
sequence arrays.

### Feature groups

The 67 input features consist of:

- Road information: latitude, longitude and road capacity.
- Weather information: temperature, rainfall, visibility and weather
  condition.
- Calendar information: public holiday, event and roadwork flags.
- Time information: hour, day of week, cyclic time features, weekend
  and peak-period flags.
- Engineered indicators: rain, low visibility, holiday-peak interaction
  and event proximity.
- Historical traffic volume: lag 1, lag 2, lag 48 and rolling
  statistics over 4 and 8 observations.
- Historical speed: lag 1, lag 2, lag 48 and rolling statistics
  over 4 and 8 observations.
- Categorical encodings: road ID, weather condition and temperature band.

## 3. Model architectures

Both recurrent models use sequences of shape 48 × 67.

### 3.1 LSTM

The implemented LSTM architecture:

1. LSTM layer with 64 units.
2. Dropout layer with rate 0.2.
3. Dense layer with 32 units and ReLU activation.
4. Dense output layer with 1 unit for traffic-volume regression.

### 3.2 GRU

The implemented GRU architecture:

1. GRU layer with 64 units.
2. Dropout layer with rate 0.2.
3. Dense layer with 32 units and ReLU activation.
4. Dense output layer with 1 unit for traffic-volume regression.

### Training configuration

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Mean Squared Error |
| Batch size | 128 |
| Maximum epochs | 40 |
| Early stopping | Enabled |
| Early-stopping patience | 8 |
| Best model selection | Validation loss |

The models were trained using the training partition and monitored
using the validation partition. The test partition was not used
during model training or validation-based model selection.

## 4. Validation results

### LSTM validation

| Metric | Result |
|---|---:|
| Epochs completed | 24 |
| Best epoch | 16 |
| MAE | 37.9756 |
| RMSE | 54.9877 |
| MAPE | 9.2687% |
| R² | 0.9606 |

### GRU validation

| Metric | Result |
|---|---:|
| Epochs completed | 21 |
| Best epoch | 13 |
| MAE | 37.1509 |
| RMSE | 53.9670 |
| MAPE | 9.5246% |
| R² | 0.9621 |

GRU achieved a lower validation RMSE than LSTM. LSTM achieved a
lower validation MAPE.

## 5. Final test evaluation

The final comparison used exactly the same 9,573 test samples for
XGBoost, LSTM and GRU.

The XGBoost preprocessing pipeline was reconstructed and fitted using
training rows only. Sequence timestamps and road IDs were matched
against the original test partition, and the sequence targets were
verified against the original target values.

### Final comparison

| Model | Test samples | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|---:|
| LSTM | 9,573 | 39.7690 | 58.7231 | 9.4401% | 0.95836 |
| XGBoost | 9,573 | 40.2339 | 59.0495 | 9.6024% | 0.95790 |
| GRU | 9,573 | 40.2883 | 59.2894 | 10.1146% | 0.95755 |

### Test observations

- LSTM achieved the lowest MAE and RMSE.
- LSTM achieved the lowest MAPE and highest R².
- XGBoost remained a strong baseline, with results close to LSTM.
- GRU performed slightly below XGBoost on the final test set.
- All three models achieved test MAPE below 12%.

LSTM reduced RMSE by approximately 0.55% relative to XGBoost on
this matched test subset. This is a modest improvement and does not
by itself establish statistical significance.

## 6. Final model selection

LSTM is selected as the final model for 30-minute-ahead traffic-volume
prediction because it achieved the best overall test metrics.

XGBoost is retained as the classical machine learning baseline.
GRU is retained as an evaluated recurrent alternative.

The model selection is based on the final matched test comparison,
while the training process and early stopping used the validation
partition.

## 7. Limitations and considerations

1. Deep learning test comparison was performed for the 30-minute
   prediction horizon. These results should not be generalized to
   60-, 90- or 120-minute horizons.
2. LSTM's improvement over XGBoost is small. Further testing across
   additional time periods would help establish robustness.
3. Statistical significance testing was not performed.
4. A large number of candidate sequence windows were rejected because
   of timestamp gaps or incomplete historical input windows.
5. The sequence preparation generated categorical dummy columns from
   the available partitions. For stricter future deployment validation,
   categorical vocabularies should be frozen using training data only.
6. The reported results evaluate traffic-volume regression and do not
   establish that the deep learning models improve congestion or
   accident-risk classification.

## 8. Conclusion

M6 successfully implemented LSTM and GRU models for 30-minute-ahead
traffic-volume prediction using 48-step historical sequences.

Both models were trained and evaluated using chronological data
partitions. The final test comparison used 9,573 identical samples
for all three models.

LSTM achieved an MAE of 39.7690, RMSE of 58.7231, MAPE of 9.4401%
and R² of 0.95836. It narrowly outperformed the XGBoost baseline
and the GRU model.

Therefore, LSTM is selected as the final M6 model for the
30-minute traffic-volume prediction task.

## 9. Generated artifacts

- `models/m6/lstm_30m.keras`
- `models/m6/gru_30m.keras`
- `reports/m6_sequence_report.json`
- `reports/m6_lstm_30m.json`
- `reports/m6_gru_30m.json`
- `reports/m6_final_test_comparison.csv`
- `reports/m6_final_test_comparison.json`
- `reports/m6_final_test_comparison.md`
