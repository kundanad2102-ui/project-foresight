# Project FORESIGHT — Model Error Analysis

## Final Model

- Selected model: HistGradientBoosting
- Final test WAPE: 39.51%
- Baseline WAPE: 47.88%
- Improvement: 8.37 percentage points
- Final model bias: approximately +2.94%

## Top 10 Important Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | demand_rolling_mean_4 | 16.8388 |
| 2 | demand_rolling_mean_8 | 15.8718 |
| 3 | demand_rolling_mean_13 | 12.7560 |
| 4 | ending_on_order_units_lag_1 | 4.1982 |
| 5 | demand_rolling_std_8 | 0.9597 |
| 6 | demand_rolling_std_13 | 0.5998 |
| 7 | demand_lag_1 | 0.5761 |
| 8 | ending_on_hand_units_lag_1 | 0.4870 |
| 9 | demand_lag_52 | 0.2667 |
| 10 | demand_rolling_std_4 | 0.2573 |

Positive permutation importance means that model performance becomes worse when the feature is randomly shuffled. Therefore, larger positive values indicate more useful forecasting features.

## Top 10 SKUs by Total Absolute Error

| Rank | SKU | Actual | Forecast | WAPE | Bias |
|---:|---|---:|---:|---:|---:|
| 1 | SKU10100 | 36705.00 | 39665.27 | 14.30% | 8.07% |
| 2 | SKU10045 | 17103.00 | 18478.69 | 22.38% | 8.04% |
| 3 | SKU10144 | 20460.00 | 19820.99 | 15.74% | -3.12% |
| 4 | SKU10114 | 9358.00 | 11074.79 | 21.98% | 18.35% |
| 5 | SKU10123 | 6754.00 | 6401.66 | 27.98% | -5.22% |
| 6 | SKU10082 | 3506.00 | 3988.52 | 52.59% | 13.76% |
| 7 | SKU10105 | 11801.00 | 10868.80 | 15.53% | -7.90% |
| 8 | SKU10143 | 5928.00 | 5455.58 | 30.62% | -7.97% |
| 9 | SKU10147 | 5027.00 | 4818.45 | 35.65% | -4.15% |
| 10 | SKU10019 | 4030.00 | 3929.28 | 44.39% | -2.50% |

## Worst Forecast Week

- Week start: 2024-12-09 00:00:00
- Actual demand: 18996.00
- Forecast demand: 24024.64
- WAPE: 50.03%
- Bias: 26.47%

## Interpretation

The final model outperformed both naive baselines, but forecast accuracy differs across SKUs and weeks. High-error SKUs should be reviewed separately before inventory decisions are generated.