# Project FORESIGHT — Week 3 Forecasting Report

## 1. Objective

The objective of Week 3 was to develop an advanced global forecasting model that performs better than the Week 2 previous-week naive baseline.

The baseline model achieved a WAPE of 47.88%.

## 2. Modelling Dataset

The weekly modelling dataset contained:

- 150 SKUs
- 104 complete weeks
- 15,600 complete SKU-week records
- 42 original weekly columns
- Leakage-safe demand, inventory, promotion and calendar features

Only information available before the forecast week was used as model input.

Current-week demand, revenue, stockout and sales information were excluded from the feature set to prevent data leakage.

## 3. Time-Based Data Split

The dataset was divided chronologically:

- Training period: 2 January 2023 to 24 June 2024
- Training weeks: 78
- Training rows: 11,700

- Validation period: 1 July 2024 to 23 September 2024
- Validation weeks: 13
- Validation rows: 1,950

- Test period: 30 September 2024 to 23 December 2024
- Test weeks: 13
- Test rows: 1,950

A chronological split was used instead of a random split to preserve the time-series structure and prevent future information from entering the training data.

## 4. Candidate Models

Two global machine-learning models were evaluated:

1. HistGradientBoostingRegressor
2. RandomForestRegressor

Both models were trained using data from all 150 SKUs.

### Validation Results

| Model | Validation WAPE |
|---|---:|
| HistGradientBoosting | 46.87% |
| Random Forest | 50.02% |

HistGradientBoosting achieved the lowest validation WAPE and was selected as the final model.

## 5. Final Test Results

| Model | WAPE | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 39.51% | 57.01 | 96.10 | +2.94% |
| Previous-week naive | 47.88% | 69.08 | 121.54 | -1.05% |
| Seasonal naive | 53.30% | 76.89 | 143.16 | +0.23% |

The HistGradientBoosting model improved WAPE by 8.37 percentage points compared with the previous-week naive baseline.

This represents approximately a 17.5% relative reduction in WAPE.

The final model also achieved lower MAE and RMSE than both baseline models.

## 6. Feature Importance

Permutation importance was used to understand which variables contributed most strongly to the model.

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

The 4-, 8- and 13-week rolling demand averages were the most important features.

This indicates that recent and medium-term demand patterns were more useful than relying only on the previous week or the same week in the previous year.

Incoming inventory and previous on-hand inventory also contributed useful information.

## 7. SKU-Level Error Analysis

The SKUs with the highest total absolute forecast error were:

| Rank | SKU | Actual | Forecast | WAPE | Bias |
|---:|---|---:|---:|---:|---:|
| 1 | SKU10100 | 36,705.00 | 39,665.27 | 14.30% | +8.07% |
| 2 | SKU10045 | 17,103.00 | 18,478.69 | 22.38% | +8.04% |
| 3 | SKU10144 | 20,460.00 | 19,820.99 | 15.74% | -3.12% |
| 4 | SKU10114 | 9,358.00 | 11,074.79 | 21.98% | +18.35% |
| 5 | SKU10123 | 6,754.00 | 6,401.66 | 27.98% | -5.22% |
| 6 | SKU10082 | 3,506.00 | 3,988.52 | 52.59% | +13.76% |
| 7 | SKU10105 | 11,801.00 | 10,868.80 | 15.53% | -7.90% |
| 8 | SKU10143 | 5,928.00 | 5,455.58 | 30.62% | -7.97% |
| 9 | SKU10147 | 5,027.00 | 4,818.45 | 35.65% | -4.15% |
| 10 | SKU10019 | 4,030.00 | 3,929.28 | 44.39% | -2.50% |

High-volume SKUs can appear near the top because even a moderate percentage error produces a large absolute error.

SKU10082 and SKU10019 require particular attention because their WAPE values remain high.

## 8. Worst Forecast Week

The worst-performing forecast week started on 9 December 2024.

- Actual demand: 18,996.00 units
- Forecast demand: 24,024.64 units
- WAPE: 50.03%
- Bias: +26.47%

The model substantially over-forecasted during this week.

Possible causes include unusual seasonal behaviour, promotion effects, holiday demand changes or demand patterns not fully represented in the historical data.

## 9. Model Limitations

- Only approximately two years of weekly history were available.
- Some SKUs have intermittent and highly variable demand.
- The model does not yet produce forecast intervals.
- Promotion and holiday information may not fully describe special-event demand.
- Accuracy varies considerably between SKUs.
- High-error products should be reviewed before automatic inventory decisions are made.

## 10. Final Conclusion

The HistGradientBoosting model successfully outperformed both naive forecasting baselines.

The final WAPE decreased from 47.88% to 39.51%, demonstrating that the leakage-safe rolling demand, lag, calendar, promotion and inventory features improved forecast accuracy.

The trained model and supporting analysis are suitable for use in Week 4, where forecasts will be converted into stockout-risk scores, overstock-risk scores, reorder recommendations and dashboard outputs.