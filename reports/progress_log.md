# Project FORESIGHT Progress Log

## Week 1

## Day 1 – Project Setup

### Completed

- Created the Project FORESIGHT folder structure.
- Installed Python 3.11.
- Created and activated the virtual environment.
- Installed the required Python libraries.
- Created README.md, requirements.txt and .gitignore.

### Next Tasks

- Add the raw datasets.
- Inspect the structure of each dataset.
- Identify missing values, duplicates and invalid records.

### Blockers

- None.

## Day 2 – Raw Data Profiling

### Completed

- Loaded all three raw datasets successfully.
- Reviewed dataset dimensions, columns, data types and sample records.
- Checked missing values, duplicates, invalid dates and numeric anomalies.
- Confirmed that all 150 SKUs are consistent across sales, products and inventory.
- Identified 10,750 cancellation records in the sales data.
- Identified 17,882 missing CustomerID values.

### Key Decision

Cancellation transactions will not be treated as fulfilled demand. Missing CustomerID records will be retained for SKU-level forecasting.

### Next Tasks

- Create the formal data-quality decision table.
- Define source-to-target column mappings.
- Document the grain of each output table.

### Blockers

- None.

## Day 3 – Data Quality Decisions and Mapping

### Completed

- Created the formal data-quality decision table.
- Documented the reason for retaining missing CustomerID records.
- Documented the handling of cancellation transactions.
- Defined source-to-target mappings for all required processed datasets.
- Defined the grain of each processed table.
- Validated product and inventory key uniqueness.
- Estimated the number of daily SKU demand records after aggregation.

### Next Tasks

- Build the reproducible Python data pipeline.
- Create sales_daily.csv.
- Create sku_master.csv.
- Create inventory_snapshots.csv.
- Create calendar.csv.
- Create analysis_ready.csv.

### Blockers

- No separate subcategory field exists, so category will temporarily be used as subcategory.
- No external holiday calendar has been approved, so holiday flags will initially use a documented default.

## Day 4 – Complete Data Pipeline

### Completed

- Built a reproducible Python data-processing pipeline.
- Cleaned and validated sales transactions.
- Excluded cancellation transactions from fulfilled demand.
- Created sales_daily.csv with one row per date and SKU.
- Created sku_master.csv with one row per SKU.
- Created inventory_snapshots.csv with one row per date and SKU.
- Created calendar.csv with one row per calendar date.
- Created analysis_ready.csv by joining sales, inventory, product and calendar data.
- Validated that all processed datasets contain no missing values.
- Confirmed unique keys and consistent SKU coverage.

### Output Summary

- sales_daily.csv: 67,626 rows and 8 columns
- sku_master.csv: 150 rows and 11 columns
- inventory_snapshots.csv: 109,650 rows and 10 columns
- calendar.csv: 731 rows and 12 columns
- analysis_ready.csv: 109,650 rows and 41 columns

### Next Tasks

- Perform exploratory data analysis.
- Study SKU demand patterns and seasonality.
- Prepare weekly demand data.
- Build and evaluate forecasting baselines.

### Blockers

- No approved holiday calendar is available, so is_holiday currently uses a documented default value.
- No separate subcategory field exists, so category is temporarily reused as subcategory.

## Week 2 – EDA Completed

### Completed

- Validated all Week 1 processed datasets.
- Reconciled the forecasting target.
- Confirmed that `demand_target` equals inventory units sold plus lost sales.
- Analysed demand distribution and zero-demand records.
- Analysed daily, weekly, monthly and yearly demand patterns.
- Identified top-moving and slow-moving SKUs.
- Analysed demand concentration across products.
- Analysed category-level demand and lost-sales performance.
- Identified stockout-risk SKUs.
- Analysed lost sales and stockout impact.
- Measured SKU-level intermittency and demand variability.
- Completed promotion-demand analysis.
- Compared promotional and non-promotional demand observations.
- Completed explicit 13-week dead-stock analysis.
- Completed Spearman demand-driver correlation analysis.
- Completed the EDA insight memo.

### Promotion Analysis

- Average demand during promotional observations: 61.01 units.
- Average demand during non-promotional observations: 8.22 units.
- Observed promotion-demand difference: 642.12%.
- The promotion result is treated as an observed association and not proof that promotions caused the full increase.
- Promotion information will be retained as a forecasting feature.

### Dead-Stock Analysis

Dead stock was defined as positive ending inventory with zero estimated demand during the latest 13 weeks.

- Analysis period: 1 October 2024 to 31 December 2024.
- Dead-stock SKUs: 0.
- Dead-stock units: 0.
- Capital tied up in dead stock: 0.00.

No SKU met the defined dead-stock conditions. Slow-moving and intermittent SKUs were present, but every SKU with positive ending inventory recorded some demand during the latest 13 weeks.

### Demand-Driver Analysis

Spearman correlation was used because demand is highly skewed and contains many zero observations.

The strongest observed correlations with demand were:

- Inventory position: 0.4753
- On-order units: 0.4746
- Promotion flag: 0.4095
- Reorder point: 0.4017
- Lost sales: 0.1767
- Stockout flag: 0.1746
- On-hand units: 0.1669
- Effective unit price: 0.0488

Inventory position, on-order units, promotion status and reorder point had the strongest positive associations with demand.

Current-week stockout and lost-sales variables are used only for exploratory analysis and are excluded from forecasting inputs to prevent data leakage.

### Key Decisions

- Use `demand_target` as the forecasting target.
- Use WAPE as the primary forecasting metric.
- Use MAE, RMSE and forecast bias as secondary evaluation metrics.
- Do not use MAPE as the primary metric because many records contain zero demand.
- Aggregate demand at weekly SKU level for forecasting.
- Use leakage-safe lag and rolling features.
- Include promotion and calendar information as forecasting features.
- Treat correlation and promotion findings as descriptive associations rather than causal conclusions.

## Week 2 – Baseline Forecasting Completed

### Completed

- Created weekly SKU-level demand data.
- Generated 15,900 SKU-week records for 150 SKUs.
- Identified 104 complete weeks and excluded partial weeks from evaluation.
- Created leakage-safe demand-lag features.
- Created leakage-safe rolling means, rolling standard deviations and zero-demand-rate features.
- Used the final 13 complete weeks as a chronological test period.
- Evaluated the required 52-week seasonal-naive baseline.
- Evaluated the previous-week naive model as an additional operational benchmark.
- Used WAPE, MAE, RMSE and forecast bias for evaluation.
- Saved baseline predictions and evaluation metrics.

### Baseline Results

| Model | WAPE | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Naive – Previous Week | 47.88% | 69.08 | 121.54 | -1.05% |
| Seasonal Naive – 52 Weeks | 53.30% | 76.89 | 143.16 | +0.23% |

### Baseline Decision

- The 52-week seasonal-naive model was retained as the baseline required by the Project FORESIGHT specification.
- The previous-week naive model achieved better performance and was retained as the stronger operational benchmark.
- Evaluation rows: 1,950.
- Evaluation weeks: 13.
- Evaluation SKUs: 150.
- More advanced forecasting models were required to improve on both naive approaches.

### Week 3 Goal

Develop forecasting models that achieve a lower WAPE than 47.88%.

## Week 3 — Forecast Modelling Completed

### Completed

- Prepared a leakage-safe global modelling dataset.
- Created chronological training, validation and test periods.
- Trained HistGradientBoosting and Random Forest models.
- Selected the final model using validation WAPE.
- Retrained the selected model using training and validation data.
- Evaluated the final model on the untouched 13-week test period.
- Compared the model against previous-week and seasonal-naive baselines.
- Calculated permutation feature importance.
- Completed SKU-level and weekly forecast-error analysis.
- Saved the trained model, predictions, metrics and reports.

### Final Model Result

- Selected model: HistGradientBoosting
- Final test WAPE: 39.51%
- Previous-week baseline WAPE: 47.88%
- Improvement: 8.37 percentage points
- Final MAE: 57.01
- Final RMSE: 96.10
- Final bias: +2.94%

### Rolling-Origin Cross-Validation

- Added expanding-window rolling-origin cross-validation.
- Used 4 validation folds.
- Used an 8-week forecast horizon per fold.
- Required at least 52 weeks of training history.
- Kept the final 13-week test period untouched.
- Compared HistGradientBoosting, Random Forest, previous-week naive and 52-week seasonal-naive models.
- Selected HistGradientBoosting based on rolling-CV WAPE.
- HistGradientBoosting rolling-CV WAPE: 44.58%.
- Previous-week naive rolling-CV WAPE: 51.16%.
- Seasonal-naive rolling-CV WAPE: 74.45%.
- Improvement over previous-week naive: 6.58 percentage points.
- Saved fold-level metrics, summary metrics and validation report.

### Inventory Risk Scoring

- Created SKU-level stockout and overstock risk scoring.
- Used an 8-week operational planning horizon.
- Combined forecast demand, on-hand inventory, on-order inventory, supplier lead time, demand variability and safety stock.
- Classified stockout and overstock exposure into High, Medium and Low risk levels.
- Scored 150 SKUs.
- Identified 9 high stockout-risk SKUs.
- Identified 93 medium stockout-risk SKUs.
- Identified 3 high overstock-risk SKUs.
- Identified 8 medium overstock-risk SKUs.
- Estimated a forecast stockout gap of 15,220.67 units.
- Estimated potential lost revenue of 793,250.16.
- Identified 250.48 excess inventory units with a value of 6,347.14.
- Recommended replenishment of 40,550 units.
- Estimated recommended replenishment cost of 1,128,180.40.
- Generated SKU-level recommended actions and priority rankings.
- Saved detailed risk scores, an executive summary and a Markdown report.

### Week 4 Goal

Build the Streamlit decision dashboard using the completed forecast and inventory-risk outputs.

The dashboard will include:

- Executive KPI summary
- SKU-level forecast performance
- Stockout-risk priorities
- Overstock-risk priorities
- Replenishment recommendations
- Potential lost revenue
- Excess inventory value
- SKU filtering and downloadable results
- Model and data limitations
- Deployment-ready scoring workflow
