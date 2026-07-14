# Project FORESIGHT Progress Log

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
- Confirmed demand_target equals units sold plus lost sales.
- Analysed demand distribution and zero-demand records.
- Analysed daily, weekly, monthly and yearly patterns.
- Identified top-moving and slow-moving SKUs.
- Analysed demand concentration.
- Analysed category performance.
- Identified stockout-risk SKUs.
- Analysed lost sales and stockout impact.
- Measured SKU-level intermittency and demand variability.
- Completed the EDA insight memo.

### Key Decisions

- Use demand_target as the forecasting target.
- Use WAPE as the primary forecasting metric.
- Do not use MAPE as the primary metric because many records contain zero demand.
- Prepare weekly SKU-level demand for baseline forecasting.

### Next Task

- Create weekly_demand.csv.
- Build seasonal-naive forecasting baseline.

## Week 2 – Baseline Forecasting Completed

### Completed

- Created weekly SKU-level demand data.
- Generated 15,900 SKU-week records for 150 SKUs.
- Created leakage-safe lag and rolling features.
- Used the final 13 complete weeks as a time-based test set.
- Evaluated previous-week and 52-week seasonal-naive baselines.
- Used WAPE, MAE, RMSE and bias for evaluation.
- Selected the previous-week naive model as the baseline.
- Saved baseline predictions and metrics.

### Baseline Result

- Selected model: Naive – Previous Week
- WAPE: 47.88%
- MAE: 69.08
- RMSE: 121.54
- Bias: -1.05%
- Evaluation rows: 1,950

### Week 3 Goal

Develop forecasting models that achieve a lower WAPE than 47.88%.