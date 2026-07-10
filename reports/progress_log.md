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