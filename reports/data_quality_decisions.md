# Project FORESIGHT – Data Quality Decisions

## 1. Source Dataset Summary

| Dataset | Rows | Columns | Main Purpose |
|---|---:|---:|---|
| sales_transactions.csv | 357,645 | 11 | Historical sales and demand |
| products_master.csv | 150 | 9 | Product, price, cost and lead-time details |
| inventory_daily.csv | 109,650 | 11 | Daily inventory and stock information |

## 2. Data-Quality Findings and Decisions

| Dataset | Issue | Records | Decision | Reason |
|---|---|---:|---|---|
| Sales | Missing CustomerID | 17,882 | Retain | CustomerID is not required for SKU-level forecasting |
| Sales | Quantity less than or equal to zero | 10,750 | Exclude from fulfilled-demand calculations | These records represent cancelled transactions |
| Sales | Invoice begins with C | 10,750 | Mark as cancellation and exclude from demand | Cancellation rows should not increase or reduce fulfilled demand |
| Sales | InvoiceDate stored as text | All rows | Convert to datetime | Required for daily and weekly time-series analysis |
| Sales | StockCode stored as text | All rows | Standardise as string | Ensures consistent joins between datasets |
| Sales | Duplicate rows | 0 | No action | No exact duplicates exist |
| Sales | Invalid UnitPrice | 0 | No action | All prices are positive |
| Products | Duplicate StockCode | 0 | No action | Every product has one master record |
| Products | Missing values | 0 | No action | Product data is complete |
| Products | Invalid cost or price | 0 | No action | All values are positive |
| Products | Invalid lead time | 0 | No action | All lead times are positive |
| Inventory | Duplicate Date-SKU records | 0 | No action | Every date-SKU combination is unique |
| Inventory | Missing values | 0 | No action | Inventory data is complete |
| Inventory | Negative values | 0 | No action | No invalid negative stock values exist |
| Inventory | Date stored as text | All rows | Convert to datetime | Required for time-series joins |
| Inventory | StockCode stored as text | All rows | Standardise as string | Ensures consistent joins |

## 3. Cleaning Principles

- Raw files will never be manually edited.
- All cleaning will be implemented in Python.
- The pipeline must recreate all processed datasets from raw data.
- Cancellation transactions will be preserved in the raw data but excluded from fulfilled-demand calculations.
- Missing CustomerID records will be retained because they still contain valid SKU demand information.
- Dates will be converted using controlled datetime parsing.
- Product identifiers will be standardised before joining tables.
- Cleaning steps and removed-row counts will be printed by the pipeline.