# Project FORESIGHT – Source-to-Target Data Mapping

## 1. sales_daily.csv

**Source:** `sales_transactions.csv`

**Grain:** One row per date and SKU.

| Target Column | Source Column / Rule |
|---|---|
| date | Date extracted from InvoiceDate |
| sku_id | StockCode |
| units_sold | Sum of valid positive Quantity |
| revenue | Sum of valid TotalPrice |
| average_unit_price | Revenue divided by units sold |
| transaction_count | Count of transaction rows |
| order_count | Number of unique invoices |
| promo_flag | 1 when transaction price is below product list price, otherwise 0 |

Cancellation invoices and non-positive quantities will be excluded.

## 2. sku_master.csv

**Sources:** `products_master.csv` and `sales_transactions.csv`

**Grain:** One row per SKU.

| Target Column | Source Column / Rule |
|---|---|
| sku_id | StockCode |
| description | Description |
| category | Category |
| subcategory | Category used temporarily because no separate subcategory exists |
| launch_date | Earliest valid sales date for the SKU |
| unit_cost | UnitCost |
| list_price | UnitPrice |
| lead_time_days | LeadTimeDays |
| shelf_life_days | ShelfLifeDays |
| supplier | Supplier |
| minimum_order_quantity | MinOrderQty |

## 3. inventory_snapshots.csv

**Source:** `inventory_daily.csv`

**Grain:** One row per date and SKU.

| Target Column | Source Column |
|---|---|
| date | Date |
| sku_id | StockCode |
| opening_stock | OpeningStock |
| units_sold | UnitsSold |
| lost_sales | LostSales |
| on_hand_units | ClosingStock |
| on_order_units | OnOrder |
| reorder_placed | ReorderPlaced |
| stockout_flag | Stockout |
| reorder_point | ReorderPoint |
| safety_stock | Derived later from demand variability, lead time and service-level assumptions; it is not directly available in the raw data |

## 4. calendar.csv

**Source:** Generated from the minimum and maximum dataset dates.

**Grain:** One row per calendar date.

| Target Column | Rule |
|---|---|
| date | Every date from minimum to maximum date |
| year | Calendar year |
| month | Month number |
| month_name | Month name |
| quarter | Calendar quarter |
| week_of_year | ISO week number |
| day_of_week | Weekday number |
| day_name | Weekday name |
| is_weekend | 1 for Saturday or Sunday |
| season | Derived from month |
| is_holiday | Initially 0 unless a justified holiday calendar is added |
| promo_event | Derived from promotional sales activity |

## 5. analysis_ready.csv

**Sources:** All processed datasets.

**Grain:** One row per date and SKU.

This dataset will combine:

- daily SKU demand
- product attributes
- price and cost data
- calendar features
- daily inventory position
- lead time
- reorder point
- safety stock

## Calendar Source Note

A separate raw calendar extract was not available. Therefore, `calendar.csv` is generated programmatically from the available sales and inventory date range. Promotion indicators are derived from sales data, while holiday indicators default to zero because no verified holiday source was supplied.