from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

SALES_FILE = RAW_DATA_DIR / "sales_transactions.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products_master.csv"
INVENTORY_FILE = RAW_DATA_DIR / "inventory_daily.csv"


# ---------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------

REQUIRED_SALES_COLUMNS = {
    "Invoice",
    "InvoiceDate",
    "CustomerID",
    "StockCode",
    "Description",
    "Category",
    "Quantity",
    "UnitPrice",
    "Country",
    "Channel",
    "TotalPrice",
}

REQUIRED_PRODUCT_COLUMNS = {
    "StockCode",
    "Description",
    "Category",
    "UnitCost",
    "UnitPrice",
    "LeadTimeDays",
    "ShelfLifeDays",
    "Supplier",
    "MinOrderQty",
}

REQUIRED_INVENTORY_COLUMNS = {
    "Date",
    "StockCode",
    "OpeningStock",
    "UnitsSold",
    "LostSales",
    "ClosingStock",
    "OnOrder",
    "ReorderPlaced",
    "Stockout",
    "ReorderPoint",
}


# ---------------------------------------------------------
# LOAD AND VALIDATE RAW DATA
# ---------------------------------------------------------

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three original raw datasets."""

    required_files = [SALES_FILE, PRODUCTS_FILE, INVENTORY_FILE]

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        missing_names = ", ".join(path.name for path in missing_files)
        raise FileNotFoundError(
            f"Required raw files are missing: {missing_names}"
        )

    sales = pd.read_csv(SALES_FILE)
    products = pd.read_csv(PRODUCTS_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    return sales, products, inventory


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Check that a dataset contains all required columns."""

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(f"{dataset_name}: required-column validation passed")


def validate_unique_key(
    dataframe: pd.DataFrame,
    key_columns: list[str],
    dataset_name: str,
) -> None:
    """Ensure that the requested key is unique."""

    duplicate_count = int(
        dataframe.duplicated(subset=key_columns).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} contains {duplicate_count} duplicate "
            f"records for key {key_columns}."
        )


# ---------------------------------------------------------
# CLEAN SALES DATA
# ---------------------------------------------------------

def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Clean sales transactions for fulfilled-demand analysis."""

    cleaned = sales.copy()
    original_rows = len(cleaned)

    cleaned["Invoice"] = (
        cleaned["Invoice"].astype("string").str.strip()
    )
    cleaned["StockCode"] = (
        cleaned["StockCode"].astype("string").str.strip()
    )

    cleaned["InvoiceDate"] = pd.to_datetime(
        cleaned["InvoiceDate"],
        errors="coerce",
    )
    cleaned["Quantity"] = pd.to_numeric(
        cleaned["Quantity"],
        errors="coerce",
    )
    cleaned["UnitPrice"] = pd.to_numeric(
        cleaned["UnitPrice"],
        errors="coerce",
    )

    cancellation_mask = (
        cleaned["Invoice"]
        .str.upper()
        .str.startswith("C", na=False)
    )
    invalid_invoice_mask = (
        cleaned["Invoice"].isna()
        | cleaned["Invoice"].eq("")
    )
    invalid_date_mask = cleaned["InvoiceDate"].isna()
    invalid_sku_mask = (
        cleaned["StockCode"].isna()
        | cleaned["StockCode"].eq("")
    )
    invalid_quantity_mask = (
        cleaned["Quantity"].isna()
        | cleaned["Quantity"].le(0)
    )
    invalid_price_mask = (
        cleaned["UnitPrice"].isna()
        | cleaned["UnitPrice"].le(0)
    )

    invalid_record_mask = (
        cancellation_mask
        | invalid_invoice_mask
        | invalid_date_mask
        | invalid_sku_mask
        | invalid_quantity_mask
        | invalid_price_mask
    )

    cleaned = cleaned.loc[~invalid_record_mask].copy()

    cleaned["TotalPrice"] = (
        cleaned["Quantity"] * cleaned["UnitPrice"]
    )
    cleaned["date"] = cleaned["InvoiceDate"].dt.normalize()

    removed_rows = original_rows - len(cleaned)

    print("\nSales cleaning summary:")
    print(f"Original sales rows: {original_rows:,}")
    print(
        "Cancellation rows identified:",
        f"{int(cancellation_mask.sum()):,}",
    )
    print(
        "Invalid-invoice rows identified:",
        f"{int(invalid_invoice_mask.sum()):,}",
    )
    print(
        "Invalid-date rows identified:",
        f"{int(invalid_date_mask.sum()):,}",
    )
    print(
        "Invalid-SKU rows identified:",
        f"{int(invalid_sku_mask.sum()):,}",
    )
    print(
        "Non-positive or missing quantity rows identified:",
        f"{int(invalid_quantity_mask.sum()):,}",
    )
    print(
        "Non-positive or missing price rows identified:",
        f"{int(invalid_price_mask.sum()):,}",
    )
    print(f"Total rows excluded: {removed_rows:,}")
    print(f"Valid sales rows retained: {len(cleaned):,}")

    if cleaned.empty:
        raise ValueError(
            "Sales cleaning removed every record."
        )

    return cleaned


# ---------------------------------------------------------
# CREATE SALES DAILY
# ---------------------------------------------------------

def create_sales_daily(
    cleaned_sales: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Create one fulfilled-demand record per date and SKU."""

    product_prices = products[
        ["StockCode", "UnitPrice"]
    ].copy()

    product_prices["StockCode"] = (
        product_prices["StockCode"]
        .astype("string")
        .str.strip()
    )
    product_prices["UnitPrice"] = pd.to_numeric(
        product_prices["UnitPrice"],
        errors="coerce",
    )

    validate_unique_key(
        product_prices,
        ["StockCode"],
        "Product prices",
    )

    invalid_product_prices = int(
        (
            product_prices["UnitPrice"].isna()
            | product_prices["UnitPrice"].le(0)
        ).sum()
    )

    if invalid_product_prices > 0:
        raise ValueError(
            "Product master contains invalid list prices."
        )

    product_prices = product_prices.rename(
        columns={"UnitPrice": "list_price"}
    )

    sales_with_price = cleaned_sales.merge(
        product_prices,
        on="StockCode",
        how="left",
        validate="many_to_one",
    )

    missing_list_prices = int(
        sales_with_price["list_price"].isna().sum()
    )

    if missing_list_prices > 0:
        raise ValueError(
            f"{missing_list_prices} sales rows have no matching "
            "product list price."
        )

    sales_with_price["promo_flag"] = (
        sales_with_price["UnitPrice"]
        < sales_with_price["list_price"]
    ).astype("int8")

    sales_daily = (
        sales_with_price
        .groupby(
            ["date", "StockCode"],
            as_index=False,
        )
        .agg(
            units_sold=("Quantity", "sum"),
            revenue=("TotalPrice", "sum"),
            transaction_count=("StockCode", "size"),
            order_count=("Invoice", "nunique"),
            promo_flag=("promo_flag", "max"),
        )
    )

    sales_daily["average_unit_price"] = (
        sales_daily["revenue"]
        / sales_daily["units_sold"]
    )

    sales_daily = sales_daily.rename(
        columns={"StockCode": "sku_id"}
    )

    sales_daily = sales_daily[
        [
            "date",
            "sku_id",
            "units_sold",
            "revenue",
            "average_unit_price",
            "transaction_count",
            "order_count",
            "promo_flag",
        ]
    ]

    sales_daily = (
        sales_daily
        .sort_values(["date", "sku_id"])
        .reset_index(drop=True)
    )

    validate_unique_key(
        sales_daily,
        ["date", "sku_id"],
        "sales_daily",
    )

    if sales_daily["units_sold"].le(0).any():
        raise ValueError(
            "sales_daily contains non-positive units_sold."
        )

    if sales_daily["revenue"].le(0).any():
        raise ValueError(
            "sales_daily contains non-positive revenue."
        )

    if sales_daily.isna().any().any():
        missing_summary = sales_daily.isna().sum()
        missing_summary = missing_summary[missing_summary > 0]
        raise ValueError(
            "sales_daily contains missing values: "
            f"{missing_summary.to_dict()}"
        )

    print("\nsales_daily created:")
    print(f"Rows: {len(sales_daily):,}")
    print(
        "Unique SKUs:",
        f"{sales_daily['sku_id'].nunique():,}",
    )
    print(
        "Date range:",
        sales_daily["date"].min(),
        "to",
        sales_daily["date"].max(),
    )

    return sales_daily


# ---------------------------------------------------------
# CREATE SKU MASTER
# ---------------------------------------------------------

def create_sku_master(
    products: pd.DataFrame,
    cleaned_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Create one product-master record per SKU."""

    product_data = products.copy()

    for column in [
        "StockCode",
        "Description",
        "Category",
        "Supplier",
    ]:
        product_data[column] = (
            product_data[column]
            .astype("string")
            .str.strip()
        )

    validate_unique_key(
        product_data,
        ["StockCode"],
        "Product master",
    )

    numeric_columns = [
        "UnitCost",
        "UnitPrice",
        "LeadTimeDays",
        "ShelfLifeDays",
        "MinOrderQty",
    ]

    for column in numeric_columns:
        product_data[column] = pd.to_numeric(
            product_data[column],
            errors="coerce",
        )

    invalid_numeric_rows = (
        product_data[numeric_columns].isna().any(axis=1)
        | product_data[numeric_columns].le(0).any(axis=1)
    )

    if invalid_numeric_rows.any():
        raise ValueError(
            "Product master contains missing or non-positive "
            "numeric values."
        )

    invalid_text_rows = (
        product_data[
            ["StockCode", "Description", "Category", "Supplier"]
        ]
        .isna()
        .any(axis=1)
        | product_data[
            ["StockCode", "Description", "Category", "Supplier"]
        ]
        .eq("")
        .any(axis=1)
    )

    if invalid_text_rows.any():
        raise ValueError(
            "Product master contains missing text identifiers."
        )

    launch_dates = (
        cleaned_sales
        .groupby("StockCode", as_index=False)
        .agg(launch_date=("date", "min"))
    )

    sku_master = product_data.merge(
        launch_dates,
        on="StockCode",
        how="left",
        validate="one_to_one",
    )

    sku_master["subcategory"] = sku_master["Category"]

    sku_master = sku_master.rename(
        columns={
            "StockCode": "sku_id",
            "Description": "description",
            "Category": "category",
            "UnitCost": "unit_cost",
            "UnitPrice": "list_price",
            "LeadTimeDays": "lead_time_days",
            "ShelfLifeDays": "shelf_life_days",
            "Supplier": "supplier",
            "MinOrderQty": "minimum_order_quantity",
        }
    )

    sku_master = sku_master[
        [
            "sku_id",
            "description",
            "category",
            "subcategory",
            "launch_date",
            "unit_cost",
            "list_price",
            "lead_time_days",
            "shelf_life_days",
            "supplier",
            "minimum_order_quantity",
        ]
    ]

    sku_master = (
        sku_master
        .sort_values("sku_id")
        .reset_index(drop=True)
    )

    validate_unique_key(
        sku_master,
        ["sku_id"],
        "sku_master",
    )

    if sku_master.isna().any().any():
        missing_summary = sku_master.isna().sum()
        missing_summary = missing_summary[missing_summary > 0]
        raise ValueError(
            "sku_master contains missing values: "
            f"{missing_summary.to_dict()}"
        )

    print("\nsku_master created:")
    print(f"Rows: {len(sku_master):,}")
    print(
        "Unique SKUs:",
        f"{sku_master['sku_id'].nunique():,}",
    )
    print(
        "Launch-date range:",
        sku_master["launch_date"].min(),
        "to",
        sku_master["launch_date"].max(),
    )

    return sku_master


# ---------------------------------------------------------
# CREATE INVENTORY SNAPSHOTS
# ---------------------------------------------------------

def create_inventory_snapshots(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Create one inventory snapshot per date and SKU."""

    inventory_data = inventory.copy()

    inventory_data["StockCode"] = (
        inventory_data["StockCode"]
        .astype("string")
        .str.strip()
    )
    inventory_data["Date"] = pd.to_datetime(
        inventory_data["Date"],
        errors="coerce",
    )

    numeric_columns = [
        "OpeningStock",
        "UnitsSold",
        "LostSales",
        "ClosingStock",
        "OnOrder",
        "ReorderPlaced",
        "Stockout",
        "ReorderPoint",
    ]

    for column in numeric_columns:
        inventory_data[column] = pd.to_numeric(
            inventory_data[column],
            errors="coerce",
        )

    invalid_dates = int(
        inventory_data["Date"].isna().sum()
    )
    invalid_skus = int(
        (
            inventory_data["StockCode"].isna()
            | inventory_data["StockCode"].eq("")
        ).sum()
    )

    if invalid_dates > 0:
        raise ValueError(
            f"Inventory contains {invalid_dates} invalid dates."
        )

    if invalid_skus > 0:
        raise ValueError(
            f"Inventory contains {invalid_skus} invalid SKU values."
        )

    missing_numeric = (
        inventory_data[numeric_columns]
        .isna()
        .sum()
    )
    missing_numeric = missing_numeric[missing_numeric > 0]

    if not missing_numeric.empty:
        raise ValueError(
            "Inventory contains missing numeric values: "
            f"{missing_numeric.to_dict()}"
        )

    negative_counts = (
        inventory_data[numeric_columns] < 0
    ).sum()
    negative_counts = negative_counts[negative_counts > 0]

    if not negative_counts.empty:
        raise ValueError(
            "Inventory contains negative numeric values: "
            f"{negative_counts.to_dict()}"
        )

    inventory_snapshots = inventory_data[
        [
            "Date",
            "StockCode",
            "OpeningStock",
            "UnitsSold",
            "LostSales",
            "ClosingStock",
            "OnOrder",
            "ReorderPlaced",
            "Stockout",
            "ReorderPoint",
        ]
    ].copy()

    inventory_snapshots = inventory_snapshots.rename(
        columns={
            "Date": "date",
            "StockCode": "sku_id",
            "OpeningStock": "opening_stock",
            "UnitsSold": "units_sold",
            "LostSales": "lost_sales",
            "ClosingStock": "on_hand_units",
            "OnOrder": "on_order_units",
            "ReorderPlaced": "reorder_placed",
            "Stockout": "stockout_flag",
            "ReorderPoint": "reorder_point",
        }
    )

    inventory_snapshots = (
        inventory_snapshots
        .sort_values(["date", "sku_id"])
        .reset_index(drop=True)
    )

    validate_unique_key(
        inventory_snapshots,
        ["date", "sku_id"],
        "inventory_snapshots",
    )

    if inventory_snapshots.isna().any().any():
        missing_summary = inventory_snapshots.isna().sum()
        missing_summary = missing_summary[missing_summary > 0]
        raise ValueError(
            "inventory_snapshots contains missing values: "
            f"{missing_summary.to_dict()}"
        )

    print("\ninventory_snapshots created:")
    print(f"Rows: {len(inventory_snapshots):,}")
    print(
        "Unique SKUs:",
        f"{inventory_snapshots['sku_id'].nunique():,}",
    )
    print(
        "Date range:",
        inventory_snapshots["date"].min(),
        "to",
        inventory_snapshots["date"].max(),
    )

    return inventory_snapshots


# ---------------------------------------------------------
# CREATE CALENDAR
# ---------------------------------------------------------

def create_calendar(
    sales_daily: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Create one calendar record for every date in the data range."""

    minimum_date = min(
        sales_daily["date"].min(),
        inventory_snapshots["date"].min(),
    )
    maximum_date = max(
        sales_daily["date"].max(),
        inventory_snapshots["date"].max(),
    )

    calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                start=minimum_date,
                end=maximum_date,
                freq="D",
            )
        }
    )

    iso_calendar = calendar["date"].dt.isocalendar()

    calendar["year"] = calendar["date"].dt.year
    calendar["month"] = calendar["date"].dt.month
    calendar["month_name"] = calendar["date"].dt.month_name()
    calendar["quarter"] = calendar["date"].dt.quarter
    calendar["week_of_year"] = iso_calendar.week.astype(int)
    calendar["day_of_week"] = calendar["date"].dt.dayofweek
    calendar["day_name"] = calendar["date"].dt.day_name()
    calendar["is_weekend"] = (
        calendar["day_of_week"] >= 5
    ).astype("int8")

    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    calendar["season"] = calendar["month"].map(season_map)

    # No approved external holiday calendar is available in Week 1.
    calendar["is_holiday"] = 0

    daily_promotions = (
        sales_daily
        .groupby("date", as_index=False)
        .agg(promo_event=("promo_flag", "max"))
    )

    calendar = calendar.merge(
        daily_promotions,
        on="date",
        how="left",
        validate="one_to_one",
    )

    calendar["promo_event"] = (
        calendar["promo_event"]
        .fillna(0)
        .astype("int8")
    )

    validate_unique_key(
        calendar,
        ["date"],
        "calendar",
    )

    if calendar.isna().any().any():
        missing_summary = calendar.isna().sum()
        missing_summary = missing_summary[missing_summary > 0]
        raise ValueError(
            "calendar contains missing values: "
            f"{missing_summary.to_dict()}"
        )

    print("\ncalendar created:")
    print(f"Rows: {len(calendar):,}")
    print(
        "Date range:",
        calendar["date"].min(),
        "to",
        calendar["date"].max(),
    )

    return calendar


# ---------------------------------------------------------
# CREATE ANALYSIS-READY DATASET
# ---------------------------------------------------------

def create_analysis_ready(
    sales_daily: pd.DataFrame,
    sku_master: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Combine sales, product, calendar and inventory information."""

    inventory_base = inventory_snapshots.rename(
        columns={
            "units_sold": "inventory_units_sold",
        }
    ).copy()

    analysis_ready = inventory_base.merge(
        sales_daily,
        on=["date", "sku_id"],
        how="left",
        validate="one_to_one",
        indicator="sales_merge_status",
    )

    analysis_ready["sales_observed_flag"] = (
        analysis_ready["sales_merge_status"].eq("both")
    ).astype("int8")

    sales_zero_fill_columns = [
        "units_sold",
        "revenue",
        "transaction_count",
        "order_count",
        "promo_flag",
    ]

    analysis_ready[sales_zero_fill_columns] = (
        analysis_ready[sales_zero_fill_columns]
        .fillna(0)
    )

    analysis_ready = analysis_ready.rename(
        columns={
            "units_sold": "fulfilled_units_sold",
        }
    )

    analysis_ready = analysis_ready.merge(
        sku_master,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    analysis_ready = analysis_ready.merge(
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
    )

    analysis_ready["effective_unit_price"] = (
        analysis_ready["average_unit_price"]
        .fillna(analysis_ready["list_price"])
    )

    analysis_ready["inventory_position"] = (
        analysis_ready["on_hand_units"]
        + analysis_ready["on_order_units"]
    )

    analysis_ready["gross_margin_per_unit"] = (
        analysis_ready["list_price"]
        - analysis_ready["unit_cost"]
    )

    analysis_ready["gross_margin_rate"] = (
        analysis_ready["gross_margin_per_unit"]
        / analysis_ready["list_price"]
    )

    analysis_ready = analysis_ready.drop(
        columns=[
            "average_unit_price",
            "sales_merge_status",
        ]
    )

    integer_columns = [
        "fulfilled_units_sold",
        "transaction_count",
        "order_count",
        "promo_flag",
    ]

    for column in integer_columns:
        analysis_ready[column] = (
            analysis_ready[column]
            .astype("int64")
        )

    analysis_ready = (
        analysis_ready
        .sort_values(["date", "sku_id"])
        .reset_index(drop=True)
    )

    validate_unique_key(
        analysis_ready,
        ["date", "sku_id"],
        "analysis_ready",
    )

    if len(analysis_ready) != len(inventory_snapshots):
        raise ValueError(
            "analysis_ready row count does not match "
            "inventory_snapshots."
        )

    if analysis_ready.isna().any().any():
        missing_summary = analysis_ready.isna().sum()
        missing_summary = missing_summary[missing_summary > 0]
        raise ValueError(
            "analysis_ready contains missing values: "
            f"{missing_summary.to_dict()}"
        )

    print("\nanalysis_ready created:")
    print(f"Rows: {len(analysis_ready):,}")
    print(f"Columns: {analysis_ready.shape[1]:,}")
    print(
        "Unique SKUs:",
        f"{analysis_ready['sku_id'].nunique():,}",
    )
    print(
        "Date range:",
        analysis_ready["date"].min(),
        "to",
        analysis_ready["date"].max(),
    )

    return analysis_ready


# ---------------------------------------------------------
# SAVE AND VERIFY OUTPUTS
# ---------------------------------------------------------

def save_dataset(
    dataframe: pd.DataFrame,
    filename: str,
) -> Path:
    """Save a processed dataset as CSV and verify it can be reopened."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = PROCESSED_DATA_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    if not output_path.exists():
        raise OSError(
            f"Failed to save {filename}."
        )

    reopened = pd.read_csv(output_path)

    if reopened.shape != dataframe.shape:
        raise OSError(
            f"Saved file verification failed for {filename}: "
            f"expected shape {dataframe.shape}, "
            f"found {reopened.shape}."
        )

    print(f"Saved: {output_path}")

    return output_path


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main() -> None:
    """Run the complete Project FORESIGHT Week 1 data pipeline."""

    print("=" * 65)
    print("PROJECT FORESIGHT DATA PIPELINE")
    print("=" * 65)

    print("\nLoading raw datasets...")
    sales, products, inventory = load_raw_data()

    print(f"Sales loaded: {sales.shape}")
    print(f"Products loaded: {products.shape}")
    print(f"Inventory loaded: {inventory.shape}")

    print("\nValidating required columns...")
    validate_columns(
        sales,
        REQUIRED_SALES_COLUMNS,
        "Sales transactions",
    )
    validate_columns(
        products,
        REQUIRED_PRODUCT_COLUMNS,
        "Product master",
    )
    validate_columns(
        inventory,
        REQUIRED_INVENTORY_COLUMNS,
        "Inventory",
    )

    print("\nCleaning sales transactions...")
    cleaned_sales = clean_sales(sales)

    print("\nCreating processed datasets...")

    sales_daily = create_sales_daily(
        cleaned_sales,
        products,
    )

    sku_master = create_sku_master(
        products,
        cleaned_sales,
    )

    inventory_snapshots = create_inventory_snapshots(
        inventory,
    )

    calendar = create_calendar(
        sales_daily,
        inventory_snapshots,
    )

    analysis_ready = create_analysis_ready(
        sales_daily,
        sku_master,
        inventory_snapshots,
        calendar,
    )

    print("\nSaving processed datasets...")

    save_dataset(
        sales_daily,
        "sales_daily.csv",
    )
    save_dataset(
        sku_master,
        "sku_master.csv",
    )
    save_dataset(
        inventory_snapshots,
        "inventory_snapshots.csv",
    )
    save_dataset(
        calendar,
        "calendar.csv",
    )
    save_dataset(
        analysis_ready,
        "analysis_ready.csv",
    )

    print("\n" + "=" * 65)
    print("COMPLETE WEEK 1 PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()