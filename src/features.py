from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "analysis_ready.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "weekly_demand.csv"
)


# ---------------------------------------------------------
# REQUIRED INPUT COLUMNS
# ---------------------------------------------------------

REQUIRED_COLUMNS = {
    "date",
    "sku_id",
    "fulfilled_units_sold",
    "inventory_units_sold",
    "lost_sales",
    "revenue",
    "stockout_flag",
    "promo_flag",
    "on_hand_units",
    "on_order_units",
    "reorder_point",
}


# ---------------------------------------------------------
# LOAD ANALYSIS-READY DATA
# ---------------------------------------------------------

def load_analysis_data() -> pd.DataFrame:
    """Load and validate the analysis-ready dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    dataframe = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "analysis_ready.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_count = int(
        dataframe.duplicated(
            subset=["date", "sku_id"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "analysis_ready.csv contains duplicate "
            "date-SKU records."
        )

    if dataframe["date"].isna().any():
        raise ValueError(
            "analysis_ready.csv contains invalid dates."
        )

    dataframe = (
        dataframe
        .sort_values(["sku_id", "date"])
        .reset_index(drop=True)
    )

    # Confirm the forecasting target.
    expected_demand = (
        dataframe["inventory_units_sold"]
        + dataframe["lost_sales"]
    )

    mismatched_rows = int(
        (
            dataframe["fulfilled_units_sold"]
            != expected_demand
        ).sum()
    )

    if mismatched_rows > 0:
        raise ValueError(
            f"Demand target reconciliation failed for "
            f"{mismatched_rows} rows."
        )

    dataframe["demand_target"] = (
        dataframe["fulfilled_units_sold"]
    )

    print("Input validation passed.")
    print("Input shape:", dataframe.shape)
    print(
        "Date range:",
        dataframe["date"].min(),
        "to",
        dataframe["date"].max(),
    )
    print(
        "Unique SKUs:",
        dataframe["sku_id"].nunique(),
    )

    return dataframe


# ---------------------------------------------------------
# CREATE WEEKLY DEMAND
# ---------------------------------------------------------

def create_weekly_demand(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate daily records into one row per SKU and week."""

    weekly_source = dataframe.copy()

    # Monday is used as the first day of each week.
    weekly_source["week_start"] = (
        weekly_source["date"]
        - pd.to_timedelta(
            weekly_source["date"].dt.dayofweek,
            unit="D",
        )
    )

    weekly_demand = (
        weekly_source
        .groupby(
            ["sku_id", "week_start"],
            as_index=False,
        )
        .agg(
            demand=("demand_target", "sum"),
            inventory_units_sold=(
                "inventory_units_sold",
                "sum",
            ),
            lost_sales=("lost_sales", "sum"),
            revenue=("revenue", "sum"),
            stockout_days=("stockout_flag", "sum"),
            promo_days=("promo_flag", "sum"),
            promo_week_flag=("promo_flag", "max"),
            average_on_hand_units=(
                "on_hand_units",
                "mean",
            ),
            ending_on_hand_units=(
                "on_hand_units",
                "last",
            ),
            ending_on_order_units=(
                "on_order_units",
                "last",
            ),
            average_reorder_point=(
                "reorder_point",
                "mean",
            ),
            active_demand_days=(
                "demand_target",
                lambda values: int(
                    values.gt(0).sum()
                ),
            ),
            observed_days=("date", "nunique"),
        )
    )

    weekly_demand["week_end"] = (
        weekly_demand["week_start"]
        + pd.Timedelta(days=6)
    )

    weekly_demand["is_complete_week"] = (
        weekly_demand["observed_days"]
        .eq(7)
        .astype("int8")
    )

    weekly_demand["stockout_rate"] = (
        weekly_demand["stockout_days"]
        / weekly_demand["observed_days"]
    )

    weekly_demand["lost_sales_rate"] = np.divide(
        weekly_demand["lost_sales"],
        weekly_demand["demand"],
        out=np.zeros(
            len(weekly_demand),
            dtype=float,
        ),
        where=weekly_demand["demand"].ne(0),
    )

    iso_calendar = (
        weekly_demand["week_start"]
        .dt.isocalendar()
    )

    weekly_demand["iso_year"] = (
        iso_calendar["year"]
        .astype(int)
    )

    weekly_demand["iso_week"] = (
        iso_calendar["week"]
        .astype(int)
    )

    weekly_demand["month"] = (
        weekly_demand["week_start"]
        .dt.month
    )

    weekly_demand["quarter"] = (
        weekly_demand["week_start"]
        .dt.quarter
    )

    weekly_demand = (
        weekly_demand
        .sort_values(["sku_id", "week_start"])
        .reset_index(drop=True)
    )

    return weekly_demand


# ---------------------------------------------------------
# CREATE LEAKAGE-SAFE FEATURES
# ---------------------------------------------------------

def add_forecasting_features(
    weekly_demand: pd.DataFrame,
) -> pd.DataFrame:
    """Create features using only previous-week information."""

    features = weekly_demand.copy()

    # Demand lags.
    demand_lags = [
        1,
        2,
        4,
        8,
        13,
        26,
        52,
    ]

    for lag in demand_lags:
        features[f"demand_lag_{lag}"] = (
            features
            .groupby("sku_id")["demand"]
            .shift(lag)
        )

    # Historical rolling demand features.
    for window in [4, 8, 13]:
        features[
            f"demand_rolling_mean_{window}"
        ] = (
            features
            .groupby("sku_id")["demand"]
            .transform(
                lambda values: (
                    values
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
        )

        features[
            f"demand_rolling_std_{window}"
        ] = (
            features
            .groupby("sku_id")["demand"]
            .transform(
                lambda values: (
                    values
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .std()
                )
            )
        )

    # Previous four-week zero-demand rate.
    features["zero_demand_rate_4"] = (
        features
        .groupby("sku_id")["demand"]
        .transform(
            lambda values: (
                values
                .shift(1)
                .eq(0)
                .rolling(
                    window=4,
                    min_periods=4,
                )
                .mean()
            )
        )
    )

    # Previous-week inventory and availability features.
    historical_columns = [
        "ending_on_hand_units",
        "ending_on_order_units",
        "stockout_days",
        "lost_sales",
        "promo_week_flag",
    ]

    for column in historical_columns:
        features[f"{column}_lag_1"] = (
            features
            .groupby("sku_id")[column]
            .shift(1)
        )

    return features


# ---------------------------------------------------------
# VALIDATE WEEKLY DATA
# ---------------------------------------------------------

def validate_weekly_data(
    weekly_data: pd.DataFrame,
) -> None:
    """Validate weekly output structure and key fields."""

    duplicate_count = int(
        weekly_data.duplicated(
            subset=["sku_id", "week_start"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "weekly_demand contains duplicate "
            "SKU-week records."
        )

    core_columns = [
        "sku_id",
        "week_start",
        "week_end",
        "demand",
        "inventory_units_sold",
        "lost_sales",
        "revenue",
        "observed_days",
        "is_complete_week",
    ]

    missing_core_values = int(
        weekly_data[core_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_core_values > 0:
        raise ValueError(
            "weekly_demand contains missing values "
            "in core columns."
        )

    negative_demand = int(
        weekly_data["demand"]
        .lt(0)
        .sum()
    )

    if negative_demand > 0:
        raise ValueError(
            "weekly_demand contains negative demand."
        )

    print("\nWEEKLY DATA VALIDATION")
    print("-" * 55)
    print("Shape:", weekly_data.shape)
    print(
        "Unique SKUs:",
        weekly_data["sku_id"].nunique(),
    )
    print(
        "Unique weeks:",
        weekly_data["week_start"].nunique(),
    )
    print(
        "Date range:",
        weekly_data["week_start"].min(),
        "to",
        weekly_data["week_end"].max(),
    )
    print(
        "Complete-week records:",
        int(
            weekly_data[
                "is_complete_week"
            ].sum()
        ),
    )
    print(
        "Partial-week records:",
        int(
            weekly_data[
                "is_complete_week"
            ].eq(0)
            .sum()
        ),
    )
    print(
        "Duplicate SKU-week records:",
        duplicate_count,
    )
    print(
        "Missing core values:",
        missing_core_values,
    )
    print(
        "Total weekly demand:",
        f"{weekly_data['demand'].sum():,.0f}",
    )

    print(
        "\nNote: missing values in lag and rolling "
        "feature columns are expected for early weeks."
    )


# ---------------------------------------------------------
# SAVE WEEKLY DATA
# ---------------------------------------------------------

def save_weekly_data(
    weekly_data: pd.DataFrame,
) -> None:
    """Save the weekly dataset."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    reopened = pd.read_csv(
        OUTPUT_FILE,
    )

    if reopened.shape != weekly_data.shape:
        raise OSError(
            "Saved weekly_demand.csv shape does "
            "not match the source dataframe."
        )

    print(f"\nSaved: {OUTPUT_FILE}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    """Run the Week 2 weekly feature pipeline."""

    print("=" * 65)
    print("PROJECT FORESIGHT – WEEKLY FEATURE PIPELINE")
    print("=" * 65)

    analysis_data = load_analysis_data()

    print("\nCreating weekly demand...")
    weekly_demand = create_weekly_demand(
        analysis_data
    )

    print(
        "Creating leakage-safe forecasting features..."
    )
    weekly_features = add_forecasting_features(
        weekly_demand
    )

    validate_weekly_data(
        weekly_features
    )

    save_weekly_data(
        weekly_features
    )

    print("\n" + "=" * 65)
    print("DAY 9 FEATURE PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()