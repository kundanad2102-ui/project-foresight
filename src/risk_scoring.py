from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEEKLY_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weekly_demand.csv"
)

SKU_MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sku_master.csv"
)

FUTURE_FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "future_8_week_forecast.csv"
)

RISK_SCORES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_risk_scores.csv"
)

RISK_SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "inventory_risk_summary.csv"
)

RISK_REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "inventory_risk_report.md"
)


# =========================================================
# RISK-SCORING SETTINGS
# =========================================================

FORECAST_HORIZON_WEEKS = 8
REVIEW_PERIOD_WEEKS = 1
SERVICE_LEVEL_Z = 1.65
EPSILON = 1e-9


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================

def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load weekly demand, SKU master and true future forecasts."""

    for filepath in [
        WEEKLY_DATA_FILE,
        SKU_MASTER_FILE,
        FUTURE_FORECAST_FILE,
    ]:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required file not found: {filepath}"
            )

    weekly = pd.read_csv(
        WEEKLY_DATA_FILE,
        parse_dates=[
            "week_start",
            "week_end",
        ],
    )

    sku_master = pd.read_csv(
        SKU_MASTER_FILE,
        parse_dates=[
            "launch_date",
        ],
    )

    future_forecast = pd.read_csv(
        FUTURE_FORECAST_FILE,
        parse_dates=[
            "forecast_week",
        ],
    )

    weekly_required = {
        "sku_id",
        "week_start",
        "week_end",
        "demand",
        "ending_on_hand_units",
        "ending_on_order_units",
        "average_reorder_point",
        "demand_rolling_std_13",
        "is_complete_week",
    }

    sku_required = {
        "sku_id",
        "description",
        "category",
        "subcategory",
        "unit_cost",
        "list_price",
        "lead_time_days",
        "supplier",
        "minimum_order_quantity",
    }

    forecast_required = {
        "sku_id",
        "forecast_week",
        "forecast_demand",
    }

    missing_weekly = weekly_required - set(weekly.columns)
    missing_sku = sku_required - set(sku_master.columns)
    missing_forecast = forecast_required - set(
        future_forecast.columns
    )

    if missing_weekly:
        raise ValueError(
            "Missing weekly-demand columns: "
            f"{sorted(missing_weekly)}"
        )

    if missing_sku:
        raise ValueError(
            "Missing SKU-master columns: "
            f"{sorted(missing_sku)}"
        )

    if missing_forecast:
        raise ValueError(
            "Missing future-forecast columns: "
            f"{sorted(missing_forecast)}"
        )

    weekly["sku_id"] = (
        weekly["sku_id"]
        .astype(str)
        .str.strip()
    )
    sku_master["sku_id"] = (
        sku_master["sku_id"]
        .astype(str)
        .str.strip()
    )
    future_forecast["sku_id"] = (
        future_forecast["sku_id"]
        .astype(str)
        .str.strip()
    )

    future_forecast["forecast_demand"] = pd.to_numeric(
        future_forecast["forecast_demand"],
        errors="coerce",
    )

    if future_forecast["forecast_week"].isna().any():
        raise ValueError(
            "future_8_week_forecast.csv contains invalid forecast dates."
        )

    if future_forecast["forecast_demand"].isna().any():
        raise ValueError(
            "future_8_week_forecast.csv contains missing or "
            "non-numeric forecast demand."
        )

    if future_forecast["forecast_demand"].lt(0).any():
        raise ValueError(
            "future_8_week_forecast.csv contains negative forecasts."
        )

    if weekly.duplicated(
        subset=[
            "sku_id",
            "week_start",
        ]
    ).any():
        raise ValueError(
            "Duplicate SKU-week rows found in weekly_demand.csv."
        )

    if sku_master["sku_id"].duplicated().any():
        raise ValueError(
            "Duplicate SKU rows found in sku_master.csv."
        )

    if future_forecast.duplicated(
        subset=[
            "sku_id",
            "forecast_week",
        ]
    ).any():
        raise ValueError(
            "Duplicate SKU-week rows found in "
            "future_8_week_forecast.csv."
        )

    weeks_per_sku = future_forecast.groupby("sku_id")[
        "forecast_week"
    ].nunique()

    invalid_horizons = weeks_per_sku.loc[
        ~weeks_per_sku.eq(FORECAST_HORIZON_WEEKS)
    ]

    if not invalid_horizons.empty:
        raise ValueError(
            "Every SKU must have exactly "
            f"{FORECAST_HORIZON_WEEKS} future forecast weeks. "
            f"Example invalid SKUs: {invalid_horizons.index[:10].tolist()}"
        )

    return (
        weekly,
        sku_master,
        future_forecast,
    )


# =========================================================
# BUILD ALIGNED SCORING DATA
# =========================================================

def build_scoring_base(
    weekly: pd.DataFrame,
    sku_master: pd.DataFrame,
    future_forecast: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Timestamp,
    pd.Timestamp,
    str,
]:
    """
    Build one aligned risk-scoring row per SKU.

    The latest complete inventory week is used as the scoring date.
    The next eight genuine future forecast weeks are summarised for
    inventory-risk and replenishment calculations.
    """

    complete_weekly = weekly.loc[
        weekly["is_complete_week"].eq(1)
    ].copy()

    if complete_weekly.empty:
        raise ValueError(
            "weekly_demand.csv contains no complete weeks."
        )

    scoring_week = pd.Timestamp(
        complete_weekly["week_start"].max()
    )

    if future_forecast["forecast_week"].min() <= scoring_week:
        raise ValueError(
            "Future forecast weeks must occur after the latest "
            f"complete scoring week ({scoring_week.date()})."
        )

    forecast_start_week = pd.Timestamp(
        future_forecast["forecast_week"].min()
    )

    forecast_end_week = pd.Timestamp(
        future_forecast["forecast_week"].max()
    )

    if "selected_model" in future_forecast.columns:
        selected_models = (
            future_forecast["selected_model"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    else:
        selected_models = []

    selected_model_name = (
        selected_models[0]
        if len(selected_models) == 1
        else (
            ", ".join(sorted(selected_models))
            if selected_models
            else "Future forecast model"
        )
    )

    forecast_summary = (
        future_forecast
        .sort_values(["sku_id", "forecast_week"])
        .groupby(
            "sku_id",
            as_index=False,
        )
        .agg(
            forecast_start_week=(
                "forecast_week",
                "min",
            ),
            forecast_end_week=(
                "forecast_week",
                "max",
            ),
            weekly_forecast_demand=(
                "forecast_demand",
                "mean",
            ),
            forecast_horizon_units=(
                "forecast_demand",
                "sum",
            ),
            forecast_weeks_available=(
                "forecast_week",
                "nunique",
            ),
        )
    )

    latest_inventory = (
        complete_weekly.loc[
            complete_weekly["week_start"].eq(scoring_week),
            [
                "sku_id",
                "week_start",
                "week_end",
                "demand",
                "ending_on_hand_units",
                "ending_on_order_units",
                "average_reorder_point",
                "demand_rolling_std_13",
            ],
        ]
        .copy()
    )

    scoring = (
        latest_inventory
        .merge(
            forecast_summary,
            on="sku_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            sku_master[
                [
                    "sku_id",
                    "description",
                    "category",
                    "subcategory",
                    "unit_cost",
                    "list_price",
                    "lead_time_days",
                    "supplier",
                    "minimum_order_quantity",
                ]
            ],
            on="sku_id",
            how="left",
            validate="one_to_one",
        )
    )

    missing_forecast_skus = scoring.loc[
        scoring["weekly_forecast_demand"].isna(),
        "sku_id",
    ].tolist()

    if missing_forecast_skus:
        raise ValueError(
            "Future forecasts are missing for inventory SKUs. "
            f"Example missing SKUs: {missing_forecast_skus[:10]}"
        )

    required_numeric_columns = [
        "ending_on_hand_units",
        "ending_on_order_units",
        "average_reorder_point",
        "demand_rolling_std_13",
        "weekly_forecast_demand",
        "forecast_horizon_units",
        "forecast_weeks_available",
        "unit_cost",
        "list_price",
        "lead_time_days",
        "minimum_order_quantity",
    ]

    for column in required_numeric_columns:
        scoring[column] = pd.to_numeric(
            scoring[column],
            errors="coerce",
        )

    scoring["weekly_forecast_demand"] = (
        scoring["weekly_forecast_demand"]
        .fillna(scoring["demand"])
        .clip(lower=0)
    )

    scoring["forecast_horizon_units"] = (
        scoring["forecast_horizon_units"]
        .fillna(
            scoring["weekly_forecast_demand"]
            * FORECAST_HORIZON_WEEKS
        )
        .clip(lower=0)
    )

    scoring["forecast_weeks_available"] = (
        scoring["forecast_weeks_available"]
        .fillna(0)
        .astype(int)
    )

    scoring["demand_rolling_std_13"] = (
        scoring["demand_rolling_std_13"]
        .fillna(0)
        .clip(lower=0)
    )

    scoring["ending_on_hand_units"] = (
        scoring["ending_on_hand_units"]
        .fillna(0)
        .clip(lower=0)
    )

    scoring["ending_on_order_units"] = (
        scoring["ending_on_order_units"]
        .fillna(0)
        .clip(lower=0)
    )

    scoring["average_reorder_point"] = (
        scoring["average_reorder_point"]
        .fillna(0)
        .clip(lower=0)
    )

    scoring["unit_cost"] = (
        scoring["unit_cost"]
        .fillna(0)
        .clip(lower=0)
    )

    scoring["list_price"] = (
        scoring["list_price"]
        .fillna(scoring["unit_cost"])
        .clip(lower=0)
    )

    scoring["lead_time_days"] = (
        scoring["lead_time_days"]
        .fillna(7)
        .clip(lower=1)
    )

    scoring["minimum_order_quantity"] = (
        scoring["minimum_order_quantity"]
        .fillna(1)
        .clip(lower=1)
    )

    if not scoring["forecast_weeks_available"].eq(
        FORECAST_HORIZON_WEEKS
    ).all():
        raise ValueError(
            "Every scored SKU must have exactly eight forecast weeks."
        )

    # Keep the exact overall window available for reports and dashboard use.
    scoring["forecast_start_week"] = forecast_start_week
    scoring["forecast_end_week"] = forecast_end_week

    return (
        scoring,
        scoring_week,
        forecast_start_week,
        selected_model_name,
    )


# =========================================================
# RISK CALCULATIONS
# =========================================================

def round_up_to_moq(
    required_units: pd.Series,
    minimum_order_quantity: pd.Series,
) -> np.ndarray:
    """Round recommended order quantity up to the nearest MOQ."""

    required = np.asarray(
        required_units,
        dtype=float,
    )

    moq = np.asarray(
        minimum_order_quantity,
        dtype=float,
    )

    return np.where(
        required > 0,
        np.ceil(
            required
            / moq
        )
        * moq,
        0,
    )


def assign_stockout_level(
    score: pd.Series,
    projected_balance: pd.Series,
) -> np.ndarray:
    """Convert stockout scores into operational risk levels."""

    return np.select(
        [
            (
                projected_balance < 0
            )
            & (
                score >= 50
            ),
            (
                projected_balance < 0
            )
            & (
                score > 0
            ),
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )


def assign_overstock_level(
    score: pd.Series,
    coverage_weeks: pd.Series,
) -> np.ndarray:
    """Convert overstock scores into operational risk levels."""

    return np.select(
        [
            (
                coverage_weeks
                > (
                    FORECAST_HORIZON_WEEKS
                    + 4
                )
            )
            | (
                score
                >= 50
            ),
            (
                coverage_weeks
                > FORECAST_HORIZON_WEEKS
            )
            | (
                score
                > 0
            ),
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )


def assign_recommended_action(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """Create one clear action recommendation per SKU."""

    return np.select(
        [
            dataframe[
                "stockout_risk_level"
            ].eq(
                "High"
            ),
            dataframe[
                "stockout_risk_level"
            ].eq(
                "Medium"
            ),
            dataframe[
                "overstock_risk_level"
            ].eq(
                "High"
            ),
            dataframe[
                "overstock_risk_level"
            ].eq(
                "Medium"
            ),
        ],
        [
            (
                "Expedite supply and place the "
                "recommended replenishment order"
            ),
            (
                "Review supplier timing and place or "
                "expedite the recommended order"
            ),
            (
                "Pause replenishment; consider transfer, "
                "promotion or markdown"
            ),
            (
                "Reduce or defer replenishment and "
                "monitor weekly demand"
            ),
        ],
        default=(
            "Maintain current plan and monitor weekly"
        ),
    )


def calculate_risk_scores(
    scoring: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate stockout, overstock, value-at-stake and actions."""

    result = scoring.copy()

    result["lead_time_weeks"] = np.maximum(
        1,
        np.ceil(
            result["lead_time_days"]
            / 7
        ),
    ).astype(int)

    result["inventory_position_units"] = (
        result[
            "ending_on_hand_units"
        ]
        + result[
            "ending_on_order_units"
        ]
    )

    result["lead_time_forecast_units"] = (
        result[
            "weekly_forecast_demand"
        ]
        * result[
            "lead_time_weeks"
        ]
    )

    result["forecast_horizon_units"] = (
        pd.to_numeric(
            result["forecast_horizon_units"],
            errors="coerce",
        )
        .fillna(
            result["weekly_forecast_demand"]
            * FORECAST_HORIZON_WEEKS
        )
        .clip(lower=0)
    )

    result["safety_stock_units"] = (
        SERVICE_LEVEL_Z
        * result[
            "demand_rolling_std_13"
        ]
        * np.sqrt(
            result[
                "lead_time_weeks"
            ]
        )
    )

    result["lead_time_need_units"] = (
        result[
            "lead_time_forecast_units"
        ]
        + result[
            "safety_stock_units"
        ]
    )

    result["projected_lead_time_balance_units"] = (
        result[
            "inventory_position_units"
        ]
        - result[
            "lead_time_need_units"
        ]
    )

    result["stockout_gap_units"] = (
        -result[
            "projected_lead_time_balance_units"
        ]
    ).clip(
        lower=0
    )

    result["stockout_risk_score"] = (
        100
        * result[
            "stockout_gap_units"
        ]
        / (
            result[
                "lead_time_need_units"
            ]
            + EPSILON
        )
    ).clip(
        lower=0,
        upper=100,
    )


    result["overstock_threshold_units"] = (
        result[
            "forecast_horizon_units"
        ]
        + result[
            "safety_stock_units"
        ]
    )

    result["excess_inventory_units"] = (
        result[
            "inventory_position_units"
        ]
        - result[
            "overstock_threshold_units"
        ]
    ).clip(
        lower=0
    )

    result["overstock_risk_score"] = (
        100
        * result[
            "excess_inventory_units"
        ]
        / (
            result[
                "inventory_position_units"
            ]
            + EPSILON
        )
    ).clip(
        lower=0,
        upper=100,
    )

    result["inventory_coverage_weeks"] = np.where(
        result[
            "weekly_forecast_demand"
        ]
        > 0,
        result[
            "inventory_position_units"
        ]
        / result[
            "weekly_forecast_demand"
        ],
        np.where(
            result[
                "inventory_position_units"
            ]
            > 0,
            np.inf,
            0,
        ),
    )

    result["stockout_risk_level"] = (
        assign_stockout_level(
            result[
                "stockout_risk_score"
            ],
            result[
                "projected_lead_time_balance_units"
            ],
        )
    )

    result["overstock_risk_level"] = (
        assign_overstock_level(
            result[
                "overstock_risk_score"
            ],
            result[
                "inventory_coverage_weeks"
            ],
        )
    )

    result["target_inventory_units"] = (
        result[
            "weekly_forecast_demand"
        ]
        * (
            result[
                "lead_time_weeks"
            ]
            + REVIEW_PERIOD_WEEKS
        )
        + result[
            "safety_stock_units"
        ]
    )

    result["raw_recommended_order_units"] = (
        result[
            "target_inventory_units"
        ]
        - result[
            "inventory_position_units"
        ]
    ).clip(
        lower=0
    )

    result["recommended_order_units"] = (
        round_up_to_moq(
            result[
                "raw_recommended_order_units"
            ],
            result[
                "minimum_order_quantity"
            ],
        )
    )

    result["potential_lost_revenue"] = (
        result[
            "stockout_gap_units"
        ]
        * result[
            "list_price"
        ]
    )

    result["excess_inventory_value"] = (
        result[
            "excess_inventory_units"
        ]
        * result[
            "unit_cost"
        ]
    )

    result["recommended_order_cost"] = (
        result[
            "recommended_order_units"
        ]
        * result[
            "unit_cost"
        ]
    )

    result["overall_risk_score"] = (
        result[
            [
                "stockout_risk_score",
                "overstock_risk_score",
            ]
        ]
        .max(
            axis=1
        )
    )

    result["primary_risk"] = np.where(
        result[
            "stockout_risk_score"
        ]
        >= result[
            "overstock_risk_score"
        ],
        "Stockout",
        "Overstock",
    )

    result["value_at_stake"] = np.where(
        result[
            "primary_risk"
        ].eq(
            "Stockout"
        ),
        result[
            "potential_lost_revenue"
        ],
        result[
            "excess_inventory_value"
        ],
    )

    result["recommended_action"] = (
        assign_recommended_action(
            result
        )
    )

    level_priority = {
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }

    result["priority_level"] = np.maximum(
        result[
            "stockout_risk_level"
        ].map(
            level_priority
        ),
        result[
            "overstock_risk_level"
        ].map(
            level_priority
        ),
    )

    result = (
        result
        .sort_values(
            [
                "priority_level",
                "overall_risk_score",
                "value_at_stake",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    result["priority_rank"] = (
        np.arange(
            len(result)
        )
        + 1
    )

    result["inventory_coverage_weeks"] = (
        result[
            "inventory_coverage_weeks"
        ]
        .replace(
            np.inf,
            np.nan,
        )
    )

    return result


# =========================================================
# SUMMARY AND REPORT
# =========================================================

def create_summary(
    risk_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Create an executive summary table."""

    summary_records = [
        {
            "metric": "SKUs scored",
            "value": int(
                len(risk_scores)
            ),
        },
        {
            "metric": "High stockout-risk SKUs",
            "value": int(
                risk_scores[
                    "stockout_risk_level"
                ].eq(
                    "High"
                ).sum()
            ),
        },
        {
            "metric": "Medium stockout-risk SKUs",
            "value": int(
                risk_scores[
                    "stockout_risk_level"
                ].eq(
                    "Medium"
                ).sum()
            ),
        },
        {
            "metric": "High overstock-risk SKUs",
            "value": int(
                risk_scores[
                    "overstock_risk_level"
                ].eq(
                    "High"
                ).sum()
            ),
        },
        {
            "metric": "Medium overstock-risk SKUs",
            "value": int(
                risk_scores[
                    "overstock_risk_level"
                ].eq(
                    "Medium"
                ).sum()
            ),
        },
        {
            "metric": "Forecast stockout gap units",
            "value": float(
                risk_scores[
                    "stockout_gap_units"
                ].sum()
            ),
        },
        {
            "metric": "Potential lost revenue",
            "value": float(
                risk_scores[
                    "potential_lost_revenue"
                ].sum()
            ),
        },
        {
            "metric": "Excess inventory units",
            "value": float(
                risk_scores[
                    "excess_inventory_units"
                ].sum()
            ),
        },
        {
            "metric": "Excess inventory value",
            "value": float(
                risk_scores[
                    "excess_inventory_value"
                ].sum()
            ),
        },
        {
            "metric": "Recommended replenishment units",
            "value": float(
                risk_scores[
                    "recommended_order_units"
                ].sum()
            ),
        },
        {
            "metric": "Recommended replenishment cost",
            "value": float(
                risk_scores[
                    "recommended_order_cost"
                ].sum()
            ),
        },
    ]

    return pd.DataFrame(
        summary_records
    )


def format_money(
    value: float,
) -> str:
    """Format a numeric value for the Markdown report."""

    return f"{value:,.2f}"


def write_report(
    risk_scores: pd.DataFrame,
    summary: pd.DataFrame,
    scoring_week: pd.Timestamp,
    forecast_start_week: pd.Timestamp,
    selected_model_name: str,
) -> None:
    """Write a readable inventory-risk report."""

    summary_lookup = dict(
        zip(
            summary["metric"],
            summary["value"],
        )
    )

    stockout_top = (
        risk_scores.loc[
            risk_scores[
                "stockout_risk_level"
            ].isin(
                [
                    "High",
                    "Medium",
                ]
            )
        ]
        .sort_values(
            [
                "stockout_risk_score",
                "potential_lost_revenue",
            ],
            ascending=False,
        )
        .head(10)
    )

    overstock_top = (
        risk_scores.loc[
            risk_scores[
                "overstock_risk_level"
            ].isin(
                [
                    "High",
                    "Medium",
                ]
            )
        ]
        .sort_values(
            [
                "overstock_risk_score",
                "excess_inventory_value",
            ],
            ascending=False,
        )
        .head(10)
    )

    report_lines = [
        "# Inventory Risk Scoring Report",
        "",
        "## Scoring Design",
        "",
        (
            f"- Scoring week: "
            f"{scoring_week.date()}"
        ),
        (
            "- True future forecast window: "
            f"{forecast_start_week.date()} to "
            f"{(forecast_start_week + pd.Timedelta(weeks=7)).date()}"
        ),
        (
            "- Forecast model used for future predictions: "
            f"{selected_model_name}"
        ),
        (
            "- Operational forecast horizon: "
            f"{FORECAST_HORIZON_WEEKS} weeks"
        ),
        (
            "- Safety-stock service factor: "
            f"{SERVICE_LEVEL_Z}"
        ),
        (
            "- Inventory position: ending on-hand units "
            "plus ending on-order units"
        ),
        "",
        "## Executive Summary",
        "",
        (
            "- SKUs scored: "
            f"{int(summary_lookup['SKUs scored'])}"
        ),
        (
            "- High stockout-risk SKUs: "
            f"{int(summary_lookup['High stockout-risk SKUs'])}"
        ),
        (
            "- Medium stockout-risk SKUs: "
            f"{int(summary_lookup['Medium stockout-risk SKUs'])}"
        ),
        (
            "- High overstock-risk SKUs: "
            f"{int(summary_lookup['High overstock-risk SKUs'])}"
        ),
        (
            "- Medium overstock-risk SKUs: "
            f"{int(summary_lookup['Medium overstock-risk SKUs'])}"
        ),
        (
            "- Potential lost revenue: "
            f"{format_money(summary_lookup['Potential lost revenue'])}"
        ),
        (
            "- Excess inventory value: "
            f"{format_money(summary_lookup['Excess inventory value'])}"
        ),
        (
            "- Recommended replenishment cost: "
            f"{format_money(summary_lookup['Recommended replenishment cost'])}"
        ),
        "",
        "## Risk Logic",
        "",
        (
            "Stockout risk compares inventory position with "
            "forecast demand over supplier lead time plus safety stock."
        ),
        (
            "Overstock risk compares inventory position with "
            "forecast demand over the 8-week planning horizon plus "
            "safety stock."
        ),
        (
            "Recommended order quantities replenish inventory to "
            "lead-time demand plus one review week and safety stock, "
            "then round upward to the SKU minimum order quantity."
        ),
        "",
        "## Top Stockout Priorities",
        "",
    ]

    if stockout_top.empty:
        report_lines.append(
            "No high or medium stockout-risk SKUs were identified."
        )
    else:
        report_lines.extend(
            [
                (
                    "| Rank | SKU | Category | Stockout Risk | "
                    "Gap Units | Potential Lost Revenue | Action |"
                ),
                (
                    "|---:|---|---|---:|---:|---:|---|"
                ),
            ]
        )

        for _, row in stockout_top.iterrows():
            report_lines.append(
                (
                    f"| {int(row['priority_rank'])} "
                    f"| {row['sku_id']} "
                    f"| {row['category']} "
                    f"| {row['stockout_risk_score']:.2f} "
                    f"| {row['stockout_gap_units']:.2f} "
                    f"| {row['potential_lost_revenue']:.2f} "
                    f"| {row['recommended_action']} |"
                )
            )

    report_lines.extend(
        [
            "",
            "## Top Overstock Priorities",
            "",
        ]
    )

    if overstock_top.empty:
        report_lines.append(
            "No high or medium overstock-risk SKUs were identified."
        )
    else:
        report_lines.extend(
            [
                (
                    "| Rank | SKU | Category | Overstock Risk | "
                    "Excess Units | Excess Value | Action |"
                ),
                (
                    "|---:|---|---|---:|---:|---:|---|"
                ),
            ]
        )

        for _, row in overstock_top.iterrows():
            report_lines.append(
                (
                    f"| {int(row['priority_rank'])} "
                    f"| {row['sku_id']} "
                    f"| {row['category']} "
                    f"| {row['overstock_risk_score']:.2f} "
                    f"| {row['excess_inventory_units']:.2f} "
                    f"| {row['excess_inventory_value']:.2f} "
                    f"| {row['recommended_action']} |"
                )
            )

    report_lines.extend(
        [
            "",
            "## Forecast Assumptions and Limitations",
            "",
            (
                "Risk scoring uses a newly generated recursive 8-week future "
                "forecast for every SKU. Each predicted week contributes to "
                "the next week's lag and rolling-demand features."
            ),
            (
                "No future promotion schedule was supplied, so future "
                "promotion flags are assumed to be zero. Latest known "
                "inventory inputs are carried forward for forecasting "
                "features, while risk calculations use the latest actual "
                "on-hand and on-order inventory position."
            ),
            (
                "Forecast uncertainty intervals and service-level optimisation "
                "are not yet included and remain production enhancements."
            ),
        ]
    )

    RISK_REPORT_FILE.write_text(
        "\n".join(
            report_lines
        )
        + "\n",
        encoding="utf-8",
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(
    risk_scores: pd.DataFrame,
    summary: pd.DataFrame,
    scoring_week: pd.Timestamp,
    forecast_start_week: pd.Timestamp,
    selected_model_name: str,
) -> None:
    """Save detailed scores, executive summary and report."""

    RISK_SCORES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RISK_SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "priority_rank",
        "sku_id",
        "description",
        "category",
        "subcategory",
        "supplier",
        "week_start",
        "week_end",
        "forecast_start_week",
        "forecast_end_week",
        "weekly_forecast_demand",
        "forecast_weeks_available",
        "lead_time_days",
        "lead_time_weeks",
        "ending_on_hand_units",
        "ending_on_order_units",
        "inventory_position_units",
        "average_reorder_point",
        "demand_rolling_std_13",
        "safety_stock_units",
        "lead_time_forecast_units",
        "lead_time_need_units",
        "projected_lead_time_balance_units",
        "stockout_gap_units",
        "stockout_risk_score",
        "stockout_risk_level",
        "forecast_horizon_units",
        "overstock_threshold_units",
        "excess_inventory_units",
        "overstock_risk_score",
        "overstock_risk_level",
        "inventory_coverage_weeks",
        "minimum_order_quantity",
        "recommended_order_units",
        "unit_cost",
        "list_price",
        "recommended_order_cost",
        "potential_lost_revenue",
        "excess_inventory_value",
        "overall_risk_score",
        "primary_risk",
        "value_at_stake",
        "recommended_action",
    ]

    risk_scores[
        output_columns
    ].to_csv(
        RISK_SCORES_FILE,
        index=False,
    )

    summary.to_csv(
        RISK_SUMMARY_FILE,
        index=False,
    )

    write_report(
        risk_scores=risk_scores,
        summary=summary,
        scoring_week=scoring_week,
        forecast_start_week=forecast_start_week,
        selected_model_name=selected_model_name,
    )

    print("\nSAVED RISK-SCORING OUTPUTS")
    print("-" * 72)
    print(RISK_SCORES_FILE)
    print(RISK_SUMMARY_FILE)
    print(RISK_REPORT_FILE)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """Run the complete inventory-risk scoring workflow."""

    print("=" * 72)
    print(
        "PROJECT FORESIGHT - "
        "INVENTORY RISK SCORING"
    )
    print("=" * 72)

    (
        weekly,
        sku_master,
        future_forecast,
    ) = load_inputs()

    (
        scoring,
        scoring_week,
        forecast_start_week,
        selected_model_name,
    ) = build_scoring_base(
        weekly=weekly,
        sku_master=sku_master,
        future_forecast=future_forecast,
    )

    risk_scores = calculate_risk_scores(
        scoring
    )

    summary = create_summary(
        risk_scores
    )

    print("\nSCORING SUMMARY")
    print("-" * 72)
    print(
        "Scoring week:",
        scoring_week.date(),
    )
    print(
        "True future forecast window:",
        forecast_start_week.date(),
        "to",
        (
            forecast_start_week
            + pd.Timedelta(weeks=7)
        ).date(),
    )
    print(
        "SKUs scored:",
        len(risk_scores),
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nTOP 10 PRIORITIES")
    print("-" * 72)
    print(
        risk_scores[
            [
                "priority_rank",
                "sku_id",
                "primary_risk",
                "stockout_risk_level",
                "overstock_risk_level",
                "overall_risk_score",
                "value_at_stake",
                "recommended_order_units",
                "recommended_action",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    save_outputs(
        risk_scores=risk_scores,
        summary=summary,
        scoring_week=scoring_week,
        forecast_start_week=forecast_start_week,
        selected_model_name=selected_model_name,
    )

    print("\n" + "=" * 72)
    print(
        "INVENTORY RISK SCORING "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()