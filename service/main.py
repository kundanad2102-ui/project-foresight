from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RISK_SCORES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_risk_scores.csv"
)

FUTURE_FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "future_8_week_forecast.csv"
)

RISK_SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "inventory_risk_summary.csv"
)


# =========================================================
# SCORING SETTINGS
# =========================================================

FORECAST_HORIZON_WEEKS = 8
REVIEW_PERIOD_WEEKS = 1
SERVICE_LEVEL_Z = 1.65
EPSILON = 1e-9


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Project FORESIGHT Scoring API",
    description=(
        "SKU-level demand forecast, stockout-risk, overstock-risk, "
        "financial-impact and inventory-action scoring service."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODELS
# =========================================================

class ScoreRequest(BaseModel):
    """
    Recalculate the risk score for an existing SKU.

    Only sku_id is required. Any supplied values override the values
    stored in inventory_risk_scores.csv.
    """

    sku_id: str = Field(
        ...,
        min_length=1,
        examples=["SKU10000"],
    )

    ending_on_hand_units: float | None = Field(
        default=None,
        ge=0,
    )

    ending_on_order_units: float | None = Field(
        default=None,
        ge=0,
    )

    lead_time_days: float | None = Field(
        default=None,
        ge=1,
    )

    demand_rolling_std_13: float | None = Field(
        default=None,
        ge=0,
    )

    minimum_order_quantity: float | None = Field(
        default=None,
        ge=1,
    )

    unit_cost: float | None = Field(
        default=None,
        ge=0,
    )

    list_price: float | None = Field(
        default=None,
        ge=0,
    )


# =========================================================
# DATA HELPERS
# =========================================================

def require_file(filepath: Path) -> None:
    """Raise a service error when a required project file is missing."""

    if not filepath.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Required project file is missing: {filepath}",
        )


def load_risk_scores() -> pd.DataFrame:
    """Load and validate current SKU-level inventory-risk scores."""

    require_file(RISK_SCORES_FILE)

    dataframe = pd.read_csv(
        RISK_SCORES_FILE,
        parse_dates=[
            "week_start",
            "week_end",
        ],
    )

    required_columns = {
        "sku_id",
        "weekly_forecast_demand",
        "ending_on_hand_units",
        "ending_on_order_units",
        "lead_time_days",
        "demand_rolling_std_13",
        "minimum_order_quantity",
        "unit_cost",
        "list_price",
        "stockout_risk_level",
        "overstock_risk_level",
        "recommended_action",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "inventory_risk_scores.csv is missing columns: "
                + ", ".join(sorted(missing_columns))
            ),
        )

    dataframe["sku_id"] = (
        dataframe["sku_id"]
        .astype(str)
        .str.strip()
    )

    return dataframe


def load_future_forecast() -> pd.DataFrame:
    """Load and validate the true 8-week future forecast."""

    require_file(FUTURE_FORECAST_FILE)

    dataframe = pd.read_csv(
        FUTURE_FORECAST_FILE,
        parse_dates=["forecast_week"],
    )

    required_columns = {
        "sku_id",
        "forecast_week",
        "forecast_demand",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "future_8_week_forecast.csv is missing columns: "
                + ", ".join(sorted(missing_columns))
            ),
        )

    dataframe["sku_id"] = (
        dataframe["sku_id"]
        .astype(str)
        .str.strip()
    )

    dataframe["forecast_demand"] = (
        pd.to_numeric(
            dataframe["forecast_demand"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    return dataframe


def load_risk_summary() -> pd.DataFrame:
    """Load the executive inventory-risk summary."""

    require_file(RISK_SUMMARY_FILE)

    return pd.read_csv(
        RISK_SUMMARY_FILE
    )


def find_sku_row(
    dataframe: pd.DataFrame,
    sku_id: str,
) -> pd.Series:
    """Find one SKU row or return a clear 404 error."""

    clean_sku_id = str(sku_id).strip()

    matching_rows = dataframe.loc[
        dataframe["sku_id"].eq(clean_sku_id)
    ]

    if matching_rows.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SKU not found: {clean_sku_id}",
        )

    return matching_rows.iloc[0].copy()


def clean_json_value(value: Any) -> Any:
    """Convert pandas and NumPy values into valid JSON values."""

    if isinstance(value, dict):
        return {
            key: clean_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            clean_json_value(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            clean_json_value(item)
            for item in value
        ]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if pd.isna(value):
        return None

    return value


def numeric_value(
    row: pd.Series,
    column: str,
    default: float = 0.0,
) -> float:
    """Safely read one numeric value from a pandas row."""

    value = pd.to_numeric(
        pd.Series([row.get(column)]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return float(default)

    return float(value)


def round_up_to_moq(
    required_units: float,
    minimum_order_quantity: float,
) -> float:
    """Round an order quantity upward to the nearest MOQ."""

    required = max(
        0.0,
        float(required_units),
    )

    moq = max(
        1.0,
        float(minimum_order_quantity),
    )

    if required <= 0:
        return 0.0

    return float(
        math.ceil(required / moq)
        * moq
    )


# =========================================================
# RISK LOGIC
# =========================================================

def assign_stockout_level(
    score: float,
    projected_balance: float,
) -> str:
    """Assign Low, Medium or High stockout risk."""

    if projected_balance < 0 and score >= 50:
        return "High"

    if projected_balance < 0 and score > 0:
        return "Medium"

    return "Low"


def assign_overstock_level(
    score: float,
    coverage_weeks: float | None,
) -> str:
    """Assign Low, Medium or High overstock risk."""

    if coverage_weeks is None:
        return "High" if score >= 50 else "Medium"

    if (
        coverage_weeks
        > FORECAST_HORIZON_WEEKS + 4
        or score >= 50
    ):
        return "High"

    if (
        coverage_weeks
        > FORECAST_HORIZON_WEEKS
        or score > 0
    ):
        return "Medium"

    return "Low"


def assign_recommended_action(
    stockout_level: str,
    overstock_level: str,
) -> str:
    """Create one operational recommendation."""

    if stockout_level == "High":
        return (
            "Expedite supply and place the recommended "
            "replenishment order"
        )

    if stockout_level == "Medium":
        return (
            "Review supplier timing and place or expedite "
            "the recommended order"
        )

    if overstock_level == "High":
        return (
            "Pause replenishment; consider transfer, "
            "promotion or markdown"
        )

    if overstock_level == "Medium":
        return (
            "Reduce or defer replenishment and monitor "
            "weekly demand"
        )

    return "Maintain current plan and monitor weekly"


def calculate_custom_score(
    base_row: pd.Series,
    request: ScoreRequest,
) -> dict[str, Any]:
    """Recalculate inventory risk using optional request overrides."""

    weekly_forecast_demand = max(
        0.0,
        numeric_value(
            base_row,
            "weekly_forecast_demand",
        ),
    )

    ending_on_hand_units = (
        request.ending_on_hand_units
        if request.ending_on_hand_units is not None
        else numeric_value(
            base_row,
            "ending_on_hand_units",
        )
    )

    ending_on_order_units = (
        request.ending_on_order_units
        if request.ending_on_order_units is not None
        else numeric_value(
            base_row,
            "ending_on_order_units",
        )
    )

    lead_time_days = (
        request.lead_time_days
        if request.lead_time_days is not None
        else numeric_value(
            base_row,
            "lead_time_days",
            default=7,
        )
    )

    demand_rolling_std_13 = (
        request.demand_rolling_std_13
        if request.demand_rolling_std_13 is not None
        else numeric_value(
            base_row,
            "demand_rolling_std_13",
        )
    )

    minimum_order_quantity = (
        request.minimum_order_quantity
        if request.minimum_order_quantity is not None
        else numeric_value(
            base_row,
            "minimum_order_quantity",
            default=1,
        )
    )

    unit_cost = (
        request.unit_cost
        if request.unit_cost is not None
        else numeric_value(
            base_row,
            "unit_cost",
        )
    )

    list_price = (
        request.list_price
        if request.list_price is not None
        else numeric_value(
            base_row,
            "list_price",
            default=unit_cost,
        )
    )

    ending_on_hand_units = max(
        0.0,
        float(ending_on_hand_units),
    )

    ending_on_order_units = max(
        0.0,
        float(ending_on_order_units),
    )

    lead_time_days = max(
        1.0,
        float(lead_time_days),
    )

    demand_rolling_std_13 = max(
        0.0,
        float(demand_rolling_std_13),
    )

    minimum_order_quantity = max(
        1.0,
        float(minimum_order_quantity),
    )

    unit_cost = max(
        0.0,
        float(unit_cost),
    )

    list_price = max(
        0.0,
        float(list_price),
    )

    lead_time_weeks = max(
        1,
        math.ceil(
            lead_time_days / 7
        ),
    )

    inventory_position_units = (
        ending_on_hand_units
        + ending_on_order_units
    )

    lead_time_forecast_units = (
        weekly_forecast_demand
        * lead_time_weeks
    )

    forecast_horizon_units = (
        weekly_forecast_demand
        * FORECAST_HORIZON_WEEKS
    )

    safety_stock_units = (
        SERVICE_LEVEL_Z
        * demand_rolling_std_13
        * math.sqrt(
            lead_time_weeks
        )
    )

    lead_time_need_units = (
        lead_time_forecast_units
        + safety_stock_units
    )

    projected_balance_units = (
        inventory_position_units
        - lead_time_need_units
    )

    stockout_gap_units = max(
        0.0,
        -projected_balance_units,
    )

    stockout_risk_score = min(
        100.0,
        max(
            0.0,
            100
            * stockout_gap_units
            / (
                lead_time_need_units
                + EPSILON
            ),
        ),
    )

    overstock_threshold_units = (
        forecast_horizon_units
        + safety_stock_units
    )

    excess_inventory_units = max(
        0.0,
        inventory_position_units
        - overstock_threshold_units,
    )

    overstock_risk_score = min(
        100.0,
        max(
            0.0,
            100
            * excess_inventory_units
            / (
                inventory_position_units
                + EPSILON
            ),
        ),
    )

    if weekly_forecast_demand > 0:
        inventory_coverage_weeks: float | None = (
            inventory_position_units
            / weekly_forecast_demand
        )
    elif inventory_position_units > 0:
        inventory_coverage_weeks = None
    else:
        inventory_coverage_weeks = 0.0

    stockout_risk_level = (
        assign_stockout_level(
            score=stockout_risk_score,
            projected_balance=projected_balance_units,
        )
    )

    overstock_risk_level = (
        assign_overstock_level(
            score=overstock_risk_score,
            coverage_weeks=inventory_coverage_weeks,
        )
    )

    target_inventory_units = (
        weekly_forecast_demand
        * (
            lead_time_weeks
            + REVIEW_PERIOD_WEEKS
        )
        + safety_stock_units
    )

    raw_recommended_order_units = max(
        0.0,
        target_inventory_units
        - inventory_position_units,
    )

    recommended_order_units = (
        round_up_to_moq(
            required_units=raw_recommended_order_units,
            minimum_order_quantity=(
                minimum_order_quantity
            ),
        )
    )

    potential_lost_revenue = (
        stockout_gap_units
        * list_price
    )

    excess_inventory_value = (
        excess_inventory_units
        * unit_cost
    )

    recommended_order_cost = (
        recommended_order_units
        * unit_cost
    )

    overall_risk_score = max(
        stockout_risk_score,
        overstock_risk_score,
    )

    if (
        stockout_risk_score
        >= overstock_risk_score
    ):
        primary_risk = "Stockout"
        value_at_stake = potential_lost_revenue
    else:
        primary_risk = "Overstock"
        value_at_stake = excess_inventory_value

    recommended_action = (
        assign_recommended_action(
            stockout_level=stockout_risk_level,
            overstock_level=overstock_risk_level,
        )
    )

    return {
        "sku_id": str(base_row["sku_id"]),
        "description": base_row.get("description"),
        "category": base_row.get("category"),
        "subcategory": base_row.get("subcategory"),
        "supplier": base_row.get("supplier"),
        "weekly_forecast_demand": weekly_forecast_demand,
        "lead_time_days": lead_time_days,
        "lead_time_weeks": lead_time_weeks,
        "ending_on_hand_units": ending_on_hand_units,
        "ending_on_order_units": ending_on_order_units,
        "inventory_position_units": inventory_position_units,
        "demand_rolling_std_13": demand_rolling_std_13,
        "safety_stock_units": safety_stock_units,
        "lead_time_forecast_units": (
            lead_time_forecast_units
        ),
        "lead_time_need_units": lead_time_need_units,
        "projected_lead_time_balance_units": (
            projected_balance_units
        ),
        "stockout_gap_units": stockout_gap_units,
        "stockout_risk_score": stockout_risk_score,
        "stockout_risk_level": stockout_risk_level,
        "forecast_horizon_units": (
            forecast_horizon_units
        ),
        "overstock_threshold_units": (
            overstock_threshold_units
        ),
        "excess_inventory_units": (
            excess_inventory_units
        ),
        "overstock_risk_score": (
            overstock_risk_score
        ),
        "overstock_risk_level": (
            overstock_risk_level
        ),
        "inventory_coverage_weeks": (
            inventory_coverage_weeks
        ),
        "minimum_order_quantity": (
            minimum_order_quantity
        ),
        "recommended_order_units": (
            recommended_order_units
        ),
        "unit_cost": unit_cost,
        "list_price": list_price,
        "recommended_order_cost": (
            recommended_order_cost
        ),
        "potential_lost_revenue": (
            potential_lost_revenue
        ),
        "excess_inventory_value": (
            excess_inventory_value
        ),
        "overall_risk_score": (
            overall_risk_score
        ),
        "primary_risk": primary_risk,
        "value_at_stake": value_at_stake,
        "recommended_action": (
            recommended_action
        ),
    }


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get(
    "/",
    tags=["General"],
)
def root() -> dict[str, Any]:
    """Return basic service information."""

    return {
        "service": "Project FORESIGHT Scoring API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get(
    "/health",
    tags=["General"],
)
def health_check() -> dict[str, Any]:
    """Check service and project-output availability."""

    files = {
        "inventory_risk_scores": {
            "path": str(
                RISK_SCORES_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
            "exists": RISK_SCORES_FILE.exists(),
        },
        "future_8_week_forecast": {
            "path": str(
                FUTURE_FORECAST_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
            "exists": FUTURE_FORECAST_FILE.exists(),
        },
        "inventory_risk_summary": {
            "path": str(
                RISK_SUMMARY_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
            "exists": RISK_SUMMARY_FILE.exists(),
        },
    }

    all_files_available = all(
        item["exists"]
        for item in files.values()
    )

    response: dict[str, Any] = {
        "status": (
            "healthy"
            if all_files_available
            else "degraded"
        ),
        "files": files,
    }

    if all_files_available:
        risk_scores = load_risk_scores()
        future_forecast = load_future_forecast()

        response.update(
            {
                "skus_scored": int(
                    risk_scores[
                        "sku_id"
                    ].nunique()
                ),
                "forecast_rows": int(
                    len(future_forecast)
                ),
                "forecast_skus": int(
                    future_forecast[
                        "sku_id"
                    ].nunique()
                ),
                "forecast_weeks": int(
                    future_forecast[
                        "forecast_week"
                    ].nunique()
                ),
                "forecast_start": (
                    future_forecast[
                        "forecast_week"
                    ]
                    .min()
                    .date()
                    .isoformat()
                ),
                "forecast_end": (
                    future_forecast[
                        "forecast_week"
                    ]
                    .max()
                    .date()
                    .isoformat()
                ),
            }
        )

    return clean_json_value(response)


@app.get(
    "/summary",
    tags=["Risk Intelligence"],
)
def get_summary() -> dict[str, Any]:
    """Return the executive inventory-risk summary."""

    summary = load_risk_summary()

    records = summary.to_dict(
        orient="records"
    )

    return {
        "record_count": len(records),
        "summary": clean_json_value(records),
    }


@app.get(
    "/skus",
    tags=["SKU Intelligence"],
)
def list_skus(
    search: str | None = Query(
        default=None,
        description=(
            "Optional SKU, description, category or "
            "supplier search text."
        ),
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> dict[str, Any]:
    """Return a searchable, paginated SKU list."""

    risk_scores = load_risk_scores()

    filtered = risk_scores.copy()

    if search:
        search_text = str(search).strip()

        searchable_columns = [
            column
            for column in [
                "sku_id",
                "description",
                "category",
                "subcategory",
                "supplier",
            ]
            if column in filtered.columns
        ]

        mask = pd.Series(
            False,
            index=filtered.index,
        )

        for column in searchable_columns:
            mask = mask | (
                filtered[column]
                .fillna("")
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    regex=False,
                )
            )

        filtered = filtered.loc[mask]

    display_columns = [
        column
        for column in [
            "priority_rank",
            "sku_id",
            "description",
            "category",
            "supplier",
            "stockout_risk_level",
            "overstock_risk_level",
            "weekly_forecast_demand",
            "recommended_order_units",
            "value_at_stake",
            "recommended_action",
        ]
        if column in filtered.columns
    ]

    page = filtered.iloc[
        offset:offset + limit
    ]

    return {
        "total_matches": int(len(filtered)),
        "offset": offset,
        "limit": limit,
        "items": clean_json_value(
            page[
                display_columns
            ].to_dict(
                orient="records"
            )
        ),
    }


@app.get(
    "/forecast/{sku_id}",
    tags=["Forecasting"],
)
def get_future_forecast(
    sku_id: str,
) -> dict[str, Any]:
    """Return the true 8-week future forecast for one SKU."""

    future_forecast = load_future_forecast()

    clean_sku_id = str(sku_id).strip()

    sku_forecast = (
        future_forecast.loc[
            future_forecast[
                "sku_id"
            ].eq(
                clean_sku_id
            )
        ]
        .sort_values(
            "forecast_week"
        )
    )

    if sku_forecast.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Future forecast not found for SKU: "
                f"{clean_sku_id}"
            ),
        )

    records = sku_forecast[
        [
            "sku_id",
            "forecast_week",
            "forecast_demand",
        ]
    ].to_dict(
        orient="records"
    )

    return {
        "sku_id": clean_sku_id,
        "forecast_weeks": int(
            sku_forecast[
                "forecast_week"
            ].nunique()
        ),
        "total_forecast_demand": float(
            sku_forecast[
                "forecast_demand"
            ].sum()
        ),
        "average_weekly_forecast": float(
            sku_forecast[
                "forecast_demand"
            ].mean()
        ),
        "forecast": clean_json_value(
            records
        ),
    }


@app.get(
    "/score/{sku_id}",
    tags=["SKU Intelligence"],
)
def get_existing_score(
    sku_id: str,
) -> dict[str, Any]:
    """Return the saved score and future forecast for one SKU."""

    risk_scores = load_risk_scores()
    future_forecast = load_future_forecast()

    row = find_sku_row(
        risk_scores,
        sku_id,
    )

    clean_sku_id = str(
        row["sku_id"]
    )

    sku_forecast = (
        future_forecast.loc[
            future_forecast[
                "sku_id"
            ].eq(
                clean_sku_id
            )
        ]
        .sort_values(
            "forecast_week"
        )
    )

    return {
        "saved_risk_score": clean_json_value(
            row.to_dict()
        ),
        "future_forecast": clean_json_value(
            sku_forecast[
                [
                    "forecast_week",
                    "forecast_demand",
                ]
            ].to_dict(
                orient="records"
            )
        ),
    }


@app.post(
    "/score",
    tags=["SKU Intelligence"],
)
def calculate_score(
    request: ScoreRequest,
) -> dict[str, Any]:
    """
    Recalculate one SKU's inventory risk.

    Values supplied in the request override the saved values. Missing
    values use the latest generated inventory-risk record.
    """

    risk_scores = load_risk_scores()

    row = find_sku_row(
        risk_scores,
        request.sku_id,
    )

    calculated_score = (
        calculate_custom_score(
            base_row=row,
            request=request,
        )
    )

    return clean_json_value(
        calculated_score
    )


@app.get(
    "/top-risks",
    tags=["Risk Intelligence"],
)
def get_top_risks(
    risk_type: Literal[
        "stockout",
        "overstock",
        "overall",
    ] = Query(
        default="overall",
    ),
    level: Literal[
        "All",
        "High",
        "Medium",
        "Low",
    ] = Query(
        default="All",
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=150,
    ),
) -> dict[str, Any]:
    """Return the highest-priority SKU risks."""

    risk_scores = load_risk_scores()

    if risk_type == "stockout":
        score_column = (
            "stockout_risk_score"
        )

        level_column = (
            "stockout_risk_level"
        )

        value_column = (
            "potential_lost_revenue"
        )

    elif risk_type == "overstock":
        score_column = (
            "overstock_risk_score"
        )

        level_column = (
            "overstock_risk_level"
        )

        value_column = (
            "excess_inventory_value"
        )

    else:
        score_column = (
            "overall_risk_score"
        )

        level_column = None
        value_column = (
            "value_at_stake"
        )

    filtered = risk_scores.copy()

    if (
        level != "All"
        and level_column is not None
    ):
        filtered = filtered.loc[
            filtered[
                level_column
            ].eq(
                level
            )
        ]

    filtered = (
        filtered
        .sort_values(
            [
                score_column,
                value_column,
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(limit)
    )

    display_columns = [
        column
        for column in [
            "priority_rank",
            "sku_id",
            "description",
            "category",
            "supplier",
            "stockout_risk_level",
            "stockout_risk_score",
            "overstock_risk_level",
            "overstock_risk_score",
            "overall_risk_score",
            "potential_lost_revenue",
            "excess_inventory_value",
            "value_at_stake",
            "recommended_order_units",
            "recommended_action",
        ]
        if column in filtered.columns
    ]

    return {
        "risk_type": risk_type,
        "risk_level": level,
        "record_count": int(
            len(filtered)
        ),
        "items": clean_json_value(
            filtered[
                display_columns
            ].to_dict(
                orient="records"
            )
        ),
    }