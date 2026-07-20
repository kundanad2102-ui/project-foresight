from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE AND PROJECT SETTINGS
# =========================================================

st.set_page_config(
    page_title="Project FORESIGHT",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RISK_SCORES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_risk_scores.csv"
)

MODEL_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_predictions.csv"
)

MODEL_TEST_METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_test_metrics.csv"
)

ROLLING_CV_SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "rolling_origin_cv_summary.csv"
)

RISK_SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "inventory_risk_summary.csv"
)


# =========================================================
# DISPLAY HELPERS
# =========================================================

RISK_ORDER = [
    "High",
    "Medium",
    "Low",
]

RISK_COLORS = {
    "High": "#DC2626",
    "Medium": "#F59E0B",
    "Low": "#16A34A",
}

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        .foresight-header {
            padding: 1.1rem 1.3rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 16px;
            margin-bottom: 1rem;
            background:
                linear-gradient(
                    135deg,
                    rgba(15, 23, 42, 0.96),
                    rgba(30, 64, 175, 0.92)
                );
            color: white;
        }

        .foresight-header h1 {
            margin: 0;
            font-size: 2rem;
        }

        .foresight-header p {
            margin: 0.35rem 0 0 0;
            opacity: 0.9;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 14px;
            padding: 0.8rem;
            background: rgba(148, 163, 184, 0.06);
        }

        .small-note {
            color: #64748B;
            font-size: 0.88rem;
        }

        .risk-high {
            color: #DC2626;
            font-weight: 700;
        }

        .risk-medium {
            color: #D97706;
            font-weight: 700;
        }

        .risk-low {
            color: #16A34A;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_number(
    value: float | int,
    decimals: int = 0,
) -> str:
    """Format a number with thousands separators."""

    if pd.isna(value):
        return "—"

    return f"{float(value):,.{decimals}f}"


def format_percent(
    value: float | int,
    decimals: int = 2,
) -> str:
    """Format a numeric percentage."""

    if pd.isna(value):
        return "—"

    return f"{float(value):,.{decimals}f}%"


def safe_sum(
    dataframe: pd.DataFrame,
    column: str,
) -> float:
    """Return a numeric column sum, or zero when unavailable."""

    if column not in dataframe.columns:
        return 0.0

    return float(
        pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )


def safe_count(
    dataframe: pd.DataFrame,
    column: str,
    value: str,
) -> int:
    """Count rows matching a value, or return zero."""

    if column not in dataframe.columns:
        return 0

    return int(
        dataframe[column]
        .astype(str)
        .eq(value)
        .sum()
    )


def dataframe_to_csv(
    dataframe: pd.DataFrame,
) -> bytes:
    """Convert a dataframe to downloadable UTF-8 CSV bytes."""

    return dataframe.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data(
    show_spinner=False
)
def load_csv(
    filepath: Path,
    date_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Load a CSV and parse only date columns that exist."""

    header = pd.read_csv(
        filepath,
        nrows=0,
    )

    available_date_columns = [
        column
        for column in date_columns
        if column in header.columns
    ]

    return pd.read_csv(
        filepath,
        parse_dates=available_date_columns,
    )


def validate_required_files() -> None:
    """Stop the app with a clear message when files are missing."""

    required_files = [
        RISK_SCORES_FILE,
        MODEL_PREDICTIONS_FILE,
        MODEL_TEST_METRICS_FILE,
        ROLLING_CV_SUMMARY_FILE,
        RISK_SUMMARY_FILE,
    ]

    missing_files = [
        filepath
        for filepath in required_files
        if not filepath.exists()
    ]

    if missing_files:
        st.error(
            "The dashboard cannot start because required "
            "project output files are missing."
        )

        st.code(
            "\n".join(
                str(filepath)
                for filepath in missing_files
            )
        )

        st.info(
            "Run the model, rolling cross-validation, and "
            "risk-scoring workflows before launching the dashboard."
        )

        st.stop()


validate_required_files()

risk_scores = load_csv(
    RISK_SCORES_FILE,
    date_columns=(
        "week_start",
        "week_end",
    ),
)

predictions = load_csv(
    MODEL_PREDICTIONS_FILE,
    date_columns=(
        "week_start",
        "week_end",
    ),
)

test_metrics = load_csv(
    MODEL_TEST_METRICS_FILE,
)

rolling_cv = load_csv(
    ROLLING_CV_SUMMARY_FILE,
)

risk_summary = load_csv(
    RISK_SUMMARY_FILE,
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title(
    "Dashboard Filters"
)

available_categories = sorted(
    risk_scores[
        "category"
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
) if "category" in risk_scores.columns else []

selected_categories = st.sidebar.multiselect(
    "Category",
    options=available_categories,
    default=available_categories,
)

available_suppliers = sorted(
    risk_scores[
        "supplier"
    ]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
) if "supplier" in risk_scores.columns else []

selected_suppliers = st.sidebar.multiselect(
    "Supplier",
    options=available_suppliers,
    default=available_suppliers,
)

selected_stockout_levels = st.sidebar.multiselect(
    "Stockout risk",
    options=RISK_ORDER,
    default=RISK_ORDER,
)

selected_overstock_levels = st.sidebar.multiselect(
    "Overstock risk",
    options=RISK_ORDER,
    default=RISK_ORDER,
)

minimum_priority = st.sidebar.number_input(
    "Maximum priority rank",
    min_value=1,
    max_value=max(
        1,
        len(risk_scores),
    ),
    value=min(
        150,
        max(
            1,
            len(risk_scores),
        ),
    ),
    step=1,
)

filtered_risk = risk_scores.copy()

if selected_categories:
    filtered_risk = filtered_risk.loc[
        filtered_risk[
            "category"
        ]
        .astype(str)
        .isin(
            selected_categories
        )
    ]

if selected_suppliers:
    filtered_risk = filtered_risk.loc[
        filtered_risk[
            "supplier"
        ]
        .astype(str)
        .isin(
            selected_suppliers
        )
    ]

if (
    "stockout_risk_level"
    in filtered_risk.columns
    and selected_stockout_levels
):
    filtered_risk = filtered_risk.loc[
        filtered_risk[
            "stockout_risk_level"
        ]
        .astype(str)
        .isin(
            selected_stockout_levels
        )
    ]

if (
    "overstock_risk_level"
    in filtered_risk.columns
    and selected_overstock_levels
):
    filtered_risk = filtered_risk.loc[
        filtered_risk[
            "overstock_risk_level"
        ]
        .astype(str)
        .isin(
            selected_overstock_levels
        )
    ]

if "priority_rank" in filtered_risk.columns:
    filtered_risk = filtered_risk.loc[
        pd.to_numeric(
            filtered_risk[
                "priority_rank"
            ],
            errors="coerce",
        )
        <= minimum_priority
    ]

filtered_risk = filtered_risk.reset_index(
    drop=True
)

st.sidebar.markdown("---")

st.sidebar.download_button(
    label="Download filtered risk data",
    data=dataframe_to_csv(
        filtered_risk
    ),
    file_name=(
        "project_foresight_filtered_risk_scores.csv"
    ),
    mime="text/csv",
    use_container_width=True,
)

st.sidebar.caption(
    "Monetary values are shown in the dataset's "
    "original currency units."
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="foresight-header">
        <h1>Project FORESIGHT</h1>
        <p>
            SKU-level demand forecasting, stockout-risk,
            overstock-risk, and inventory action intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

latest_scoring_week = (
    pd.to_datetime(
        risk_scores[
            "week_start"
        ],
        errors="coerce",
    ).max()
    if "week_start" in risk_scores.columns
    else pd.NaT
)

header_left, header_right = st.columns(
    [
        3,
        1,
    ]
)

with header_left:
    st.caption(
        f"Showing {len(filtered_risk):,} of "
        f"{len(risk_scores):,} scored SKUs"
    )

with header_right:
    if not pd.isna(
        latest_scoring_week
    ):
        st.caption(
            "Scoring week: "
            f"{latest_scoring_week.date()}"
        )


# =========================================================
# DASHBOARD TABS
# =========================================================

(
    overview_tab,
    priorities_tab,
    forecast_tab,
    sku_tab,
    methodology_tab,
) = st.tabs(
    [
        "Executive Overview",
        "Risk Priorities",
        "Forecast Performance",
        "SKU Explorer",
        "Methodology",
    ]
)


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

with overview_tab:
    high_stockout = safe_count(
        filtered_risk,
        "stockout_risk_level",
        "High",
    )

    medium_stockout = safe_count(
        filtered_risk,
        "stockout_risk_level",
        "Medium",
    )

    high_overstock = safe_count(
        filtered_risk,
        "overstock_risk_level",
        "High",
    )

    potential_lost_revenue = safe_sum(
        filtered_risk,
        "potential_lost_revenue",
    )

    excess_inventory_value = safe_sum(
        filtered_risk,
        "excess_inventory_value",
    )

    recommended_order_cost = safe_sum(
        filtered_risk,
        "recommended_order_cost",
    )

    (
        kpi_1,
        kpi_2,
        kpi_3,
        kpi_4,
        kpi_5,
        kpi_6,
    ) = st.columns(6)

    kpi_1.metric(
        "SKUs shown",
        format_number(
            len(filtered_risk)
        ),
    )

    kpi_2.metric(
        "High stockout",
        format_number(
            high_stockout
        ),
    )

    kpi_3.metric(
        "Medium stockout",
        format_number(
            medium_stockout
        ),
    )

    kpi_4.metric(
        "High overstock",
        format_number(
            high_overstock
        ),
    )

    kpi_5.metric(
        "Potential lost revenue",
        format_number(
            potential_lost_revenue,
            2,
        ),
    )

    kpi_6.metric(
        "Excess inventory value",
        format_number(
            excess_inventory_value,
            2,
        ),
    )

    st.caption(
        "Recommended replenishment cost for the current "
        f"filtered view: {format_number(recommended_order_cost, 2)}"
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        if (
            "stockout_risk_level"
            in filtered_risk.columns
            and not filtered_risk.empty
        ):
            stockout_distribution = (
                filtered_risk[
                    "stockout_risk_level"
                ]
                .value_counts()
                .reindex(
                    RISK_ORDER,
                    fill_value=0,
                )
                .rename_axis(
                    "risk_level"
                )
                .reset_index(
                    name="sku_count"
                )
            )

            stockout_fig = px.bar(
                stockout_distribution,
                x="risk_level",
                y="sku_count",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                title="Stockout-Risk Distribution",
                category_orders={
                    "risk_level": RISK_ORDER
                },
                labels={
                    "risk_level": "Risk Level",
                    "sku_count": "Number of SKUs",
                },
                text="sku_count",
            )

            stockout_fig.update_layout(
                showlegend=False,
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10,
                ),
            )

            st.plotly_chart(
                stockout_fig,
                use_container_width=True,
            )

    with chart_right:
        if (
            "overstock_risk_level"
            in filtered_risk.columns
            and not filtered_risk.empty
        ):
            overstock_distribution = (
                filtered_risk[
                    "overstock_risk_level"
                ]
                .value_counts()
                .reindex(
                    RISK_ORDER,
                    fill_value=0,
                )
                .rename_axis(
                    "risk_level"
                )
                .reset_index(
                    name="sku_count"
                )
            )

            overstock_fig = px.bar(
                overstock_distribution,
                x="risk_level",
                y="sku_count",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                title="Overstock-Risk Distribution",
                category_orders={
                    "risk_level": RISK_ORDER
                },
                labels={
                    "risk_level": "Risk Level",
                    "sku_count": "Number of SKUs",
                },
                text="sku_count",
            )

            overstock_fig.update_layout(
                showlegend=False,
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10,
                ),
            )

            st.plotly_chart(
                overstock_fig,
                use_container_width=True,
            )

    if (
        "category"
        in filtered_risk.columns
        and not filtered_risk.empty
    ):
        category_value = (
            filtered_risk
            .groupby(
                "category",
                as_index=False,
            )
            .agg(
                potential_lost_revenue=(
                    "potential_lost_revenue",
                    "sum",
                ),
                excess_inventory_value=(
                    "excess_inventory_value",
                    "sum",
                ),
                recommended_order_cost=(
                    "recommended_order_cost",
                    "sum",
                ),
            )
        )

        category_long = category_value.melt(
            id_vars="category",
            value_vars=[
                "potential_lost_revenue",
                "excess_inventory_value",
                "recommended_order_cost",
            ],
            var_name="metric",
            value_name="value",
        )

        metric_labels = {
            "potential_lost_revenue": (
                "Potential Lost Revenue"
            ),
            "excess_inventory_value": (
                "Excess Inventory Value"
            ),
            "recommended_order_cost": (
                "Recommended Order Cost"
            ),
        }

        category_long["metric"] = (
            category_long[
                "metric"
            ].map(
                metric_labels
            )
        )

        category_fig = px.bar(
            category_long,
            x="category",
            y="value",
            color="metric",
            barmode="group",
            title="Value Exposure by Category",
            labels={
                "category": "Category",
                "value": "Value",
                "metric": "Metric",
            },
        )

        category_fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            ),
        )

        st.plotly_chart(
            category_fig,
            use_container_width=True,
        )


# =========================================================
# RISK PRIORITIES
# =========================================================

with priorities_tab:
    st.subheader(
        "Stockout Priorities"
    )

    stockout_priority_columns = [
        "priority_rank",
        "sku_id",
        "description",
        "category",
        "supplier",
        "stockout_risk_level",
        "stockout_risk_score",
        "stockout_gap_units",
        "potential_lost_revenue",
        "recommended_order_units",
        "recommended_order_cost",
        "recommended_action",
    ]

    available_stockout_columns = [
        column
        for column in stockout_priority_columns
        if column in filtered_risk.columns
    ]

    stockout_priority = (
        filtered_risk.loc[
            filtered_risk[
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
    )

    stockout_chart_column, stockout_table_column = st.columns(
        [
            1,
            1.5,
        ]
    )

    with stockout_chart_column:
        if not stockout_priority.empty:
            top_stockout = (
                stockout_priority
                .head(12)
                .sort_values(
                    "potential_lost_revenue",
                    ascending=True,
                )
            )

            top_stockout_fig = px.bar(
                top_stockout,
                x="potential_lost_revenue",
                y="sku_id",
                orientation="h",
                color="stockout_risk_level",
                color_discrete_map=RISK_COLORS,
                title="Top Stockout Value at Stake",
                labels={
                    "potential_lost_revenue": (
                        "Potential Lost Revenue"
                    ),
                    "sku_id": "SKU",
                    "stockout_risk_level": (
                        "Risk Level"
                    ),
                },
                hover_data=[
                    "stockout_gap_units",
                    "recommended_order_units",
                ],
            )

            top_stockout_fig.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10,
                ),
            )

            st.plotly_chart(
                top_stockout_fig,
                use_container_width=True,
            )
        else:
            st.info(
                "No stockout-risk SKUs match the current filters."
            )

    with stockout_table_column:
        st.dataframe(
            stockout_priority[
                available_stockout_columns
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "stockout_risk_score": st.column_config.NumberColumn(
                    "Stockout score",
                    format="%.2f",
                ),
                "stockout_gap_units": st.column_config.NumberColumn(
                    "Gap units",
                    format="%.2f",
                ),
                "potential_lost_revenue": st.column_config.NumberColumn(
                    "Potential lost revenue",
                    format="%.2f",
                ),
                "recommended_order_units": st.column_config.NumberColumn(
                    "Recommended units",
                    format="%.0f",
                ),
                "recommended_order_cost": st.column_config.NumberColumn(
                    "Order cost",
                    format="%.2f",
                ),
            },
        )

    st.download_button(
        "Download stockout priorities",
        data=dataframe_to_csv(
            stockout_priority[
                available_stockout_columns
            ]
        ),
        file_name=(
            "project_foresight_stockout_priorities.csv"
        ),
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader(
        "Overstock Priorities"
    )

    overstock_priority_columns = [
        "priority_rank",
        "sku_id",
        "description",
        "category",
        "supplier",
        "overstock_risk_level",
        "overstock_risk_score",
        "inventory_coverage_weeks",
        "excess_inventory_units",
        "excess_inventory_value",
        "recommended_action",
    ]

    available_overstock_columns = [
        column
        for column in overstock_priority_columns
        if column in filtered_risk.columns
    ]

    overstock_priority = (
        filtered_risk.loc[
            filtered_risk[
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
    )

    overstock_chart_column, overstock_table_column = st.columns(
        [
            1,
            1.5,
        ]
    )

    with overstock_chart_column:
        if not overstock_priority.empty:
            top_overstock = (
                overstock_priority
                .head(12)
                .sort_values(
                    "excess_inventory_value",
                    ascending=True,
                )
            )

            top_overstock_fig = px.bar(
                top_overstock,
                x="excess_inventory_value",
                y="sku_id",
                orientation="h",
                color="overstock_risk_level",
                color_discrete_map=RISK_COLORS,
                title="Top Overstock Value Exposure",
                labels={
                    "excess_inventory_value": (
                        "Excess Inventory Value"
                    ),
                    "sku_id": "SKU",
                    "overstock_risk_level": (
                        "Risk Level"
                    ),
                },
                hover_data=[
                    "excess_inventory_units",
                    "inventory_coverage_weeks",
                ],
            )

            top_overstock_fig.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10,
                ),
            )

            st.plotly_chart(
                top_overstock_fig,
                use_container_width=True,
            )
        else:
            st.info(
                "No overstock-risk SKUs match the current filters."
            )

    with overstock_table_column:
        st.dataframe(
            overstock_priority[
                available_overstock_columns
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "overstock_risk_score": st.column_config.NumberColumn(
                    "Overstock score",
                    format="%.2f",
                ),
                "inventory_coverage_weeks": st.column_config.NumberColumn(
                    "Coverage weeks",
                    format="%.2f",
                ),
                "excess_inventory_units": st.column_config.NumberColumn(
                    "Excess units",
                    format="%.2f",
                ),
                "excess_inventory_value": st.column_config.NumberColumn(
                    "Excess value",
                    format="%.2f",
                ),
            },
        )

    st.download_button(
        "Download overstock priorities",
        data=dataframe_to_csv(
            overstock_priority[
                available_overstock_columns
            ]
        ),
        file_name=(
            "project_foresight_overstock_priorities.csv"
        ),
        mime="text/csv",
    )


# =========================================================
# FORECAST PERFORMANCE
# =========================================================

with forecast_tab:
    st.subheader(
        "Final-Test Forecast Performance"
    )

    if not test_metrics.empty:
        metric_columns = [
            column
            for column in [
                "model",
                "wape_percent",
                "mae",
                "rmse",
                "bias_percent",
                "actual_total",
                "forecast_total",
                "test_rank",
            ]
            if column in test_metrics.columns
        ]

        st.dataframe(
            test_metrics[
                metric_columns
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "wape_percent": st.column_config.NumberColumn(
                    "WAPE",
                    format="%.2f%%",
                ),
                "mae": st.column_config.NumberColumn(
                    "MAE",
                    format="%.2f",
                ),
                "rmse": st.column_config.NumberColumn(
                    "RMSE",
                    format="%.2f",
                ),
                "bias_percent": st.column_config.NumberColumn(
                    "Bias",
                    format="%.2f%%",
                ),
            },
        )

        if (
            "model"
            in test_metrics.columns
            and "wape_percent"
            in test_metrics.columns
        ):
            test_metric_fig = px.bar(
                test_metrics.sort_values(
                    "wape_percent",
                    ascending=False,
                ),
                x="wape_percent",
                y="model",
                orientation="h",
                title="Final-Test WAPE by Model",
                labels={
                    "wape_percent": "WAPE (%)",
                    "model": "Model",
                },
                text_auto=".2f",
            )

            test_metric_fig.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10,
                ),
            )

            st.plotly_chart(
                test_metric_fig,
                use_container_width=True,
            )

    st.subheader(
        "Rolling-Origin Cross-Validation"
    )

    if not rolling_cv.empty:
        rolling_columns = [
            column
            for column in [
                "model",
                "wape_percent",
                "mae",
                "rmse",
                "bias_percent",
                "mean_fold_wape_percent",
                "std_fold_wape_percent",
                "overall_rank",
            ]
            if column in rolling_cv.columns
        ]

        st.dataframe(
            rolling_cv[
                rolling_columns
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "wape_percent": st.column_config.NumberColumn(
                    "Combined WAPE",
                    format="%.2f%%",
                ),
                "mean_fold_wape_percent": st.column_config.NumberColumn(
                    "Mean fold WAPE",
                    format="%.2f%%",
                ),
                "std_fold_wape_percent": st.column_config.NumberColumn(
                    "Fold WAPE standard deviation",
                    format="%.2f",
                ),
                "bias_percent": st.column_config.NumberColumn(
                    "Bias",
                    format="%.2f%%",
                ),
            },
        )

    st.subheader(
        "Actual Demand vs Forecast"
    )

    if (
        "week_start"
        in predictions.columns
        and "actual_demand"
        in predictions.columns
        and "model_prediction"
        in predictions.columns
    ):
        weekly_forecast = (
            predictions
            .groupby(
                "week_start",
                as_index=False,
            )
            .agg(
                actual_demand=(
                    "actual_demand",
                    "sum",
                ),
                model_prediction=(
                    "model_prediction",
                    "sum",
                ),
                naive_previous_week=(
                    "naive_previous_week",
                    "sum",
                ),
                seasonal_naive_52=(
                    "seasonal_naive_52",
                    "sum",
                ),
            )
            .sort_values(
                "week_start"
            )
        )

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=weekly_forecast[
                    "week_start"
                ],
                y=weekly_forecast[
                    "actual_demand"
                ],
                mode="lines+markers",
                name="Actual Demand",
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=weekly_forecast[
                    "week_start"
                ],
                y=weekly_forecast[
                    "model_prediction"
                ],
                mode="lines+markers",
                name="Selected Model",
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=weekly_forecast[
                    "week_start"
                ],
                y=weekly_forecast[
                    "naive_previous_week"
                ],
                mode="lines",
                name="Previous-Week Naive",
                line=dict(
                    dash="dot"
                ),
            )
        )

        forecast_fig.update_layout(
            title="Weekly Total Demand: Actual vs Forecast",
            xaxis_title="Week",
            yaxis_title="Demand Units",
            hovermode="x unified",
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            ),
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True,
        )


# =========================================================
# SKU EXPLORER
# =========================================================

with sku_tab:
    sku_options = sorted(
        risk_scores[
            "sku_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not sku_options:
        st.warning(
            "No SKU identifiers are available."
        )
    else:
        selected_sku = st.selectbox(
            "Select an SKU",
            options=sku_options,
        )

        sku_risk = risk_scores.loc[
            risk_scores[
                "sku_id"
            ]
            .astype(str)
            .eq(
                selected_sku
            )
        ]

        sku_predictions = predictions.loc[
            predictions[
                "sku_id"
            ]
            .astype(str)
            .eq(
                selected_sku
            )
        ].sort_values(
            "week_start"
        )

        if not sku_risk.empty:
            current_sku = sku_risk.iloc[0]

            (
                sku_kpi_1,
                sku_kpi_2,
                sku_kpi_3,
                sku_kpi_4,
                sku_kpi_5,
            ) = st.columns(5)

            sku_kpi_1.metric(
                "Stockout risk",
                str(
                    current_sku.get(
                        "stockout_risk_level",
                        "—",
                    )
                ),
            )

            sku_kpi_2.metric(
                "Overstock risk",
                str(
                    current_sku.get(
                        "overstock_risk_level",
                        "—",
                    )
                ),
            )

            sku_kpi_3.metric(
                "Forecast demand/week",
                format_number(
                    current_sku.get(
                        "weekly_forecast_demand",
                        np.nan,
                    ),
                    2,
                ),
            )

            sku_kpi_4.metric(
                "Recommended order",
                format_number(
                    current_sku.get(
                        "recommended_order_units",
                        np.nan,
                    )
                ),
            )

            sku_kpi_5.metric(
                "Value at stake",
                format_number(
                    current_sku.get(
                        "value_at_stake",
                        np.nan,
                    ),
                    2,
                ),
            )

            st.info(
                str(
                    current_sku.get(
                        "recommended_action",
                        "No action is available.",
                    )
                )
            )

            detail_columns = [
                column
                for column in [
                    "sku_id",
                    "description",
                    "category",
                    "subcategory",
                    "supplier",
                    "lead_time_days",
                    "ending_on_hand_units",
                    "ending_on_order_units",
                    "inventory_position_units",
                    "inventory_coverage_weeks",
                    "stockout_gap_units",
                    "excess_inventory_units",
                    "potential_lost_revenue",
                    "excess_inventory_value",
                    "recommended_order_cost",
                ]
                if column in sku_risk.columns
            ]

            st.dataframe(
                sku_risk[
                    detail_columns
                ],
                use_container_width=True,
                hide_index=True,
            )

        if not sku_predictions.empty:
            sku_forecast_fig = go.Figure()

            sku_forecast_fig.add_trace(
                go.Scatter(
                    x=sku_predictions[
                        "week_start"
                    ],
                    y=sku_predictions[
                        "actual_demand"
                    ],
                    mode="lines+markers",
                    name="Actual Demand",
                )
            )

            sku_forecast_fig.add_trace(
                go.Scatter(
                    x=sku_predictions[
                        "week_start"
                    ],
                    y=sku_predictions[
                        "model_prediction"
                    ],
                    mode="lines+markers",
                    name="Model Forecast",
                )
            )

            sku_forecast_fig.add_trace(
                go.Scatter(
                    x=sku_predictions[
                        "week_start"
                    ],
                    y=sku_predictions[
                        "naive_previous_week"
                    ],
                    mode="lines",
                    name="Previous-Week Naive",
                    line=dict(
                        dash="dot"
                    ),
                )
            )

            sku_forecast_fig.update_layout(
                title=f"{selected_sku}: Actual Demand vs Forecast",
                xaxis_title="Week",
                yaxis_title="Demand Units",
                hovermode="x unified",
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10,
                ),
            )

            st.plotly_chart(
                sku_forecast_fig,
                use_container_width=True,
            )

            sku_actual = pd.to_numeric(
                sku_predictions[
                    "actual_demand"
                ],
                errors="coerce",
            ).fillna(0)

            sku_model = pd.to_numeric(
                sku_predictions[
                    "model_prediction"
                ],
                errors="coerce",
            ).fillna(0)

            sku_absolute_error = (
                sku_model
                - sku_actual
            ).abs()

            sku_wape = (
                sku_absolute_error.sum()
                / sku_actual.sum()
                * 100
                if sku_actual.sum() > 0
                else np.nan
            )

            sku_mae = float(
                sku_absolute_error.mean()
            )

            metric_col_1, metric_col_2 = st.columns(2)

            metric_col_1.metric(
                "SKU test WAPE",
                format_percent(
                    sku_wape
                ),
            )

            metric_col_2.metric(
                "SKU test MAE",
                format_number(
                    sku_mae,
                    2,
                ),
            )
        else:
            st.info(
                "No final-test forecast history is available "
                "for the selected SKU."
            )


# =========================================================
# METHODOLOGY AND LIMITATIONS
# =========================================================

with methodology_tab:
    st.subheader(
        "Forecasting and Validation"
    )

    st.markdown(
        """
        - Weekly SKU-level demand is the forecasting target.
        - Leakage-safe lag, rolling, calendar, promotion, and
          previous-week inventory features are used.
        - HistGradientBoosting and Random Forest are compared.
        - The selected model is tested against previous-week and
          52-week seasonal-naive baselines.
        - Rolling-origin cross-validation uses expanding historical
          windows and an 8-week forecast horizon.
        - The final test period remains untouched during model
          selection.
        """
    )

    st.subheader(
        "Inventory-Risk Logic"
    )

    st.markdown(
        """
        - Inventory position equals ending on-hand units plus
          ending on-order units.
        - Stockout risk compares inventory position with lead-time
          forecast demand plus safety stock.
        - High stockout risk represents a shortage score of at least
          50%; Medium represents a smaller projected shortage.
        - Overstock risk compares inventory position with the
          8-week planning requirement plus safety stock.
        - Replenishment recommendations are rounded to the SKU
          minimum order quantity.
        - Value-at-stake measures potential lost revenue for
          stockouts and inventory carrying value for overstock.
        """
    )

    st.subheader(
        "Important Limitation"
    )

    st.warning(
        "The current risk-scoring demonstration uses the latest "
        "available historical test-period model predictions as an "
        "operational weekly-demand proxy. A deployed production "
        "workflow should replace this proxy with a newly generated "
        "8-week future forecast on every scoring run."
    )

    st.subheader(
        "Source Files"
    )

    source_files = pd.DataFrame(
        {
            "Purpose": [
                "SKU risk scores",
                "Forecast history",
                "Final-test model metrics",
                "Rolling-CV metrics",
                "Executive risk summary",
            ],
            "File": [
                str(
                    RISK_SCORES_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),
                str(
                    MODEL_PREDICTIONS_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),
                str(
                    MODEL_TEST_METRICS_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),
                str(
                    ROLLING_CV_SUMMARY_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),
                str(
                    RISK_SUMMARY_FILE.relative_to(
                        PROJECT_ROOT
                    )
                ),
            ],
        }
    )

    st.dataframe(
        source_files,
        use_container_width=True,
        hide_index=True,
    )