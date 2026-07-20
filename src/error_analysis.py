from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from src.model import (
    ALL_FEATURES,
    FINAL_MODEL_FILE,
    create_time_splits,
    load_modelling_data,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_predictions.csv"
)

SKU_MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sku_master.csv"
)

FEATURE_IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "reports"
    / "feature_importance.csv"
)

SKU_ERROR_FILE = (
    PROJECT_ROOT
    / "reports"
    / "sku_error_analysis.csv"
)

WEEK_ERROR_FILE = (
    PROJECT_ROOT
    / "reports"
    / "weekly_error_analysis.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "error_analysis_summary.md"
)

RANDOM_STATE = 42


# =========================================================
# METRIC
# =========================================================

def wape_metric(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate Weighted Absolute Percentage Error."""

    actual_values = np.asarray(
        actual,
        dtype=float,
    )

    prediction_values = np.clip(
        np.asarray(
            prediction,
            dtype=float,
        ),
        0,
        None,
    )

    actual_total = actual_values.sum()

    if actual_total == 0:
        return 0.0

    return float(
        np.abs(
            actual_values
            - prediction_values
        ).sum()
        / actual_total
        * 100
    )


WAPE_SCORER = make_scorer(
    wape_metric,
    greater_is_better=False,
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def calculate_feature_importance() -> pd.DataFrame:
    """Calculate raw-feature importance using permutation importance."""

    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Final model not found: {FINAL_MODEL_FILE}"
        )

    print("Loading final forecasting model...")

    final_model = joblib.load(
        FINAL_MODEL_FILE
    )

    modelling_data = load_modelling_data()

    (
        _,
        _,
        test_data,
    ) = create_time_splits(
        modelling_data
    )

    x_test = test_data[
        ALL_FEATURES
    ]

    y_test = test_data[
        "demand"
    ]

    print("\nCalculating permutation feature importance...")
    print("This may take a few minutes.")

    importance_result = permutation_importance(
        estimator=final_model,
        X=x_test,
        y=y_test,
        scoring=WAPE_SCORER,
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_data = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "importance_mean": (
                importance_result
                .importances_mean
            ),
            "importance_std": (
                importance_result
                .importances_std
            ),
        }
    )

    importance_data = (
        importance_data
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_data[
        "importance_rank"
    ] = (
        np.arange(
            len(importance_data)
        )
        + 1
    )

    importance_data[
        "importance_interpretation"
    ] = np.where(
        importance_data[
            "importance_mean"
        ] > 0,
        "Useful feature",
        "Little or negative contribution",
    )

    return importance_data


# =========================================================
# PREDICTION ERROR ANALYSIS
# =========================================================

def load_predictions() -> pd.DataFrame:
    """Load final test-period model predictions."""

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=[
            "week_start",
            "week_end",
        ],
    )

    required_columns = {
        "sku_id",
        "week_start",
        "week_end",
        "actual_demand",
        "model_prediction",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    predictions[
        "forecast_error"
    ] = (
        predictions["model_prediction"]
        - predictions["actual_demand"]
    )

    predictions[
        "absolute_error"
    ] = predictions[
        "forecast_error"
    ].abs()

    predictions[
        "squared_error"
    ] = np.square(
        predictions[
            "forecast_error"
        ]
    )

    predictions[
        "forecast_direction"
    ] = np.select(
        [
            predictions[
                "forecast_error"
            ] > 0,
            predictions[
                "forecast_error"
            ] < 0,
        ],
        [
            "Over-forecast",
            "Under-forecast",
        ],
        default="Exact",
    )

    return predictions


def calculate_sku_errors(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise forecasting performance for every SKU."""

    sku_errors = (
        predictions
        .groupby(
            "sku_id",
            as_index=False,
        )
        .agg(
            actual_total=(
                "actual_demand",
                "sum",
            ),
            forecast_total=(
                "model_prediction",
                "sum",
            ),
            absolute_error_total=(
                "absolute_error",
                "sum",
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean",
            ),
            mean_squared_error=(
                "squared_error",
                "mean",
            ),
            overforecast_weeks=(
                "forecast_direction",
                lambda series: int(
                    series.eq(
                        "Over-forecast"
                    ).sum()
                ),
            ),
            underforecast_weeks=(
                "forecast_direction",
                lambda series: int(
                    series.eq(
                        "Under-forecast"
                    ).sum()
                ),
            ),
            evaluation_weeks=(
                "week_start",
                "nunique",
            ),
        )
    )

    sku_errors[
        "rmse"
    ] = np.sqrt(
        sku_errors[
            "mean_squared_error"
        ]
    )

    sku_errors[
        "forecast_error_total"
    ] = (
        sku_errors[
            "forecast_total"
        ]
        - sku_errors[
            "actual_total"
        ]
    )

    sku_errors[
        "wape_percent"
    ] = np.where(
        sku_errors[
            "actual_total"
        ] > 0,
        (
            sku_errors[
                "absolute_error_total"
            ]
            / sku_errors[
                "actual_total"
            ]
            * 100
        ),
        np.nan,
    )

    sku_errors[
        "bias_percent"
    ] = np.where(
        sku_errors[
            "actual_total"
        ] > 0,
        (
            sku_errors[
                "forecast_error_total"
            ]
            / sku_errors[
                "actual_total"
            ]
            * 100
        ),
        np.nan,
    )

    sku_errors[
        "bias_direction"
    ] = np.select(
        [
            sku_errors[
                "bias_percent"
            ] > 5,
            sku_errors[
                "bias_percent"
            ] < -5,
        ],
        [
            "Over-forecasting",
            "Under-forecasting",
        ],
        default="Balanced",
    )

    if SKU_MASTER_FILE.exists():
        sku_master = pd.read_csv(
            SKU_MASTER_FILE
        )

        optional_columns = [
            column
            for column in [
                "sku_id",
                "product_name",
                "description",
                "category",
            ]
            if column in sku_master.columns
        ]

        if "sku_id" in optional_columns:
            sku_errors = sku_errors.merge(
                sku_master[
                    optional_columns
                ].drop_duplicates(
                    subset=["sku_id"]
                ),
                on="sku_id",
                how="left",
            )

    sku_errors = (
        sku_errors
        .sort_values(
            "absolute_error_total",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    sku_errors[
        "error_rank"
    ] = (
        np.arange(
            len(sku_errors)
        )
        + 1
    )

    return sku_errors


def calculate_weekly_errors(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise forecasting performance by week."""

    weekly_errors = (
        predictions
        .groupby(
            [
                "week_start",
                "week_end",
            ],
            as_index=False,
        )
        .agg(
            actual_total=(
                "actual_demand",
                "sum",
            ),
            forecast_total=(
                "model_prediction",
                "sum",
            ),
            absolute_error_total=(
                "absolute_error",
                "sum",
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean",
            ),
            mean_squared_error=(
                "squared_error",
                "mean",
            ),
            sku_count=(
                "sku_id",
                "nunique",
            ),
        )
    )

    weekly_errors[
        "rmse"
    ] = np.sqrt(
        weekly_errors[
            "mean_squared_error"
        ]
    )

    weekly_errors[
        "forecast_error_total"
    ] = (
        weekly_errors[
            "forecast_total"
        ]
        - weekly_errors[
            "actual_total"
        ]
    )

    weekly_errors[
        "wape_percent"
    ] = np.where(
        weekly_errors[
            "actual_total"
        ] > 0,
        (
            weekly_errors[
                "absolute_error_total"
            ]
            / weekly_errors[
                "actual_total"
            ]
            * 100
        ),
        np.nan,
    )

    weekly_errors[
        "bias_percent"
    ] = np.where(
        weekly_errors[
            "actual_total"
        ] > 0,
        (
            weekly_errors[
                "forecast_error_total"
            ]
            / weekly_errors[
                "actual_total"
            ]
            * 100
        ),
        np.nan,
    )

    weekly_errors = (
        weekly_errors
        .sort_values(
            "week_start"
        )
        .reset_index(drop=True)
    )

    return weekly_errors


# =========================================================
# SUMMARY REPORT
# =========================================================

def create_summary_report(
    feature_importance: pd.DataFrame,
    sku_errors: pd.DataFrame,
    weekly_errors: pd.DataFrame,
) -> None:
    """Create a readable Markdown summary."""

    top_features = feature_importance.head(
        10
    )

    worst_skus = sku_errors.head(
        10
    )

    worst_week = weekly_errors.sort_values(
        "wape_percent",
        ascending=False,
    ).iloc[0]

    summary_lines = [
        "# Project FORESIGHT — Model Error Analysis",
        "",
        "## Final Model",
        "",
        "- Selected model: HistGradientBoosting",
        "- Final test WAPE: 39.51%",
        "- Baseline WAPE: 47.88%",
        "- Improvement: 8.37 percentage points",
        "- Final model bias: approximately +2.94%",
        "",
        "## Top 10 Important Features",
        "",
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ]

    for _, row in top_features.iterrows():
        summary_lines.append(
            f"| {int(row['importance_rank'])} "
            f"| {row['feature']} "
            f"| {row['importance_mean']:.4f} |"
        )

    summary_lines.extend(
        [
            "",
            "Positive permutation importance means that "
            "model performance becomes worse when the feature "
            "is randomly shuffled. Therefore, larger positive "
            "values indicate more useful forecasting features.",
            "",
            "## Top 10 SKUs by Total Absolute Error",
            "",
            "| Rank | SKU | Actual | Forecast | WAPE | Bias |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )

    for _, row in worst_skus.iterrows():
        summary_lines.append(
            f"| {int(row['error_rank'])} "
            f"| {row['sku_id']} "
            f"| {row['actual_total']:.2f} "
            f"| {row['forecast_total']:.2f} "
            f"| {row['wape_percent']:.2f}% "
            f"| {row['bias_percent']:.2f}% |"
        )

    summary_lines.extend(
        [
            "",
            "## Worst Forecast Week",
            "",
            f"- Week start: {worst_week['week_start']}",
            f"- Actual demand: {worst_week['actual_total']:.2f}",
            f"- Forecast demand: {worst_week['forecast_total']:.2f}",
            f"- WAPE: {worst_week['wape_percent']:.2f}%",
            f"- Bias: {worst_week['bias_percent']:.2f}%",
            "",
            "## Interpretation",
            "",
            "The final model outperformed both naive baselines, "
            "but forecast accuracy differs across SKUs and weeks. "
            "High-error SKUs should be reviewed separately before "
            "inventory decisions are generated.",
        ]
    )

    SUMMARY_FILE.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(
    feature_importance: pd.DataFrame,
    sku_errors: pd.DataFrame,
    weekly_errors: pd.DataFrame,
) -> None:
    """Save all Day 14 analysis outputs."""

    FEATURE_IMPORTANCE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    sku_errors.to_csv(
        SKU_ERROR_FILE,
        index=False,
    )

    weekly_errors.to_csv(
        WEEK_ERROR_FILE,
        index=False,
    )

    create_summary_report(
        feature_importance,
        sku_errors,
        weekly_errors,
    )

    print("\nSAVED OUTPUTS")
    print("-" * 65)
    print(FEATURE_IMPORTANCE_FILE)
    print(SKU_ERROR_FILE)
    print(WEEK_ERROR_FILE)
    print(SUMMARY_FILE)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """Run Week 3 Day 14 model analysis."""

    print("=" * 65)
    print("PROJECT FORESIGHT - FEATURE AND ERROR ANALYSIS")
    print("=" * 65)

    feature_importance = (
        calculate_feature_importance()
    )

    predictions = load_predictions()

    sku_errors = calculate_sku_errors(
        predictions
    )

    weekly_errors = calculate_weekly_errors(
        predictions
    )

    print("\nTOP 10 IMPORTANT FEATURES")
    print("-" * 65)

    print(
        feature_importance[
            [
                "importance_rank",
                "feature",
                "importance_mean",
                "importance_std",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nTOP 10 SKUs BY ABSOLUTE ERROR")
    print("-" * 65)

    print(
        sku_errors[
            [
                "error_rank",
                "sku_id",
                "actual_total",
                "forecast_total",
                "wape_percent",
                "bias_percent",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    worst_week = weekly_errors.sort_values(
        "wape_percent",
        ascending=False,
    ).iloc[0]

    print("\nWORST FORECAST WEEK")
    print("-" * 65)
    print(
        "Week start:",
        worst_week["week_start"],
    )
    print(
        "WAPE:",
        f"{worst_week['wape_percent']:.2f}%",
    )
    print(
        "Bias:",
        f"{worst_week['bias_percent']:.2f}%",
    )

    save_outputs(
        feature_importance,
        sku_errors,
        weekly_errors,
    )

    print("\n" + "=" * 65)
    print("WEEK 3 DAY 14 COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()