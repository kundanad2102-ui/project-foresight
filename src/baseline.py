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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "weekly_demand.csv"
)

PREDICTIONS_FILE = (
    PROCESSED_DATA_DIR
    / "baseline_predictions.csv"
)

METRICS_FILE = (
    REPORTS_DIR
    / "baseline_metrics.csv"
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TEST_WEEKS = 13

REQUIRED_COLUMNS = {
    "sku_id",
    "week_start",
    "week_end",
    "demand",
    "demand_lag_1",
    "demand_lag_52",
    "is_complete_week",
}


# ---------------------------------------------------------
# LOAD WEEKLY DATA
# ---------------------------------------------------------

def load_weekly_data() -> pd.DataFrame:
    """Load and validate weekly demand data."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    dataframe = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "week_start",
            "week_end",
        ],
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "weekly_demand.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_count = int(
        dataframe.duplicated(
            subset=["sku_id", "week_start"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "weekly_demand.csv contains duplicate "
            "SKU-week records."
        )

    dataframe["demand"] = pd.to_numeric(
        dataframe["demand"],
        errors="coerce",
    )

    dataframe["demand_lag_1"] = pd.to_numeric(
        dataframe["demand_lag_1"],
        errors="coerce",
    )

    dataframe["demand_lag_52"] = pd.to_numeric(
        dataframe["demand_lag_52"],
        errors="coerce",
    )

    dataframe = (
        dataframe
        .sort_values(["week_start", "sku_id"])
        .reset_index(drop=True)
    )

    print("Weekly dataset loaded:", dataframe.shape)
    print(
        "Unique SKUs:",
        dataframe["sku_id"].nunique(),
    )
    print(
        "Unique weeks:",
        dataframe["week_start"].nunique(),
    )

    return dataframe


# ---------------------------------------------------------
# CREATE TIME-BASED TEST SET
# ---------------------------------------------------------

def create_test_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Use the final 13 complete weeks as the test period."""

    complete_data = dataframe.loc[
        dataframe["is_complete_week"].eq(1)
    ].copy()

    complete_weeks = sorted(
        complete_data["week_start"].unique()
    )

    if len(complete_weeks) <= TEST_WEEKS:
        raise ValueError(
            "Not enough complete weeks to create "
            "the requested test period."
        )

    test_start = pd.Timestamp(
        complete_weeks[-TEST_WEEKS]
    )

    test_data = complete_data.loc[
        complete_data["week_start"] >= test_start
    ].copy()

    test_data = test_data.dropna(
        subset=[
            "demand",
            "demand_lag_1",
            "demand_lag_52",
        ]
    )

    if test_data.empty:
        raise ValueError(
            "The test dataset is empty after "
            "removing unavailable baseline predictions."
        )

    print("\nTIME-BASED TEST PERIOD")
    print("-" * 55)
    print("Test start:", test_data["week_start"].min())
    print("Test end:", test_data["week_end"].max())
    print(
        "Test weeks:",
        test_data["week_start"].nunique(),
    )
    print("Test rows:", len(test_data))
    print(
        "Test SKUs:",
        test_data["sku_id"].nunique(),
    )

    return test_data, test_start


# ---------------------------------------------------------
# FORECAST METRICS
# ---------------------------------------------------------

def calculate_metrics(
    actual: pd.Series,
    prediction: pd.Series,
    model_name: str,
) -> dict[str, float | str]:
    """Calculate WAPE, MAE, RMSE and forecast bias."""

    actual_values = actual.astype(float)
    prediction_values = prediction.astype(float)

    errors = prediction_values - actual_values
    absolute_errors = errors.abs()

    actual_total = actual_values.sum()

    if actual_total == 0:
        wape = np.nan
        bias_percentage = np.nan
    else:
        wape = (
            absolute_errors.sum()
            / actual_total
            * 100
        )

        bias_percentage = (
            errors.sum()
            / actual_total
            * 100
        )

    mae = absolute_errors.mean()

    rmse = np.sqrt(
        np.mean(
            np.square(errors)
        )
    )

    return {
        "model": model_name,
        "wape_percent": wape,
        "mae": mae,
        "rmse": rmse,
        "bias_percent": bias_percentage,
        "actual_total": actual_total,
        "forecast_total": prediction_values.sum(),
        "evaluation_rows": len(actual_values),
    }


# ---------------------------------------------------------
# CREATE BASELINE FORECASTS
# ---------------------------------------------------------

def evaluate_baselines(
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate last-week and seasonal-naive forecasts."""

    predictions = test_data[
        [
            "sku_id",
            "week_start",
            "week_end",
            "demand",
        ]
    ].copy()

    predictions = predictions.rename(
        columns={
            "demand": "actual_demand",
        }
    )

    predictions["naive_last_week"] = (
        test_data["demand_lag_1"]
        .clip(lower=0)
        .to_numpy()
    )

    predictions["seasonal_naive_52"] = (
        test_data["demand_lag_52"]
        .clip(lower=0)
        .to_numpy()
    )

    predictions["naive_absolute_error"] = (
        predictions["naive_last_week"]
        - predictions["actual_demand"]
    ).abs()

    predictions["seasonal_absolute_error"] = (
        predictions["seasonal_naive_52"]
        - predictions["actual_demand"]
    ).abs()

    metric_records = [
        calculate_metrics(
            predictions["actual_demand"],
            predictions["naive_last_week"],
            "Naive – Previous Week",
        ),
        calculate_metrics(
            predictions["actual_demand"],
            predictions["seasonal_naive_52"],
            "Seasonal Naive – 52 Weeks",
        ),
    ]

    metrics = pd.DataFrame(
        metric_records
    ).sort_values(
        "wape_percent",
        ascending=True,
    ).reset_index(drop=True)

    metrics["rank"] = (
        np.arange(len(metrics)) + 1
    )

    metrics["selected_baseline"] = (
        metrics["rank"].eq(1)
    )

    return predictions, metrics


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_results(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Save predictions and baseline metrics."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    print(f"\nSaved: {PREDICTIONS_FILE}")
    print(f"Saved: {METRICS_FILE}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    """Run Project FORESIGHT baseline evaluation."""

    print("=" * 65)
    print("PROJECT FORESIGHT – FORECAST BASELINE")
    print("=" * 65)

    weekly_data = load_weekly_data()

    test_data, test_start = create_test_dataset(
        weekly_data
    )

    predictions, metrics = evaluate_baselines(
        test_data
    )

    print("\nBASELINE METRICS")
    print("-" * 65)

    print(
        metrics[
            [
                "model",
                "wape_percent",
                "mae",
                "rmse",
                "bias_percent",
                "actual_total",
                "forecast_total",
                "evaluation_rows",
                "rank",
            ]
        ].to_string(index=False)
    )

    best_model = metrics.iloc[0]

    print("\nSELECTED BASELINE")
    print("-" * 65)
    print("Model:", best_model["model"])
    print(
        "WAPE:",
        f"{best_model['wape_percent']:.2f}%",
    )
    print(
        "MAE:",
        f"{best_model['mae']:.2f}",
    )
    print(
        "Bias:",
        f"{best_model['bias_percent']:.2f}%",
    )

    save_results(
        predictions,
        metrics,
    )

    print("\n" + "=" * 65)
    print("DAY 10 BASELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()