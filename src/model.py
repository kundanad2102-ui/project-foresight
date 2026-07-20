from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_predictions.csv"
)

VALIDATION_METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_validation_metrics.csv"
)

TEST_METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_test_metrics.csv"
)

MODEL_SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_summary.json"
)

FINAL_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_forecast_model.joblib"
)


# =========================================================
# MODELLING SETTINGS
# =========================================================

TARGET_COLUMN = "demand"

VALIDATION_WEEKS = 13
TEST_WEEKS = 13
RANDOM_STATE = 42


CATEGORICAL_FEATURES = [
    "sku_id",
]


NUMERIC_FEATURES = [
    # Calendar features known before the forecast week
    "iso_year",
    "iso_week",
    "month",
    "quarter",
    "iso_week_sin",
    "iso_week_cos",
    "month_sin",
    "month_cos",

    # Promotion schedule
    "promo_week_flag",

    # Historical demand lags
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_4",
    "demand_lag_8",
    "demand_lag_13",
    "demand_lag_26",
    "demand_lag_52",

    # Historical rolling statistics
    "demand_rolling_mean_4",
    "demand_rolling_std_4",
    "demand_rolling_mean_8",
    "demand_rolling_std_8",
    "demand_rolling_mean_13",
    "demand_rolling_std_13",
    "zero_demand_rate_4",

    # Previous-week inventory information
    "ending_on_hand_units_lag_1",
    "ending_on_order_units_lag_1",
    "stockout_days_lag_1",
    "lost_sales_lag_1",
    "promo_week_flag_lag_1",
]


ALL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


# =========================================================
# LOAD AND PREPARE DATA
# =========================================================

def load_modelling_data() -> pd.DataFrame:
    """Load weekly demand data and create safe model features."""

    if not WEEKLY_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Weekly data file not found: {WEEKLY_DATA_FILE}"
        )

    dataframe = pd.read_csv(
        WEEKLY_DATA_FILE,
        parse_dates=[
            "week_start",
            "week_end",
        ],
    )

    required_columns = {
        "sku_id",
        "week_start",
        "week_end",
        "demand",
        "is_complete_week",
        *[
            column
            for column in ALL_FEATURES
            if column not in {
                "iso_week_sin",
                "iso_week_cos",
                "month_sin",
                "month_cos",
            }
        ],
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.loc[
        dataframe["is_complete_week"].eq(1)
    ].copy()

    dataframe["iso_week_sin"] = np.sin(
        2
        * np.pi
        * dataframe["iso_week"]
        / 52
    )

    dataframe["iso_week_cos"] = np.cos(
        2
        * np.pi
        * dataframe["iso_week"]
        / 52
    )

    dataframe["month_sin"] = np.sin(
        2
        * np.pi
        * dataframe["month"]
        / 12
    )

    dataframe["month_cos"] = np.cos(
        2
        * np.pi
        * dataframe["month"]
        / 12
    )

    dataframe["demand"] = pd.to_numeric(
        dataframe["demand"],
        errors="coerce",
    )

    duplicate_count = int(
        dataframe.duplicated(
            subset=[
                "sku_id",
                "week_start",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "Duplicate SKU-week records were found."
        )

    if dataframe["demand"].isna().any():
        raise ValueError(
            "The target demand column contains missing values."
        )

    if dataframe["demand"].lt(0).any():
        raise ValueError(
            "The target demand column contains negative values."
        )

    dataframe = (
        dataframe
        .sort_values(
            [
                "week_start",
                "sku_id",
            ]
        )
        .reset_index(drop=True)
    )

    print("Complete-week dataset:", dataframe.shape)
    print(
        "Unique SKUs:",
        dataframe["sku_id"].nunique(),
    )
    print(
        "Complete weeks:",
        dataframe["week_start"].nunique(),
    )

    return dataframe


# =========================================================
# TIME-BASED DATA SPLIT
# =========================================================

def create_time_splits(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create chronological train, validation and test datasets."""

    all_weeks = sorted(
        dataframe["week_start"].unique()
    )

    minimum_weeks = (
        VALIDATION_WEEKS
        + TEST_WEEKS
        + 1
    )

    if len(all_weeks) < minimum_weeks:
        raise ValueError(
            "Not enough complete weeks for the "
            "requested data split."
        )

    validation_start = pd.Timestamp(
        all_weeks[
            -(
                VALIDATION_WEEKS
                + TEST_WEEKS
            )
        ]
    )

    test_start = pd.Timestamp(
        all_weeks[-TEST_WEEKS]
    )

    train_data = dataframe.loc[
        dataframe["week_start"]
        < validation_start
    ].copy()

    validation_data = dataframe.loc[
        (
            dataframe["week_start"]
            >= validation_start
        )
        & (
            dataframe["week_start"]
            < test_start
        )
    ].copy()

    test_data = dataframe.loc[
        dataframe["week_start"]
        >= test_start
    ].copy()

    print("\nTIME-BASED DATA SPLIT")
    print("-" * 65)

    print(
        "Training:",
        train_data["week_start"].min().date(),
        "to",
        train_data["week_start"].max().date(),
    )
    print(
        "Training rows:",
        len(train_data),
    )
    print(
        "Training weeks:",
        train_data["week_start"].nunique(),
    )

    print(
        "\nValidation:",
        validation_data["week_start"].min().date(),
        "to",
        validation_data["week_start"].max().date(),
    )
    print(
        "Validation rows:",
        len(validation_data),
    )
    print(
        "Validation weeks:",
        validation_data["week_start"].nunique(),
    )

    print(
        "\nTesting:",
        test_data["week_start"].min().date(),
        "to",
        test_data["week_start"].max().date(),
    )
    print(
        "Test rows:",
        len(test_data),
    )
    print(
        "Test weeks:",
        test_data["week_start"].nunique(),
    )

    return (
        train_data,
        validation_data,
        test_data,
    )


# =========================================================
# PREPROCESSING
# =========================================================

def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing for numeric and SKU features."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# =========================================================
# MODEL BUILDERS
# =========================================================

def build_model(
    model_name: str,
) -> Pipeline:
    """Create a complete preprocessing and forecasting pipeline."""

    if model_name == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

    elif model_name == "RandomForest":
        estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )

    return pipeline


# =========================================================
# FORECAST METRICS
# =========================================================

def calculate_metrics(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    model_name: str,
    evaluation_period: str,
) -> dict:
    """Calculate WAPE, MAE, RMSE and forecast bias."""

    actual_values = np.asarray(
        actual,
        dtype=float,
    )

    prediction_values = np.asarray(
        prediction,
        dtype=float,
    )

    prediction_values = np.clip(
        prediction_values,
        0,
        None,
    )

    errors = (
        prediction_values
        - actual_values
    )

    absolute_errors = np.abs(errors)

    actual_total = float(
        actual_values.sum()
    )

    forecast_total = float(
        prediction_values.sum()
    )

    if actual_total == 0:
        wape = np.nan
        bias = np.nan
    else:
        wape = float(
            absolute_errors.sum()
            / actual_total
            * 100
        )

        bias = float(
            errors.sum()
            / actual_total
            * 100
        )

    mae = float(
        absolute_errors.mean()
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(errors)
            )
        )
    )

    return {
        "model": model_name,
        "evaluation_period": evaluation_period,
        "wape_percent": wape,
        "mae": mae,
        "rmse": rmse,
        "bias_percent": bias,
        "actual_total": actual_total,
        "forecast_total": forecast_total,
        "evaluation_rows": int(
            len(actual_values)
        ),
    }


# =========================================================
# VALIDATE CANDIDATE MODELS
# =========================================================

def evaluate_candidate_models(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Train and compare candidate models on validation data."""

    model_names = [
        "HistGradientBoosting",
        "RandomForest",
    ]

    x_train = train_data[ALL_FEATURES]
    y_train = train_data[TARGET_COLUMN]

    x_validation = validation_data[ALL_FEATURES]
    y_validation = validation_data[TARGET_COLUMN]

    metric_records = []

    print("\nMODEL VALIDATION")
    print("=" * 65)

    for model_name in model_names:
        print(f"\nTraining {model_name}...")

        model = build_model(
            model_name
        )

        model.fit(
            x_train,
            y_train,
        )

        predictions = model.predict(
            x_validation
        )

        predictions = np.clip(
            predictions,
            0,
            None,
        )

        metrics = calculate_metrics(
            actual=y_validation,
            prediction=predictions,
            model_name=model_name,
            evaluation_period="Validation",
        )

        metric_records.append(
            metrics
        )

        print(
            f"{model_name} WAPE: "
            f"{metrics['wape_percent']:.2f}%"
        )

    validation_metrics = pd.DataFrame(
        metric_records
    ).sort_values(
        "wape_percent",
        ascending=True,
    ).reset_index(drop=True)

    validation_metrics["validation_rank"] = (
        np.arange(
            len(validation_metrics)
        )
        + 1
    )

    selected_model_name = str(
        validation_metrics.iloc[0][
            "model"
        ]
    )

    print("\nVALIDATION RESULTS")
    print("-" * 65)

    print(
        validation_metrics.to_string(
            index=False
        )
    )

    print(
        "\nSelected model:",
        selected_model_name,
    )

    return (
        validation_metrics,
        selected_model_name,
    )


# =========================================================
# FINAL MODEL AND TEST EVALUATION
# =========================================================

def train_and_test_final_model(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    selected_model_name: str,
) -> tuple[
    Pipeline,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Retrain selected model and evaluate it on final test data."""

    development_data = pd.concat(
        [
            train_data,
            validation_data,
        ],
        ignore_index=True,
    )

    x_development = development_data[
        ALL_FEATURES
    ]

    y_development = development_data[
        TARGET_COLUMN
    ]

    x_test = test_data[
        ALL_FEATURES
    ]

    y_test = test_data[
        TARGET_COLUMN
    ]

    final_model = build_model(
        selected_model_name
    )

    print(
        f"\nRetraining {selected_model_name} "
        "using training and validation data..."
    )

    final_model.fit(
        x_development,
        y_development,
    )

    model_predictions = np.clip(
        final_model.predict(
            x_test
        ),
        0,
        None,
    )

    previous_week_predictions = np.clip(
        test_data["demand_lag_1"]
        .fillna(0)
        .to_numpy(),
        0,
        None,
    )

    seasonal_predictions = np.clip(
        test_data["demand_lag_52"]
        .fillna(0)
        .to_numpy(),
        0,
        None,
    )

    metric_records = [
        calculate_metrics(
            actual=y_test,
            prediction=model_predictions,
            model_name=selected_model_name,
            evaluation_period="Final Test",
        ),
        calculate_metrics(
            actual=y_test,
            prediction=previous_week_predictions,
            model_name="Naive - Previous Week",
            evaluation_period="Final Test",
        ),
        calculate_metrics(
            actual=y_test,
            prediction=seasonal_predictions,
            model_name="Seasonal Naive - 52 Weeks",
            evaluation_period="Final Test",
        ),
    ]

    test_metrics = pd.DataFrame(
        metric_records
    ).sort_values(
        "wape_percent",
        ascending=True,
    ).reset_index(drop=True)

    test_metrics["test_rank"] = (
        np.arange(
            len(test_metrics)
        )
        + 1
    )

    baseline_wape = float(
        test_metrics.loc[
            test_metrics["model"].eq(
                "Naive - Previous Week"
            ),
            "wape_percent",
        ].iloc[0]
    )

    model_wape = float(
        test_metrics.loc[
            test_metrics["model"].eq(
                selected_model_name
            ),
            "wape_percent",
        ].iloc[0]
    )

    improvement = (
        baseline_wape
        - model_wape
    )

    test_metrics[
        "improvement_over_baseline_points"
    ] = np.where(
        test_metrics["model"].eq(
            selected_model_name
        ),
        improvement,
        0.0,
    )

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

    predictions[
        "naive_previous_week"
    ] = previous_week_predictions

    predictions[
        "seasonal_naive_52"
    ] = seasonal_predictions

    predictions[
        "selected_model"
    ] = selected_model_name

    predictions[
        "model_prediction"
    ] = model_predictions

    predictions[
        "absolute_error"
    ] = np.abs(
        predictions["model_prediction"]
        - predictions["actual_demand"]
    )

    return (
        final_model,
        test_metrics,
        predictions,
    )


# =========================================================
# SAVE RESULTS
# =========================================================

def save_outputs(
    final_model: Pipeline,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    selected_model_name: str,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save model, predictions, metrics and summary."""

    FINAL_MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        FINAL_MODEL_FILE,
    )

    validation_metrics.to_csv(
        VALIDATION_METRICS_FILE,
        index=False,
    )

    test_metrics.to_csv(
        TEST_METRICS_FILE,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    selected_test_row = test_metrics.loc[
        test_metrics["model"].eq(
            selected_model_name
        )
    ].iloc[0]

    baseline_test_row = test_metrics.loc[
        test_metrics["model"].eq(
            "Naive - Previous Week"
        )
    ].iloc[0]

    summary = {
        "selected_model": selected_model_name,
        "target": TARGET_COLUMN,
        "number_of_features": len(
            ALL_FEATURES
        ),
        "number_of_skus": int(
            test_data["sku_id"].nunique()
        ),
        "training_rows": int(
            len(train_data)
        ),
        "validation_rows": int(
            len(validation_data)
        ),
        "test_rows": int(
            len(test_data)
        ),
        "test_start": str(
            test_data["week_start"]
            .min()
            .date()
        ),
        "test_end": str(
            test_data["week_end"]
            .max()
            .date()
        ),
        "model_test_wape_percent": float(
            selected_test_row[
                "wape_percent"
            ]
        ),
        "baseline_test_wape_percent": float(
            baseline_test_row[
                "wape_percent"
            ]
        ),
        "wape_improvement_points": float(
            baseline_test_row[
                "wape_percent"
            ]
            - selected_test_row[
                "wape_percent"
            ]
        ),
        "features": ALL_FEATURES,
    }

    with open(
        MODEL_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\nSAVED OUTPUTS")
    print("-" * 65)
    print(FINAL_MODEL_FILE)
    print(VALIDATION_METRICS_FILE)
    print(TEST_METRICS_FILE)
    print(PREDICTIONS_FILE)
    print(MODEL_SUMMARY_FILE)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """Run the complete Week 3 forecasting workflow."""

    print("=" * 65)
    print("PROJECT FORESIGHT - WEEK 3 FORECAST MODELLING")
    print("=" * 65)

    dataframe = load_modelling_data()

    (
        train_data,
        validation_data,
        test_data,
    ) = create_time_splits(
        dataframe
    )

    (
        validation_metrics,
        selected_model_name,
    ) = evaluate_candidate_models(
        train_data,
        validation_data,
    )

    (
        final_model,
        test_metrics,
        predictions,
    ) = train_and_test_final_model(
        train_data,
        validation_data,
        test_data,
        selected_model_name,
    )

    print("\nFINAL TEST RESULTS")
    print("=" * 65)

    print(
        test_metrics.to_string(
            index=False
        )
    )

    selected_result = test_metrics.loc[
        test_metrics["model"].eq(
            selected_model_name
        )
    ].iloc[0]

    baseline_result = test_metrics.loc[
        test_metrics["model"].eq(
            "Naive - Previous Week"
        )
    ].iloc[0]

    improvement = (
        baseline_result["wape_percent"]
        - selected_result["wape_percent"]
    )

    print("\nFINAL MODEL SUMMARY")
    print("-" * 65)
    print(
        "Selected model:",
        selected_model_name,
    )
    print(
        "Model WAPE:",
        f"{selected_result['wape_percent']:.2f}%",
    )
    print(
        "Baseline WAPE:",
        f"{baseline_result['wape_percent']:.2f}%",
    )
    print(
        "Improvement:",
        f"{improvement:.2f} percentage points",
    )

    if improvement > 0:
        print(
            "Result: The model beat the baseline."
        )
    else:
        print(
            "Result: The model did not beat the baseline."
        )

    save_outputs(
        final_model=final_model,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        predictions=predictions,
        selected_model_name=selected_model_name,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
    )

    print("\n" + "=" * 65)
    print("WEEK 3 MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()