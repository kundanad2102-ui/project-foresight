from pathlib import Path

import numpy as np
import pandas as pd

from src.model import (
    ALL_FEATURES,
    TARGET_COLUMN,
    build_model,
    calculate_metrics,
    create_time_splits,
    load_modelling_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CV_FOLD_METRICS_FILE = PROJECT_ROOT / "reports" / "rolling_origin_cv_folds.csv"
CV_SUMMARY_FILE = PROJECT_ROOT / "reports" / "rolling_origin_cv_summary.csv"
CV_REPORT_FILE = PROJECT_ROOT / "reports" / "rolling_origin_cv_report.md"

CV_FOLDS = 4
FORECAST_HORIZON_WEEKS = 8
MINIMUM_TRAINING_WEEKS = 52

MACHINE_LEARNING_MODELS = [
    "HistGradientBoosting",
    "RandomForest",
]

BASELINE_MODELS = [
    "Naive - Previous Week",
    "Seasonal Naive - 52 Weeks",
]

ALL_MODELS = MACHINE_LEARNING_MODELS + BASELINE_MODELS


def create_rolling_origin_folds(
    development_data: pd.DataFrame,
) -> list[dict]:
    """Create expanding-window folds with an 8-week validation horizon."""

    all_weeks = sorted(
        pd.to_datetime(
            development_data["week_start"].unique()
        )
    )

    required_weeks = (
        MINIMUM_TRAINING_WEEKS
        + CV_FOLDS * FORECAST_HORIZON_WEEKS
    )

    if len(all_weeks) < required_weeks:
        raise ValueError(
            "Not enough development weeks. "
            f"Required at least {required_weeks}, found {len(all_weeks)}."
        )

    first_validation_index = (
        len(all_weeks)
        - CV_FOLDS * FORECAST_HORIZON_WEEKS
    )

    folds = []

    for fold_number in range(1, CV_FOLDS + 1):
        validation_start_index = (
            first_validation_index
            + (fold_number - 1) * FORECAST_HORIZON_WEEKS
        )
        validation_end_index = (
            validation_start_index
            + FORECAST_HORIZON_WEEKS
        )

        training_weeks = all_weeks[:validation_start_index]
        validation_weeks = all_weeks[
            validation_start_index:validation_end_index
        ]

        if len(training_weeks) < MINIMUM_TRAINING_WEEKS:
            raise ValueError(
                f"Fold {fold_number} has only {len(training_weeks)} "
                "training weeks."
            )

        validation_start = pd.Timestamp(validation_weeks[0])

        train_data = development_data.loc[
            development_data["week_start"] < validation_start
        ].copy()

        validation_data = development_data.loc[
            development_data["week_start"].isin(validation_weeks)
        ].copy()

        folds.append(
            {
                "fold": fold_number,
                "train_data": train_data,
                "validation_data": validation_data,
                "train_start": pd.Timestamp(training_weeks[0]),
                "train_end": pd.Timestamp(training_weeks[-1]),
                "validation_start": pd.Timestamp(validation_weeks[0]),
                "validation_end": pd.Timestamp(validation_weeks[-1]),
                "training_weeks": len(training_weeks),
                "validation_weeks": len(validation_weeks),
            }
        )

    return folds


def generate_predictions(
    model_name: str,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> np.ndarray:
    """Generate model or baseline predictions for one fold."""

    if model_name in MACHINE_LEARNING_MODELS:
        model = build_model(model_name)
        model.fit(
            train_data[ALL_FEATURES],
            train_data[TARGET_COLUMN],
        )
        predictions = model.predict(
            validation_data[ALL_FEATURES]
        )

    elif model_name == "Naive - Previous Week":
        predictions = (
            validation_data["demand_lag_1"]
            .fillna(0)
            .to_numpy()
        )

    elif model_name == "Seasonal Naive - 52 Weeks":
        predictions = (
            validation_data["demand_lag_52"]
            .fillna(0)
            .to_numpy()
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return np.clip(
        np.asarray(predictions, dtype=float),
        0,
        None,
    )


def run_rolling_origin_cv(
    development_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Evaluate candidate and baseline models across expanding folds."""

    folds = create_rolling_origin_folds(development_data)

    fold_records = []
    combined = {
        name: {"actual": [], "prediction": []}
        for name in ALL_MODELS
    }

    print("\nROLLING-ORIGIN CROSS-VALIDATION")
    print("=" * 72)
    print("Folds:", CV_FOLDS)
    print("Forecast horizon:", FORECAST_HORIZON_WEEKS, "weeks")
    print("Minimum training history:", MINIMUM_TRAINING_WEEKS, "weeks")

    for fold in folds:
        train_data = fold["train_data"]
        validation_data = fold["validation_data"]
        actual = validation_data[TARGET_COLUMN].to_numpy()

        print("\n" + "-" * 72)
        print(
            f"Fold {fold['fold']}: "
            f"train {fold['train_start'].date()} to "
            f"{fold['train_end'].date()} | "
            f"validate {fold['validation_start'].date()} to "
            f"{fold['validation_end'].date()}"
        )

        for model_name in ALL_MODELS:
            predictions = generate_predictions(
                model_name,
                train_data,
                validation_data,
            )

            metrics = calculate_metrics(
                actual=actual,
                prediction=predictions,
                model_name=model_name,
                evaluation_period=f"Rolling CV Fold {fold['fold']}",
            )

            metrics.update(
                {
                    "fold": fold["fold"],
                    "train_start": fold["train_start"].date(),
                    "train_end": fold["train_end"].date(),
                    "validation_start": fold["validation_start"].date(),
                    "validation_end": fold["validation_end"].date(),
                    "training_weeks": fold["training_weeks"],
                    "validation_weeks": fold["validation_weeks"],
                    "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
                }
            )
            fold_records.append(metrics)

            combined[model_name]["actual"].append(actual)
            combined[model_name]["prediction"].append(predictions)

            print(
                f"{model_name}: "
                f"WAPE {metrics['wape_percent']:.2f}%"
            )

    fold_metrics = pd.DataFrame(fold_records)
    summary_records = []

    for model_name in ALL_MODELS:
        model_folds = fold_metrics.loc[
            fold_metrics["model"].eq(model_name)
        ]

        combined_metrics = calculate_metrics(
            actual=np.concatenate(combined[model_name]["actual"]),
            prediction=np.concatenate(combined[model_name]["prediction"]),
            model_name=model_name,
            evaluation_period="Rolling-Origin Cross-Validation",
        )

        combined_metrics.update(
            {
                "folds": CV_FOLDS,
                "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
                "minimum_training_weeks": MINIMUM_TRAINING_WEEKS,
                "mean_fold_wape_percent": float(
                    model_folds["wape_percent"].mean()
                ),
                "std_fold_wape_percent": float(
                    model_folds["wape_percent"].std(ddof=0)
                ),
                "minimum_fold_wape_percent": float(
                    model_folds["wape_percent"].min()
                ),
                "maximum_fold_wape_percent": float(
                    model_folds["wape_percent"].max()
                ),
            }
        )
        summary_records.append(combined_metrics)

    summary = (
        pd.DataFrame(summary_records)
        .sort_values("wape_percent")
        .reset_index(drop=True)
    )
    summary["overall_rank"] = np.arange(len(summary)) + 1

    selected_model_name = str(
        summary.loc[
            summary["model"].isin(MACHINE_LEARNING_MODELS)
        ]
        .sort_values("wape_percent")
        .iloc[0]["model"]
    )

    print("\nROLLING-CV SUMMARY")
    print("=" * 72)
    print(
        summary[
            [
                "model",
                "wape_percent",
                "mae",
                "rmse",
                "bias_percent",
                "mean_fold_wape_percent",
                "std_fold_wape_percent",
                "overall_rank",
            ]
        ].to_string(index=False)
    )
    print(
        "\nSelected machine-learning model:",
        selected_model_name,
    )

    return fold_metrics, summary, selected_model_name


def save_cv_outputs(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    selected_model_name: str,
) -> None:
    """Save detailed fold metrics, summary metrics and a Markdown report."""

    CV_FOLD_METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        CV_FOLD_METRICS_FILE,
        index=False,
    )
    summary.to_csv(
        CV_SUMMARY_FILE,
        index=False,
    )

    selected_row = summary.loc[
        summary["model"].eq(selected_model_name)
    ].iloc[0]

    previous_week_row = summary.loc[
        summary["model"].eq("Naive - Previous Week")
    ].iloc[0]

    seasonal_row = summary.loc[
        summary["model"].eq("Seasonal Naive - 52 Weeks")
    ].iloc[0]

    report = f"""# Rolling-Origin Cross-Validation Report

## Validation Design

- Expanding-window folds: {CV_FOLDS}
- Forecast horizon per fold: {FORECAST_HORIZON_WEEKS} weeks
- Minimum training history: {MINIMUM_TRAINING_WEEKS} weeks
- Final test period remained untouched during cross-validation.

## Model Selection

- Selected machine-learning model: {selected_model_name}
- Rolling-CV WAPE: {selected_row['wape_percent']:.2f}%
- Rolling-CV MAE: {selected_row['mae']:.2f}
- Rolling-CV RMSE: {selected_row['rmse']:.2f}
- Rolling-CV bias: {selected_row['bias_percent']:.2f}%

## Baseline Comparison

- Previous-week naive WAPE: {previous_week_row['wape_percent']:.2f}%
- Seasonal-naive 52-week WAPE: {seasonal_row['wape_percent']:.2f}%
- Improvement over previous-week naive: {previous_week_row['wape_percent'] - selected_row['wape_percent']:.2f} percentage points
- Improvement over seasonal naive: {seasonal_row['wape_percent'] - selected_row['wape_percent']:.2f} percentage points

## Interpretation

Rolling-origin validation evaluates performance across multiple historical forecast origins instead of relying on one validation window.

Every training window contains only information available before its validation period, preserving chronological order and preventing future-data leakage.
"""

    CV_REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print("\nSAVED OUTPUTS")
    print(CV_FOLD_METRICS_FILE)
    print(CV_SUMMARY_FILE)
    print(CV_REPORT_FILE)


def main() -> None:
    """Run the Week 3 rolling-origin validation workflow."""

    print("=" * 72)
    print("PROJECT FORESIGHT - ROLLING-ORIGIN CROSS-VALIDATION")
    print("=" * 72)

    dataframe = load_modelling_data()

    train_data, validation_data, test_data = create_time_splits(
        dataframe
    )

    development_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    print("\nCROSS-VALIDATION DATA BOUNDARY")
    print("-" * 72)
    print(
        "Development data:",
        development_data["week_start"].min().date(),
        "to",
        development_data["week_start"].max().date(),
    )
    print(
        "Untouched final test:",
        test_data["week_start"].min().date(),
        "to",
        test_data["week_start"].max().date(),
    )

    fold_metrics, summary, selected_model_name = (
        run_rolling_origin_cv(development_data)
    )

    save_cv_outputs(
        fold_metrics,
        summary,
        selected_model_name,
    )

    print("\n" + "=" * 72)
    print("ROLLING-ORIGIN CROSS-VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()