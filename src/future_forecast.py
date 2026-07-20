from __future__ import annotations

from pathlib import Path
import json

import joblib
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

BASE_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_forecast_model.joblib"
)

REFITTED_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "future_forecast_model.joblib"
)

MODEL_SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_summary.json"
)

FUTURE_FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "future_8_week_forecast.csv"
)

FUTURE_FORECAST_REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "future_forecast_report.md"
)


# =========================================================
# FORECAST SETTINGS
# =========================================================

TARGET_COLUMN = "demand"
FORECAST_HORIZON_WEEKS = 8
DEFAULT_FUTURE_PROMO_FLAG = 0

CATEGORICAL_FEATURES = [
    "sku_id",
]

NUMERIC_FEATURES = [
    "iso_year",
    "iso_week",
    "month",
    "quarter",
    "iso_week_sin",
    "iso_week_cos",
    "month_sin",
    "month_cos",
    "promo_week_flag",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_4",
    "demand_lag_8",
    "demand_lag_13",
    "demand_lag_26",
    "demand_lag_52",
    "demand_rolling_mean_4",
    "demand_rolling_std_4",
    "demand_rolling_mean_8",
    "demand_rolling_std_8",
    "demand_rolling_mean_13",
    "demand_rolling_std_13",
    "zero_demand_rate_4",
    "ending_on_hand_units_lag_1",
    "ending_on_order_units_lag_1",
    "stockout_days_lag_1",
    "lost_sales_lag_1",
    "promo_week_flag_lag_1",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

REQUIRED_WEEKLY_COLUMNS = {
    "sku_id",
    "week_start",
    "week_end",
    "demand",
    "is_complete_week",
    "promo_week_flag",
    "ending_on_hand_units",
    "ending_on_order_units",
    "stockout_days",
    "lost_sales",
}


# =========================================================
# LOAD AND VALIDATE INPUTS
# =========================================================

def load_inputs() -> tuple[pd.DataFrame, object, str]:
    """Load complete weekly history and the fitted forecasting pipeline."""

    for filepath in [WEEKLY_DATA_FILE, BASE_MODEL_FILE]:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required file not found: {filepath}"
            )

    weekly = pd.read_csv(
        WEEKLY_DATA_FILE,
        parse_dates=["week_start", "week_end"],
    )

    missing_columns = REQUIRED_WEEKLY_COLUMNS - set(weekly.columns)

    if missing_columns:
        raise ValueError(
            "weekly_demand.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    weekly = weekly.loc[
        weekly["is_complete_week"].eq(1)
    ].copy()

    if weekly.empty:
        raise ValueError(
            "weekly_demand.csv contains no complete weeks."
        )

    weekly["sku_id"] = (
        weekly["sku_id"]
        .astype(str)
        .str.strip()
    )

    weekly["demand"] = pd.to_numeric(
        weekly["demand"],
        errors="coerce",
    )

    if weekly["demand"].isna().any():
        raise ValueError(
            "Complete weekly demand contains missing or non-numeric values."
        )

    if weekly["demand"].lt(0).any():
        raise ValueError(
            "Complete weekly demand contains negative values."
        )

    if weekly.duplicated(
        subset=["sku_id", "week_start"]
    ).any():
        raise ValueError(
            "Duplicate SKU-week rows found in weekly_demand.csv."
        )

    weekly = (
        weekly
        .sort_values(["sku_id", "week_start"])
        .reset_index(drop=True)
    )

    model = joblib.load(BASE_MODEL_FILE)

    selected_model_name = get_selected_model_name(model)

    return weekly, model, selected_model_name


def get_selected_model_name(model: object) -> str:
    """Read the selected model name from the saved summary when available."""

    if MODEL_SUMMARY_FILE.exists():
        try:
            with open(
                MODEL_SUMMARY_FILE,
                "r",
                encoding="utf-8",
            ) as file:
                summary = json.load(file)

            selected_model = summary.get("selected_model")

            if selected_model:
                return str(selected_model)
        except (OSError, json.JSONDecodeError):
            pass

    named_steps = getattr(model, "named_steps", {})
    estimator = named_steps.get("model")

    if estimator is not None:
        return estimator.__class__.__name__

    return model.__class__.__name__


# =========================================================
# PREPARE HISTORICAL MODEL FEATURES
# =========================================================

def add_calendar_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create cyclic calendar features used by the forecasting model."""

    result = dataframe.copy()

    if "iso_year" not in result.columns or "iso_week" not in result.columns:
        iso_calendar = result["week_start"].dt.isocalendar()
        result["iso_year"] = iso_calendar["year"].astype(int)
        result["iso_week"] = iso_calendar["week"].astype(int)

    if "month" not in result.columns:
        result["month"] = result["week_start"].dt.month

    if "quarter" not in result.columns:
        result["quarter"] = result["week_start"].dt.quarter

    result["iso_week_sin"] = np.sin(
        2 * np.pi * result["iso_week"] / 52
    )
    result["iso_week_cos"] = np.cos(
        2 * np.pi * result["iso_week"] / 52
    )
    result["month_sin"] = np.sin(
        2 * np.pi * result["month"] / 12
    )
    result["month_cos"] = np.cos(
        2 * np.pi * result["month"] / 12
    )

    return result


def validate_model_features(dataframe: pd.DataFrame) -> None:
    """Confirm that historical training features match the saved model design."""

    missing_features = set(ALL_FEATURES) - set(dataframe.columns)

    if missing_features:
        raise ValueError(
            "weekly_demand.csv is missing model features: "
            f"{sorted(missing_features)}"
        )


# =========================================================
# RECURSIVE FEATURE CREATION
# =========================================================

def lag_value(values: list[float], lag: int) -> float:
    """Return a historical or recursively forecast demand lag."""

    if len(values) < lag:
        return np.nan

    return float(values[-lag])


def rolling_mean(values: list[float], window: int) -> float:
    """Calculate a rolling demand mean using prior observations only."""

    if len(values) < window:
        return np.nan

    return float(np.mean(values[-window:]))


def rolling_std(values: list[float], window: int) -> float:
    """Calculate sample rolling standard deviation like pandas rolling.std."""

    if len(values) < window:
        return np.nan

    return float(np.std(values[-window:], ddof=1))


def zero_demand_rate(values: list[float], window: int = 4) -> float:
    """Calculate the prior-window share of zero-demand weeks."""

    if len(values) < window:
        return np.nan

    recent_values = np.asarray(values[-window:], dtype=float)
    return float(np.mean(recent_values == 0))


def build_future_feature_row(
    sku_id: str,
    forecast_week: pd.Timestamp,
    demand_history: list[float],
    latest_state: pd.Series,
    horizon_number: int,
) -> dict:
    """Build one leakage-safe future feature row for a single SKU."""

    iso_calendar = forecast_week.isocalendar()
    iso_week = int(iso_calendar.week)
    month = int(forecast_week.month)

    first_future_week = horizon_number == 1

    if first_future_week:
        stockout_days_lag_1 = latest_state["stockout_days"]
        lost_sales_lag_1 = latest_state["lost_sales"]
        promo_week_flag_lag_1 = latest_state["promo_week_flag"]
    else:
        # No future stockout, lost-sales or promotion plan is supplied.
        # These unknown future operational values are therefore set to zero.
        stockout_days_lag_1 = 0
        lost_sales_lag_1 = 0
        promo_week_flag_lag_1 = DEFAULT_FUTURE_PROMO_FLAG

    return {
        "sku_id": sku_id,
        "iso_year": int(iso_calendar.year),
        "iso_week": iso_week,
        "month": month,
        "quarter": int(forecast_week.quarter),
        "iso_week_sin": float(
            np.sin(2 * np.pi * iso_week / 52)
        ),
        "iso_week_cos": float(
            np.cos(2 * np.pi * iso_week / 52)
        ),
        "month_sin": float(
            np.sin(2 * np.pi * month / 12)
        ),
        "month_cos": float(
            np.cos(2 * np.pi * month / 12)
        ),
        "promo_week_flag": DEFAULT_FUTURE_PROMO_FLAG,
        "demand_lag_1": lag_value(demand_history, 1),
        "demand_lag_2": lag_value(demand_history, 2),
        "demand_lag_4": lag_value(demand_history, 4),
        "demand_lag_8": lag_value(demand_history, 8),
        "demand_lag_13": lag_value(demand_history, 13),
        "demand_lag_26": lag_value(demand_history, 26),
        "demand_lag_52": lag_value(demand_history, 52),
        "demand_rolling_mean_4": rolling_mean(
            demand_history,
            4,
        ),
        "demand_rolling_std_4": rolling_std(
            demand_history,
            4,
        ),
        "demand_rolling_mean_8": rolling_mean(
            demand_history,
            8,
        ),
        "demand_rolling_std_8": rolling_std(
            demand_history,
            8,
        ),
        "demand_rolling_mean_13": rolling_mean(
            demand_history,
            13,
        ),
        "demand_rolling_std_13": rolling_std(
            demand_history,
            13,
        ),
        "zero_demand_rate_4": zero_demand_rate(
            demand_history,
            4,
        ),
        # Future inventory snapshots are unknown. Carry the latest known
        # inventory-position inputs forward as a transparent scenario assumption.
        "ending_on_hand_units_lag_1": latest_state[
            "ending_on_hand_units"
        ],
        "ending_on_order_units_lag_1": latest_state[
            "ending_on_order_units"
        ],
        "stockout_days_lag_1": stockout_days_lag_1,
        "lost_sales_lag_1": lost_sales_lag_1,
        "promo_week_flag_lag_1": promo_week_flag_lag_1,
    }


# =========================================================
# MODEL REFIT AND FUTURE FORECAST
# =========================================================

def refit_model_on_all_history(
    model: object,
    weekly: pd.DataFrame,
) -> object:
    """Refit the selected pipeline on all complete historical weeks."""

    modelling_data = add_calendar_features(weekly)
    validate_model_features(modelling_data)

    x_all = modelling_data[ALL_FEATURES].copy()
    y_all = modelling_data[TARGET_COLUMN].copy()

    print("Refitting the selected model on all complete history...")

    model.fit(x_all, y_all)

    REFITTED_MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, REFITTED_MODEL_FILE)

    return model


def generate_recursive_forecast(
    weekly: pd.DataFrame,
    model: object,
    selected_model_name: str,
) -> pd.DataFrame:
    """Generate eight recursive weekly forecasts for every SKU."""

    latest_complete_week = pd.Timestamp(
        weekly["week_start"].max()
    )

    latest_rows = (
        weekly.loc[
            weekly["week_start"].eq(latest_complete_week)
        ]
        .sort_values("sku_id")
        .set_index("sku_id")
    )

    all_skus = sorted(weekly["sku_id"].unique().tolist())

    missing_latest_skus = sorted(
        set(all_skus) - set(latest_rows.index)
    )

    if missing_latest_skus:
        raise ValueError(
            "The latest complete week does not contain every SKU. "
            f"Example missing SKUs: {missing_latest_skus[:10]}"
        )

    demand_histories: dict[str, list[float]] = {}

    for sku_id, group in weekly.groupby("sku_id", sort=True):
        demand_histories[str(sku_id)] = (
            group
            .sort_values("week_start")["demand"]
            .astype(float)
            .tolist()
        )

    short_history_skus = [
        sku_id
        for sku_id, values in demand_histories.items()
        if len(values) < 52
    ]

    if short_history_skus:
        print(
            "Warning: some SKUs have fewer than 52 complete weeks; "
            "the model imputer will handle unavailable long lags."
        )

    forecast_records: list[dict] = []

    for horizon_number in range(1, FORECAST_HORIZON_WEEKS + 1):
        forecast_week = (
            latest_complete_week
            + pd.Timedelta(weeks=horizon_number)
        )

        feature_records = []

        for sku_id in all_skus:
            feature_records.append(
                build_future_feature_row(
                    sku_id=sku_id,
                    forecast_week=forecast_week,
                    demand_history=demand_histories[sku_id],
                    latest_state=latest_rows.loc[sku_id],
                    horizon_number=horizon_number,
                )
            )

        future_features = pd.DataFrame(feature_records)

        missing_features = set(ALL_FEATURES) - set(
            future_features.columns
        )

        if missing_features:
            raise ValueError(
                "Future feature construction failed. Missing: "
                f"{sorted(missing_features)}"
            )

        predictions = np.clip(
            np.asarray(
                model.predict(future_features[ALL_FEATURES]),
                dtype=float,
            ),
            0,
            None,
        )

        for sku_id, prediction in zip(
            future_features["sku_id"],
            predictions,
            strict=True,
        ):
            forecast_value = float(prediction)
            demand_histories[str(sku_id)].append(forecast_value)

            forecast_records.append(
                {
                    "sku_id": str(sku_id),
                    "forecast_horizon_week": horizon_number,
                    "forecast_week": forecast_week,
                    "forecast_week_end": (
                        forecast_week + pd.Timedelta(days=6)
                    ),
                    "forecast_demand": forecast_value,
                    "selected_model": selected_model_name,
                    "forecast_generated_from_week": latest_complete_week,
                    "future_promo_flag_assumption": (
                        DEFAULT_FUTURE_PROMO_FLAG
                    ),
                }
            )

        print(
            f"Generated horizon week {horizon_number}: "
            f"{forecast_week.date()}"
        )

    forecast = pd.DataFrame(forecast_records)

    expected_rows = len(all_skus) * FORECAST_HORIZON_WEEKS

    if len(forecast) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} forecast rows, "
            f"but generated {len(forecast)}."
        )

    weeks_per_sku = forecast.groupby("sku_id")[
        "forecast_week"
    ].nunique()

    if not weeks_per_sku.eq(FORECAST_HORIZON_WEEKS).all():
        raise RuntimeError(
            "Every SKU must have exactly eight future forecast weeks."
        )

    return (
        forecast
        .sort_values(["forecast_week", "sku_id"])
        .reset_index(drop=True)
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(forecast: pd.DataFrame) -> None:
    """Save the detailed future forecast and a short methodology report."""

    FUTURE_FORECAST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    FUTURE_FORECAST_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        FUTURE_FORECAST_FILE,
        index=False,
    )

    forecast_start = pd.Timestamp(
        forecast["forecast_week"].min()
    )
    forecast_end = pd.Timestamp(
        forecast["forecast_week"].max()
    )
    generated_from = pd.Timestamp(
        forecast["forecast_generated_from_week"].iloc[0]
    )
    selected_model = str(
        forecast["selected_model"].iloc[0]
    )

    report_lines = [
        "# Future 8-Week Forecast Report",
        "",
        f"- Model: {selected_model}",
        f"- Latest complete historical week: {generated_from.date()}",
        f"- Forecast start week: {forecast_start.date()}",
        f"- Forecast end week: {forecast_end.date()}",
        f"- Forecast horizon: {FORECAST_HORIZON_WEEKS} weeks",
        f"- SKUs forecast: {forecast['sku_id'].nunique()}",
        f"- Forecast rows: {len(forecast)}",
        "",
        "## Forecast Method",
        "",
        (
            "The selected forecasting pipeline is refitted on all complete "
            "historical weeks and then used recursively. Each predicted week "
            "is appended to demand history before the next week's lag and "
            "rolling features are created."
        ),
        "",
        "## Scenario Assumptions",
        "",
        (
            "Future promotion flags are set to zero because no future promotion "
            "schedule was supplied."
        ),
        (
            "Latest known on-hand and on-order inventory inputs are carried "
            "forward for model features because future inventory snapshots are "
            "not yet available."
        ),
        (
            "Future stockout-day and lost-sales lag inputs are set to zero after "
            "the first forecast week."
        ),
    ]

    FUTURE_FORECAST_REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\nSAVED FUTURE-FORECAST OUTPUTS")
    print("-" * 72)
    print(FUTURE_FORECAST_FILE)
    print(REFITTED_MODEL_FILE)
    print(FUTURE_FORECAST_REPORT_FILE)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """Run the complete eight-week future-forecast workflow."""

    print("=" * 72)
    print("PROJECT FORESIGHT - TRUE 8-WEEK FUTURE FORECAST")
    print("=" * 72)

    weekly, model, selected_model_name = load_inputs()

    print("Complete historical rows:", len(weekly))
    print("Unique SKUs:", weekly["sku_id"].nunique())
    print(
        "Historical window:",
        weekly["week_start"].min().date(),
        "to",
        weekly["week_start"].max().date(),
    )
    print("Selected model:", selected_model_name)

    model = refit_model_on_all_history(
        model=model,
        weekly=weekly,
    )

    forecast = generate_recursive_forecast(
        weekly=weekly,
        model=model,
        selected_model_name=selected_model_name,
    )

    print("\nFORECAST SUMMARY")
    print("-" * 72)
    print("Forecast rows:", len(forecast))
    print("SKUs forecast:", forecast["sku_id"].nunique())
    print(
        "Forecast weeks:",
        forecast["forecast_week"].nunique(),
    )
    print(
        "Forecast window:",
        forecast["forecast_week"].min().date(),
        "to",
        forecast["forecast_week"].max().date(),
    )
    print(
        "Total forecast demand:",
        f"{forecast['forecast_demand'].sum():,.2f}",
    )

    save_outputs(forecast)

    print("\n" + "=" * 72)
    print("TRUE 8-WEEK FUTURE FORECAST COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()