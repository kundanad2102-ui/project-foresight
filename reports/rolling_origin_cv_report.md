# Rolling-Origin Cross-Validation Report

## Validation Design

- Expanding-window folds: 4
- Forecast horizon per fold: 8 weeks
- Minimum training history: 52 weeks
- Final test period remained untouched during cross-validation.

## Model Selection

- Selected machine-learning model: HistGradientBoosting
- Rolling-CV WAPE: 44.58%
- Rolling-CV MAE: 59.04
- Rolling-CV RMSE: 111.78
- Rolling-CV bias: 2.27%

## Baseline Comparison

- Previous-week naive WAPE: 51.16%
- Seasonal-naive 52-week WAPE: 74.45%
- Improvement over previous-week naive: 6.58 percentage points
- Improvement over seasonal naive: 29.88 percentage points

## Interpretation

Rolling-origin validation evaluates performance across multiple historical forecast origins instead of relying on one validation window.

Every training window contains only information available before its validation period, preserving chronological order and preventing future-data leakage.
