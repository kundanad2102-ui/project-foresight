# Future 8-Week Forecast Report

- Model: HistGradientBoosting
- Latest complete historical week: 2024-12-23
- Forecast start week: 2024-12-30
- Forecast end week: 2025-02-17
- Forecast horizon: 8 weeks
- SKUs forecast: 150
- Forecast rows: 1200

## Forecast Method

The selected forecasting pipeline is refitted on all complete historical weeks and then used recursively. Each predicted week is appended to demand history before the next week's lag and rolling features are created.

## Scenario Assumptions

Future promotion flags are set to zero because no future promotion schedule was supplied.
Latest known on-hand and on-order inventory inputs are carried forward for model features because future inventory snapshots are not yet available.
Future stockout-day and lost-sales lag inputs are set to zero after the first forecast week.
