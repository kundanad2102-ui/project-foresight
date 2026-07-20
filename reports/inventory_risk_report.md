# Inventory Risk Scoring Report

## Scoring Design

- Scoring week: 2024-12-23
- True future forecast window: 2024-12-30 to 2025-02-17
- Forecast model used for future predictions: HistGradientBoosting
- Operational forecast horizon: 8 weeks
- Safety-stock service factor: 1.65
- Inventory position: ending on-hand units plus ending on-order units

## Executive Summary

- SKUs scored: 150
- High stockout-risk SKUs: 4
- Medium stockout-risk SKUs: 82
- High overstock-risk SKUs: 1
- Medium overstock-risk SKUs: 8
- Potential lost revenue: 405,296.04
- Excess inventory value: 5,029.76
- Recommended replenishment cost: 556,031.40

## Risk Logic

Stockout risk compares inventory position with forecast demand over supplier lead time plus safety stock.
Overstock risk compares inventory position with forecast demand over the 8-week planning horizon plus safety stock.
Recommended order quantities replenish inventory to lead-time demand plus one review week and safety stock, then round upward to the SKU minimum order quantity.

## Top Stockout Priorities

| Rank | SKU | Category | Stockout Risk | Gap Units | Potential Lost Revenue | Action |
|---:|---|---|---:|---:|---:|---|
| 1 | SKU10148 | Food & Bev | 74.28 | 170.44 | 3761.53 | Expedite supply and place the recommended replenishment order |
| 2 | SKU10016 | Home Decor | 55.27 | 98.85 | 1511.44 | Expedite supply and place the recommended replenishment order |
| 3 | SKU10037 | Kitchenware | 54.33 | 229.58 | 1921.61 | Expedite supply and place the recommended replenishment order |
| 4 | SKU10022 | Home Decor | 50.54 | 124.64 | 2054.07 | Expedite supply and place the recommended replenishment order |
| 6 | SKU10120 | Beauty | 49.95 | 322.32 | 13582.66 | Review supplier timing and place or expedite the recommended order |
| 7 | SKU10048 | Stationery | 49.58 | 61.96 | 623.89 | Review supplier timing and place or expedite the recommended order |
| 8 | SKU10036 | Kitchenware | 46.34 | 124.38 | 1606.94 | Review supplier timing and place or expedite the recommended order |
| 9 | SKU10118 | Electronics | 46.04 | 215.01 | 28117.35 | Review supplier timing and place or expedite the recommended order |
| 10 | SKU10008 | Home Decor | 46.03 | 79.31 | 436.99 | Review supplier timing and place or expedite the recommended order |
| 11 | SKU10119 | Electronics | 44.87 | 244.13 | 46172.57 | Review supplier timing and place or expedite the recommended order |

## Top Overstock Priorities

| Rank | SKU | Category | Overstock Risk | Excess Units | Excess Value | Action |
|---:|---|---|---:|---:|---:|---|
| 5 | SKU10097 | Apparel | 49.36 | 124.88 | 3907.36 | Pause replenishment; consider transfer, promotion or markdown |
| 68 | SKU10088 | Apparel | 14.41 | 22.05 | 458.46 | Reduce or defer replenishment and monitor weekly demand |
| 77 | SKU10132 | Beauty | 7.47 | 10.09 | 53.27 | Reduce or defer replenishment and monitor weekly demand |
| 79 | SKU10104 | Apparel | 6.90 | 7.80 | 298.51 | Reduce or defer replenishment and monitor weekly demand |
| 82 | SKU10086 | Apparel | 5.04 | 14.42 | 283.41 | Reduce or defer replenishment and monitor weekly demand |
| 90 | SKU10098 | Apparel | 0.63 | 1.12 | 28.75 | Reduce or defer replenishment and monitor weekly demand |
| 93 | SKU10023 | Home Decor | 0.00 | 0.00 | 0.00 | Reduce or defer replenishment and monitor weekly demand |
| 94 | SKU10067 | Toys | 0.00 | 0.00 | 0.00 | Reduce or defer replenishment and monitor weekly demand |
| 95 | SKU10084 | Toys | 0.00 | 0.00 | 0.00 | Reduce or defer replenishment and monitor weekly demand |

## Forecast Assumptions and Limitations

Risk scoring uses a newly generated recursive 8-week future forecast for every SKU. Each predicted week contributes to the next week's lag and rolling-demand features.
No future promotion schedule was supplied, so future promotion flags are assumed to be zero. Latest known inventory inputs are carried forward for forecasting features, while risk calculations use the latest actual on-hand and on-order inventory position.
Forecast uncertainty intervals and service-level optimisation are not yet included and remain production enhancements.
