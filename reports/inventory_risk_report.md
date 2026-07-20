# Inventory Risk Scoring Report

## Scoring Design

- Scoring week: 2024-12-23
- Forecast-output averaging window: 2024-11-04 to 2024-12-23
- Forecast model represented in the prediction file: HistGradientBoosting
- Operational forecast horizon: 8 weeks
- Safety-stock service factor: 1.65
- Inventory position: ending on-hand units plus ending on-order units

## Executive Summary

- SKUs scored: 150
- High stockout-risk SKUs: 9
- Medium stockout-risk SKUs: 93
- High overstock-risk SKUs: 3
- Medium overstock-risk SKUs: 8
- Potential lost revenue: 793,250.16
- Excess inventory value: 6,347.14
- Recommended replenishment cost: 1,128,180.40

## Risk Logic

Stockout risk compares inventory position with forecast demand over supplier lead time plus safety stock.
Overstock risk compares inventory position with forecast demand over the 8-week planning horizon plus safety stock.
Recommended order quantities replenish inventory to lead-time demand plus one review week and safety stock, then round upward to the SKU minimum order quantity.

## Top Stockout Priorities

| Rank | SKU | Category | Stockout Risk | Gap Units | Potential Lost Revenue | Action |
|---:|---|---|---:|---:|---:|---|
| 1 | SKU10148 | Food & Bev | 75.85 | 185.28 | 4089.03 | Expedite supply and place the recommended replenishment order |
| 2 | SKU10016 | Home Decor | 57.64 | 108.86 | 1664.50 | Expedite supply and place the recommended replenishment order |
| 3 | SKU10037 | Kitchenware | 57.47 | 260.77 | 2182.63 | Expedite supply and place the recommended replenishment order |
| 4 | SKU10120 | Beauty | 56.25 | 415.20 | 17496.64 | Expedite supply and place the recommended replenishment order |
| 5 | SKU10144 | Food & Bev | 55.68 | 1203.77 | 20789.02 | Expedite supply and place the recommended replenishment order |
| 6 | SKU10048 | Stationery | 53.84 | 73.47 | 739.84 | Expedite supply and place the recommended replenishment order |
| 7 | SKU10118 | Electronics | 51.55 | 268.15 | 35065.34 | Expedite supply and place the recommended replenishment order |
| 8 | SKU10008 | Home Decor | 51.43 | 98.48 | 542.62 | Expedite supply and place the recommended replenishment order |
| 9 | SKU10022 | Home Decor | 50.69 | 125.42 | 2066.85 | Expedite supply and place the recommended replenishment order |
| 13 | SKU10024 | Home Decor | 49.26 | 189.32 | 3714.49 | Review supplier timing and place or expedite the recommended order |

## Top Overstock Priorities

| Rank | SKU | Category | Overstock Risk | Excess Units | Excess Value | Action |
|---:|---|---|---:|---:|---:|---|
| 10 | SKU10097 | Apparel | 45.38 | 114.80 | 3592.07 | Pause replenishment; consider transfer, promotion or markdown |
| 11 | SKU10088 | Apparel | 36.25 | 55.46 | 1152.99 | Pause replenishment; consider transfer, promotion or markdown |
| 12 | SKU10098 | Apparel | 32.70 | 57.88 | 1489.25 | Pause replenishment; consider transfer, promotion or markdown |
| 89 | SKU10132 | Beauty | 9.11 | 12.29 | 64.91 | Reduce or defer replenishment and monitor weekly demand |
| 91 | SKU10005 | Home Decor | 8.66 | 10.05 | 47.93 | Reduce or defer replenishment and monitor weekly demand |
| 85 | SKU10093 | Apparel | 0.00 | 0.00 | 0.00 | Review supplier timing and place or expedite the recommended order |
| 90 | SKU10032 | Kitchenware | 0.00 | 0.00 | 0.00 | Review supplier timing and place or expedite the recommended order |
| 108 | SKU10027 | Kitchenware | 0.00 | 0.00 | 0.00 | Reduce or defer replenishment and monitor weekly demand |
| 109 | SKU10095 | Apparel | 0.00 | 0.00 | 0.00 | Reduce or defer replenishment and monitor weekly demand |
| 110 | SKU10104 | Apparel | 0.00 | 0.00 | 0.00 | Reduce or defer replenishment and monitor weekly demand |

## Interpretation Limitation

The available prediction file contains historical test-period model outputs rather than a newly generated future forecast. The latest eight available model predictions are therefore averaged as an operational weekly-demand proxy for this risk-scoring demonstration.
The scoring framework is production-ready in structure, but a live deployment should replace this proxy with the current 8-week future forecast on every scoring run.
