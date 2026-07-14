# Project FORESIGHT – EDA Insight Memo

## Executive Summary

The analysis covers 150 SKUs across 731 days from 2023 to 2024.  
Total estimated demand was 1,552,211 units.

Demand is highly uneven, intermittent and concentrated among a small number of products. Stockouts caused 95,085 units of lost sales. Demand and revenue roughly doubled in 2024, but stockout days increased faster, indicating that inventory planning did not keep pace with business growth.

## Dataset Coverage

- Date range: 2023-01-01 to 2024-12-31
- Number of days: 731
- Number of SKUs: 150
- Analysis-ready records: 109,650
- Missing values: 0
- Duplicate date-SKU records: 0

## Forecasting Target

The forecasting target is:

`demand_target = inventory_units_sold + lost_sales`

The existing `fulfilled_units_sold` column exactly matched this calculation for all 109,650 records.

Validation results:

- Mismatched rows: 0
- Maximum difference: 0
- Stockout and lost-sales agreement: 100%

Although the source column is named `fulfilled_units_sold`, it represents total estimated demand because it includes lost sales.

## Demand Distribution

Demand is strongly right-skewed and intermittent.

- Average daily SKU demand: 13.29 units
- Median daily SKU demand: 2 units
- 90th percentile: 22 units
- 95th percentile: 61 units
- 99th percentile: 216 units
- Maximum: 1,224 units
- Zero-demand date-SKU records: 40.62%

Because many observations contain zero demand, MAPE is not suitable as the primary forecasting metric. WAPE and MAE will be used instead.

## Demand Trends

Total demand increased from 514,550 units in 2023 to 1,037,661 units in 2024, an increase of approximately 101.66%.

Revenue increased from 25.09 million to 50.04 million, approximately 99.46%.

However:

- Lost sales increased by approximately 114.74%
- Stockout days increased by approximately 135.73%
- Lost-sales rate increased from 5.87% to 6.25%

Inventory availability therefore did not improve at the same rate as demand growth.

## Seasonality and Time Patterns

Tuesday recorded the highest total demand at 242,229 units.

Thursday had the highest average daily demand at approximately 15.44 units.

Sunday recorded the lowest total demand at 180,944 units.

Weekend demand was lower than weekday demand.

Monthly demand was highest during:

- December: 205,378 units
- November: 188,525 units
- October: 171,125 units

Demand increased sharply from September through December. However, this pattern must not be treated as pure seasonality because the dataset also shows strong year-over-year growth.

## Top-Moving SKUs

The highest-demand SKUs were:

1. SKU10100 – BELT LEATHER BLACK: 206,723 units
2. SKU10144 – SPICE MIX CURRY: 114,568 units
3. SKU10045 – CASSEROLE DISH OVAL: 93,410 units
4. SKU10105 – BLUETOOTH SPEAKER PORTABLE: 62,654 units
5. SKU10021 – WIND CHIME BAMBOO: 60,172 units

The top 10 SKUs generated 46.02% of total demand.

The top 20 SKUs generated 59.75% of total demand.

Forecasting and replenishment accuracy for these products will have the greatest business impact.

## Slow-Moving and Intermittent SKUs

The slowest-moving SKU was:

- SKU10104 – PYJAMA SET COTTON
- Total demand: 409 units
- Active sales days: 129
- Lost sales: 0
- Stockout days: 0

This indicates genuinely low demand rather than demand being suppressed by stockouts.

SKU-level variability results:

- Median zero-demand rate: 33.52%
- SKUs with more than 50% zero-demand days: 42
- SKUs with coefficient of variation above 1: 146
- SKUs with coefficient of variation above 2: 121

Most SKUs therefore have volatile or intermittent demand.

## Category Performance

Apparel generated the highest demand at 369,475 units.

Other high-demand categories included:

- Food & Bev: 240,984 units
- Home Decor: 209,788 units
- Electronics: 208,871 units
- Kitchenware: 191,016 units

Beauty had the highest lost-sales rate at approximately 10.46%.

Other high lost-sales rates included:

- Toys: 9.22%
- Stationery: 9.21%
- Home Decor: 8.15%

Home Decor recorded the highest number of stockout days at 726.

## Stockouts and Lost Sales

- Stockout date-SKU records: 3,589
- Stockout rate: 3.27%
- Total demand during stockouts: 192,199 units
- Fulfilled units during stockouts: 97,114 units
- Lost sales: 95,085 units
- Average lost sales per stockout record: 26.49 units

The products with the most stockout days included:

- SKU10009 – CUSHION COVER VINTAGE: 56 days
- SKU10028 – IVORY KNITTED MUG COSY: 55 days
- SKU10109 – PHONE CASE SILICONE: 53 days
- SKU10120 – FACE CREAM MOISTURIZING: 52 days
- SKU10011 – WALL CLOCK ROMAN: 52 days

Stockout frequency and lost-sales volume must both be considered in future inventory-risk scoring.

## Business Insights

1. Demand and revenue approximately doubled in 2024, but stockout days increased by more than 135%, showing that replenishment performance did not keep pace with growth.

2. The top 10 SKUs generate 46.02% of total demand. These products should receive the highest forecasting, safety-stock and monitoring priority.

3. Demand is intermittent for many SKUs. Forty-two SKUs have zero demand on more than half of all observed days, so one forecasting approach may not work equally well for every product.

4. Beauty, Toys and Stationery have relatively high lost-sales rates despite having lower total demand than Apparel. Category-level inventory policies should therefore not be based only on sales volume.

5. Weekend demand is lower than weekday demand, while Tuesday and Thursday are the strongest demand days. This pattern may support day-of-week features in forecasting models.

## Limitations

- No approved external holiday calendar is available, so holiday effects cannot yet be evaluated reliably.
- Category is temporarily reused as subcategory because no separate subcategory field exists.
- The dataset contains only two years, which limits confidence in long-term seasonal conclusions.
- Strong business growth may be mixed with seasonal effects.
- Promotion analysis is observational and does not prove that promotions caused higher demand.

## Forecast Baseline Results

The final 13 complete weeks were used as a time-based test period covering 150 SKUs and 1,950 SKU-week observations.

Two baseline models were evaluated:

| Model | WAPE | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Naive – Previous Week | 47.88% | 69.08 | 121.54 | -1.05% |
| Seasonal Naive – 52 Weeks | 53.30% | 76.89 | 143.16 | 0.23% |

The previous-week naive model was selected because it achieved the lowest WAPE.

Its negative bias indicates slight under-forecasting. The WAPE of 47.88% also confirms that simple historical baselines are not accurate enough for final inventory planning. More advanced forecasting models must improve on this result.