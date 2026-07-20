# Project FORESIGHT

## Demand Forecasting and Inventory Intelligence Platform

Project FORESIGHT is a machine-learning-powered demand forecasting and inventory intelligence platform designed to improve inventory planning and supply-chain decisions.

The system forecasts weekly SKU-level demand, evaluates forecasting models, identifies stockout and overstock risks, estimates financial exposure, recommends inventory actions, and presents the results through an interactive Streamlit dashboard.

## Live Application

[Open the Project FORESIGHT Dashboard](https://project-foresight-szbpqvrcxp8x6sbrf8qmzx.streamlit.app/)

## GitHub Repository

[Project FORESIGHT on GitHub](https://github.com/kundanad2102-ui/project-foresight)

---

## Business Problem

Businesses frequently face two major inventory problems:

- Popular products run out of stock, causing lost sales and reduced customer satisfaction.
- Slow-moving products remain in inventory, locking working capital and increasing carrying costs.

Project FORESIGHT converts historical sales, product, inventory, promotion and calendar data into actionable demand and inventory recommendations.

---

## Project Objectives

- Build a reproducible data-processing pipeline.
- Create an analysis-ready SKU-level dataset.
- Analyse demand patterns, promotions, stockouts and product movement.
- Forecast weekly demand for every SKU.
- Compare machine-learning models with naive forecasting baselines.
- Use rolling-origin cross-validation for leakage-safe model evaluation.
- Identify stockout and overstock risks.
- Estimate potential lost revenue and excess inventory value.
- Recommend replenishment, expediting, monitoring, markdown or transfer actions.
- Build and deploy an interactive Streamlit decision dashboard.

---

## Project Status

| Project Stage | Status |
|---|---|
| Week 1 – Data Foundation | Completed |
| Week 2 – EDA and Baseline Forecasting | Completed |
| Week 3 – Forecast Modelling and Risk Scoring | Completed |
| Week 4 – Streamlit Dashboard and Deployment | Completed |
| Streamlit Community Cloud Deployment | Completed |
| FastAPI Scoring-Service Deployment | Planned |

---

## Technology Stack

- Python 3.11
- pandas
- NumPy
- scikit-learn
- matplotlib
- Plotly
- Streamlit
- FastAPI
- Uvicorn
- Joblib
- Jupyter Notebook
- Pytest
- Git and GitHub

---

## Project Structure

```text
project-foresight/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── final_forecast_model.joblib
│
├── notebooks/
│   ├── 00_data_profile.ipynb
│   └── 01_eda.ipynb
│
├── reports/
│   ├── data_mapping.md
│   ├── data_quality_decisions.md
│   ├── eda_insight_memo.md
│   ├── week3_model_report.md
│   ├── error_analysis_summary.md
│   ├── rolling_origin_cv_report.md
│   ├── inventory_risk_report.md
│   └── progress_log.md
│
├── service/
│
├── src/
│   ├── pipeline.py
│   ├── features.py
│   ├── baseline.py
│   ├── model.py
│   ├── rolling_cv.py
│   ├── error_analysis.py
│   └── risk_scoring.py
│
├── tests/
├── requirements.txt
└── README.md
```

---

## Data Foundation

The data pipeline transforms raw sales, product and inventory extracts into standardised analytical datasets.

### Main Processed Datasets

| Dataset | Purpose |
|---|---|
| `sales_daily.csv` | Daily SKU-level sales and estimated demand |
| `sku_master.csv` | Product, category, cost, price and supplier information |
| `inventory_snapshots.csv` | Daily inventory and stock-position information |
| `calendar.csv` | Calendar and promotion attributes |
| `analysis_ready.csv` | Combined daily analysis dataset |
| `weekly_demand.csv` | Weekly SKU-level modelling dataset |
| `model_predictions.csv` | Final-test model and baseline forecasts |
| `inventory_risk_scores.csv` | SKU-level inventory risks and recommendations |

### Data Volume

- 150 SKUs
- 731 daily dates
- 109,650 daily SKU records
- 15,900 weekly SKU records
- 104 complete modelling weeks

### Forecasting Target

The forecasting target is:

```text
demand_target = inventory units sold + lost sales
```

This provides an estimate of underlying demand rather than relying only on fulfilled sales.

---

## Exploratory Data Analysis

The EDA workflow includes:

- Demand distributions
- Zero-demand analysis
- Daily, weekly, monthly and yearly trends
- Top-moving and slow-moving SKUs
- Demand concentration
- Category performance
- Demand intermittency and variability
- Stockout and lost-sales analysis
- Promotion-demand comparison
- Dead-stock analysis
- Demand-driver correlation analysis

### Promotion Analysis

- Average demand during promotional observations: **61.01 units**
- Average demand during non-promotional observations: **8.22 units**
- Observed promotion-demand difference: **642.12%**

This is treated as an observed association and not proof that promotion alone caused the full increase.

### Dead-Stock Analysis

Dead stock was defined as positive ending inventory with zero estimated demand during the latest 13 weeks.

- Dead-stock SKUs: **0**
- Dead-stock units: **0**
- Capital tied up in dead stock: **0.00**

Slow-moving and intermittent products were present, but no SKU met the defined dead-stock condition.

---

## Feature Engineering

Leakage-safe weekly features include:

- Demand lags: 1, 2, 4, 8, 13, 26 and 52 weeks
- Rolling demand means
- Rolling demand standard deviations
- Zero-demand rates
- Promotion indicators
- Calendar features
- Category and product attributes
- Previous-week inventory features
- Previous-week stockout and lost-sales features

All forecasting features use information available before the forecasted week.

---

## Baseline Forecasting

Two baseline methods were evaluated:

| Baseline | WAPE | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Previous-Week Naive | 47.88% | 69.08 | 121.54 | -1.05% |
| Seasonal Naive – 52 Weeks | 53.30% | 76.89 | 143.16 | +0.23% |

The 52-week seasonal-naive model was retained as the specification baseline, while the previous-week naive model was retained as the stronger operational benchmark.

---

## Forecast Modelling

The following machine-learning models were evaluated:

- HistGradientBoosting
- Random Forest

### Final-Test Results

| Model | WAPE | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 39.51% | 57.01 | 96.10 | +2.94% |
| Previous-Week Naive | 47.88% | 69.08 | 121.54 | -1.05% |
| Seasonal Naive – 52 Weeks | 53.30% | 76.89 | 143.16 | +0.23% |

HistGradientBoosting achieved the best final-test performance.

It improved WAPE by:

- **8.37 percentage points** over the previous-week naive benchmark
- **13.79 percentage points** over the 52-week seasonal-naive baseline

---

## Rolling-Origin Cross-Validation

Rolling-origin cross-validation was performed using:

- 4 expanding-window folds
- 8-week forecast horizon per fold
- Minimum 52-week training history
- Chronological training and validation periods
- An untouched final test period

### Rolling-CV Results

| Model | WAPE | Mean Fold WAPE | Bias |
|---|---:|---:|---:|
| HistGradientBoosting | 44.58% | 44.60% | +2.27% |
| Random Forest | 45.19% | 45.26% | +9.21% |
| Previous-Week Naive | 51.16% | 51.19% | -0.47% |
| Seasonal Naive – 52 Weeks | 74.45% | 74.04% | -63.76% |

HistGradientBoosting was retained as the selected forecasting model.

---

## Feature Importance

The most influential forecasting features included:

1. 4-week rolling demand mean
2. 8-week rolling demand mean
3. 13-week rolling demand mean
4. Previous-week on-order inventory
5. Rolling demand variability
6. Previous-week demand
7. Previous-week on-hand inventory
8. 52-week demand lag

The results show that recent demand history, medium-term demand behaviour and inventory-planning signals are important forecasting inputs.

---

## Inventory Risk Scoring

Forecast outputs were combined with inventory position, supplier lead time, demand variability, safety stock, product cost, selling price and minimum order quantity.

### Risk Logic

- Inventory position equals ending on-hand units plus ending on-order units.
- Stockout risk compares inventory position with lead-time forecast demand plus safety stock.
- Overstock risk compares inventory position with the 8-week planning requirement plus safety stock.
- Recommended order quantities are rounded to the SKU minimum order quantity.

### Stockout Risk Levels

- **High:** shortage score is at least 50%
- **Medium:** projected shortage exists but is below 50%
- **Low:** no projected lead-time shortage

### Risk-Scoring Results

| Metric | Result |
|---|---:|
| SKUs scored | 150 |
| High stockout-risk SKUs | 9 |
| Medium stockout-risk SKUs | 93 |
| High overstock-risk SKUs | 3 |
| Medium overstock-risk SKUs | 8 |
| Forecast stockout gap | 15,220.67 units |
| Potential lost revenue | 793,250.16 |
| Excess inventory | 250.48 units |
| Excess inventory value | 6,347.14 |
| Recommended replenishment | 40,550 units |
| Recommended replenishment cost | 1,128,180.40 |

Monetary values are shown in the original dataset currency units.

### Recommended Actions

The platform generates SKU-level actions such as:

- Expedite supply
- Place a replenishment order
- Review supplier timing
- Reduce or defer replenishment
- Transfer excess stock
- Apply promotion or markdown
- Maintain the current inventory plan
- Monitor weekly demand

---

## Streamlit Dashboard

The deployed dashboard contains five main sections.

### Executive Overview

- SKU counts
- Stockout-risk distribution
- Overstock-risk distribution
- Potential lost revenue
- Excess inventory value
- Recommended replenishment cost
- Category-level value exposure

### Risk Priorities

- Top stockout-risk SKUs
- Top overstock-risk SKUs
- Value-at-stake charts
- Priority tables
- Downloadable CSV reports

### Forecast Performance

- Final-test metrics
- Model WAPE comparison
- Rolling-origin validation results
- Weekly actual-demand versus forecast chart

### SKU Explorer

- SKU-level risk classification
- Weekly forecast demand
- Recommended order quantity
- Value at stake
- Recommended action
- Actual-demand versus forecast chart
- SKU-level WAPE and MAE

### Methodology

- Forecasting method
- Leakage-prevention approach
- Inventory-risk logic
- Model limitations
- Dashboard source files

### Dashboard Filters

Users can filter by:

- Category
- Supplier
- Stockout-risk level
- Overstock-risk level
- Priority rank

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/kundanad2102-ui/project-foresight.git
cd project-foresight
```

### 2. Create the virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## Run the Complete Workflow

Run the commands from the project root.

### Data pipeline

```powershell
python -m src.pipeline
```

### Feature engineering

```powershell
python -m src.features
```

### Baseline forecasting

```powershell
python -m src.baseline
```

### Model training and evaluation

```powershell
python -m src.model
```

### Rolling-origin cross-validation

```powershell
python -m src.rolling_cv
```

### Error analysis

```powershell
python -m src.error_analysis
```

### Inventory risk scoring

```powershell
python -m src.risk_scoring
```

### Streamlit dashboard

```powershell
python -m streamlit run app/dashboard.py
```

The local dashboard will normally open at:

```text
http://localhost:8501
```

---

## Testing

Run the automated tests with:

```powershell
python -m pytest
```

---

## Deployment

The Streamlit dashboard is deployed using Streamlit Community Cloud.

Live application:

[https://project-foresight-szbpqvrcxp8x6sbrf8qmzx.streamlit.app/](https://project-foresight-szbpqvrcxp8x6sbrf8qmzx.streamlit.app/)

Deployment configuration:

```text
Repository: kundanad2102-ui/project-foresight
Branch: main
Main file: app/dashboard.py
```

---

## Important Limitation

The current inventory-risk demonstration uses the latest available historical test-period model predictions as an operational weekly-demand proxy.

A production deployment should generate a new 8-week future forecast on every scoring run and pass that forecast directly to the inventory-risk scoring workflow.

---

## Future Improvements

- Generate true recursive 8-week future forecasts.
- Deploy the FastAPI scoring service.
- Add automated scheduled model retraining.
- Add prediction intervals and uncertainty bands.
- Add service-level optimisation.
- Add supplier performance monitoring.
- Add authentication and organisation-level access.
- Add automated data refresh and dashboard updates.
- Integrate cloud storage and database services.

---

## Project Outcome

Project FORESIGHT demonstrates an end-to-end machine-learning workflow that converts raw operational data into:

- Clean analytical datasets
- Demand forecasts
- Leakage-safe model validation
- Stockout and overstock risk scores
- Financial-impact estimates
- SKU-level inventory recommendations
- An interactive deployed decision dashboard