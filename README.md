# Project FORESIGHT

## Demand and Inventory Intelligence Platform

Project FORESIGHT is an AI-powered demand forecasting and inventory
intelligence platform designed to improve inventory planning and
supply-chain decisions.

The system will forecast weekly SKU-level demand, identify stockout and
overstock risks, calculate financial impact, and recommend inventory actions
through an interactive dashboard.

## Business Problem

Businesses frequently face two inventory problems:

1. Popular products run out of stock, causing lost sales.
2. Slow-moving products remain in inventory, locking working capital.

## Project Objectives

- Build a reproducible data-processing pipeline.
- Forecast weekly demand for every SKU.
- Compare the model with a seasonal-naive baseline.
- Identify stockout and overstock risks.
- Calculate sales at risk and capital locked in excess inventory.
- Recommend reorder, markdown, watch, or healthy actions.
- Develop a Streamlit planning dashboard.
- Deploy a FastAPI scoring service.

## Technology Stack

- Python 3.11
- pandas
- NumPy
- scikit-learn
- matplotlib
- Plotly
- Streamlit
- FastAPI
- Git and GitHub

## Project Structure

- `data/raw/` – original datasets
- `data/processed/` – cleaned and generated datasets
- `notebooks/` – profiling, EDA and model evaluation
- `src/` – pipeline, features, model and risk-scoring code
- `app/` – Streamlit dashboard
- `service/` – FastAPI scoring service
- `models/` – trained model files
- `reports/` – data-quality and executive reports
- `tests/` – automated tests

## Current Progress

### Week 1 – Data Foundation

- [x] Project folder created
- [x] Python 3.11 virtual environment created
- [x] Required packages installed
- [ ] Raw datasets added
- [ ] Data profiling completed
- [ ] Data-cleaning pipeline completed
- [ ] Analysis-ready dataset generated
- [ ] Data-quality report completed