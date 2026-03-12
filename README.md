# Car Price Estimation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Databricks](https://img.shields.io/badge/Platform-Databricks-orange)
![PyCaret](https://img.shields.io/badge/AutoML-PyCaret%203.x-green)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

A fully automated, end-to-end machine learning system for estimating used car prices
for a local dealership. Built on Databricks, the pipeline ingests raw vehicle data,
cleans and enriches it, trains and tunes regression models automatically, explains
individual predictions to business stakeholders, and delivers a ready-to-use valuation
function.

**Business Goal:** Replace manual/gut-feel pricing with a data-driven model that reduces
over- and under-pricing, improves margin, and speeds up inventory turnover.

---

## Architecture & Data Flow

```
Databricks Table (car_price_assignment)
         |
         v
01_Data_Gathering       <-- PySpark load, schema check, Parquet export
         |
         v
02_Data_Cleaning        <-- Missing values, outliers, type fixes, dedup
         |
         v
03_EDA                  <-- Distributions, correlations, grouped analyses
         |
         v
04_Feature_Engineering  <-- Car age, mileage bins, interactions, log-price
         |
         v
05_Preprocessing_Dataset<-- Train/test split, PyCaret setup, MLflow logging
         |
         v
06_3Models_Training     <-- AutoML compare, tune, evaluate, SHAP, LIME, MLflow
         |
         v
07_Valuation            <-- Load model, predict_price(), batch/widget predictions
```

**DBFS Storage Locations:**
- `/tmp/car_price/raw_data.parquet` - Raw PySpark export
- `/tmp/car_price/clean_data.parquet` - After cleaning
- `/tmp/car_price/featured_data.parquet` - After feature engineering
- `/tmp/car_price/train_data.parquet` - Training split
- `/tmp/car_price/test_data.parquet` - Test split
- `/tmp/car_price/final_model` - Saved PyCaret model pipeline

---

## Repository Structure

```
car-price-estimation/
├── notebooks/
│   ├── 01_Data_Gathering.py
│   ├── 02_Data_Cleaning.py
│   ├── 03_EDA.py
│   ├── 04_Feature_Engineering.py
│   ├── 05_Preprocessing_Dataset.py
│   ├── 06_3Models_Training.py
│   └── 07_Valuation.py
├── README.md
├── requirements.txt
└── config.yml
```

---

## Technical Stack

| Layer | Technology |
|-------|------------|
| Platform | Databricks (DBR 14+) |
| Data Processing | PySpark + Pandas |
| AutoML / Modelling | PyCaret 3.x |
| Interpretability | SHAP 0.44+, LIME |
| Experiment Tracking | MLflow (integrated) |
| Orchestration | Databricks Workflows |
| Version Control | GitHub (Databricks Repos) |

---

## Dataset

**Table:** `car_price_assignment` (Databricks workspace)

| Column | Type | Description |
|--------|------|-------------|
| Make | string | Car manufacturer |
| Model | string | Model name |
| Price | float | Target variable (USD) |
| Year | int | Model year |
| Condition | string | Car condition rating |
| Mileage | float | Odometer reading (km) |
| Fuel Type | string | Petrol / Diesel / Electric / Hybrid |
| Volume | float | Engine displacement (cc) |
| Color | string | Exterior colour |
| Transmission | string | Manual / Automatic |
| Drive Unit | string | FWD / RWD / AWD |
| Segment | string | Market segment (manually entered) |

---

## KPIs

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error (USD) |
| RMSE | Root Mean Squared Error (USD) |
| R² | Coefficient of Determination |
| MAPE | Mean Absolute Percentage Error |

---

## Setup Instructions

### 1. Prerequisites

- Databricks workspace with DBR 14.0+ cluster
- Table `car_price_assignment` loaded in the `default` database
- GitHub account with Personal Access Token (PAT)

### 2. Connect Databricks Repos to GitHub

```bash
# In Databricks workspace:
# 1. Go to Repos > Add Repo
# 2. Enter GitHub URL: https://github.com/hgabrali/car-price-estimation
# 3. Authenticate with your GitHub PAT
# 4. Clone the repo - notebooks appear under /Repos/hgabrali/car-price-estimation
```

### 3. Install Libraries on Cluster

Option A - via cluster UI (Compute > Libraries):
```
pycaret==3.3.2
shap>=0.44.0
lime>=0.2.0.1
lightgbm>=4.0.0
xgboost>=2.0.0
optuna>=3.4.0
```

Option B - add to notebook 05 or 06 init cell:
```python
%pip install pycaret==3.3.2 shap lime lightgbm xgboost optuna --quiet
dbutils.library.restartPython()
```

### 4. Configure MLflow Experiment

The experiment path `/Users/car-price-estimation/car_price_pipeline` is created
automatically. To change it, update `MLFLOW_EXP_NAME` in notebooks 05 and 06.

---

## Running the Pipeline

### Manual (Notebook by Notebook)

Run notebooks in order from your Databricks workspace:
```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

### Automated (Databricks Workflow)

1. Go to **Workflows > Create Job**
2. Add 7 tasks (one per notebook) in sequence
3. Set each task to depend on the previous one
4. Configure notebook paths (from Repos):

```yaml
tasks:
  - task_key: data_gathering
    notebook_task:
      notebook_path: /Repos/hgabrali/car-price-estimation/notebooks/01_Data_Gathering
  - task_key: data_cleaning
    depends_on: [data_gathering]
    notebook_task:
      notebook_path: /Repos/hgabrali/car-price-estimation/notebooks/02_Data_Cleaning
  # ... (repeat for 03-07)
```

### Valuation via Widget

```python
# Pass a car's attributes as JSON to notebook 07:
car_json = '{"make": "Toyota", "model": "Corolla", "year": 2020, "condition": "Good",
             "mileage": 45000, "fuel_type": "Petrol", "volume": 1800,
             "color": "Blue", "transmission": "Automatic",
             "drive_unit": "FWD", "segment": "Sedan"}'
```

---

## Local Development

```bash
git clone https://github.com/hgabrali/car-price-estimation.git
cd car-price-estimation
pip install -r requirements.txt
```

> Note: Full execution requires a Databricks cluster. For local testing,
> mock `spark`, `dbutils`, and `display` as needed.

---

## MLflow Experiment Tracking

Every training run logs:
- Model hyperparameters
- MAE, RMSE, R², MAPE metrics
- Trained model artifact (registered as `CarPriceRegressor`)
- Actual vs Predicted scatter plot
- SHAP summary and importance plots

Access via: **Databricks > Machine Learning > Experiments**

---

## Model Interpretability

- **SHAP:** Global feature importance (beeswarm + bar plots) using TreeExplainer
  or KernelExplainer fallback.
- **LIME:** Local explanations for individual predictions, showing which features
  raised or lowered the price estimate for that specific car.

---

## Author

Hande Gabriali-Knobloch | [GitHub](https://github.com/hgabrali) |
[LinkedIn](https://www.linkedin.com/in/hande-gabriali-knobloch-5b176615)
