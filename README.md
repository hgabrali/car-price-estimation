# Car Price Estimation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Databricks](https://img.shields.io/badge/Platform-Databricks-orange)
![sklearn](https://img.shields.io/badge/ML-scikit--learn-blue)
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
05_Preprocessing_Dataset<-- Train/test split, Spark table save, MLflow logging
    |
    v
06_3Models_Training     <-- sklearn model compare, tune, evaluate, SHAP, LIME, MLflow
    |
    v
07_Valuation            <-- Load model, predict_price(), batch/widget predictions
```

**Storage Locations:**
- `car_price_train` / `car_price_test` - Spark tables (train/test split)
- `/Workspace/Users/.../models/final_model.pkl` - Saved sklearn Pipeline model
- `/Workspace/Users/.../plots/` - Evaluation and SHAP/LIME plots

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
├── LICENSE
├── requirements.txt
└── config.yml
```

---

## Technical Stack

| Layer | Technology |
|-------|------------|
| Platform | Databricks (DBR 14+, Free Edition compatible) |
| Data Processing | PySpark + Pandas |
| Modelling | scikit-learn Pipelines (Ridge, RandomForest, GradientBoosting) |
| Hyperparameter Tuning | GridSearchCV |
| Interpretability | SHAP 0.44+, LIME |
| Experiment Tracking | MLflow (with Free Edition fallback) |
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

## Model Results

| Metric | Value |
|--------|-------|
| Model | Random Forest (sklearn Pipeline) |
| MAE | $1,376 |
| RMSE | $1,915 |
| R² | 0.9536 (95.4% variance explained) |
| MAPE | 10.02% |

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

- Databricks workspace with DBR 14.0+ cluster (Free Edition supported)
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
shap>=0.44.0
lime>=0.2.0.1
numpy<2
```

Option B - add to notebook 06 init cell (already included):
```python
%pip install shap lime "numpy<2" --quiet
dbutils.library.restartPython()
```

### 4. Configure MLflow Experiment

The experiment path `/Users/hande.gabrali@gmail.com/car_price_pipeline` is created automatically.
To change it, update `MLFLOW_EXP_NAME` in notebooks 05 and 06.

> **Note:** On Databricks Free Edition, MLflow Model Registry is not available.
> The notebook automatically falls back to saving the model locally via `joblib`
> when MLflow logging fails.

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
- Trained model artifact (sklearn Pipeline)
- Actual vs Predicted scatter plot
- SHAP summary and importance plots

Access via: **Databricks > Machine Learning > Experiments**

> **Free Edition Note:** If MLflow Model Registry is unavailable, the model is
> automatically saved locally as a `.pkl` file via `joblib`. All metrics and
> parameters are still logged where possible.

---

## Model Interpretability

- **SHAP:** Global feature importance (beeswarm + bar plots) using TreeExplainer. Key findings: curbweight and enginesize are the strongest price predictors; horsepower and car dimensions are significant positive drivers; fuel efficiency negatively correlates with price.
- **LIME:** Local explanations for individual predictions, showing which features raised or lowered the price estimate for that specific car.

---

## Agent Audit Framework

Following the AI Agent audit methodology for Data Analysts/Scientists:

| Audit Dimension | Assessment |
|-----------------|------------|
| **Logical Validation** | Random Forest validated against Ridge and Gradient Boosting via cross-validation |
| **Hallucination Detection** | SHAP and LIME confirm the model relies on physically meaningful features |
| **Information Sources** | Verified CarPrice_Assignment dataset with 205 records and 26 features |
| **Error Margins** | MAPE ~10% is acceptable for dealership pricing guidance |
| **Business Translation** | Data-driven pricing replaces gut-feel estimates with quantified predictions |
| **Strategic Decision** | 10% MAPE balances over-pricing (losing customers) vs under-pricing (losing margin) |

---

## Author

Hande Gabrali-Knobloch | [GitHub](https://github.com/hgabrali) | [LinkedIn](https://www.linkedin.com/in/hande-gabral%C4%B1-knobloch/)
