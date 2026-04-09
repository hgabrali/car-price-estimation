# Car Price Estimation System

<img width="1280" height="400" alt="Car Price Estimation System - ML Pipeline Banner" src="https://github.com/user-attachments/assets/16323393-9858-4710-87bf-737b664af739" />


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

## 🚗 Regression Analysis Results

> **Dataset:** `default.car_price_train` | 152 rows, 24 numeric features | Train: 121 / Test: 31 | `random_state=42`

---

### 📈 Linear Regression Results

| Metric | Value |
|--------|-------|
| R² (R-squared) | 0.9936 |
| MSE | 226,113.39 |
| RMSE | 475.51 |
| MAE | 348.81 |
| Intercept | 12,620.12 |

**Top 10 Feature Coefficients (by absolute value):**

| Feature | Coefficient |
|---------|-------------|
| price_per_weight | +4,462.77 |
| power_to_weight | −2,129.00 |
| enginesize | +1,885.17 |
| hp_per_cc | +1,432.43 |
| compression_per_cyl | −899.22 |
| horsepower | +885.33 |
| curbweight | +735.32 |
| citympg | +640.96 |
| compressionratio | +620.46 |
| brand_avg_price | +566.43 |

> **Interpretation:** Linear Regression achieves an outstanding R² of 0.9936, meaning it explains 99.36% of price variance. The most influential predictor is `price_per_weight` (a derived ratio feature), followed by `power_to_weight` and `enginesize`. The negative coefficient on `power_to_weight` suggests multicollinearity with correlated features.

---

### 🌳 Decision Tree Regressor Results (max_depth=5)

| Metric | Value |
|--------|-------|
| R² (R-squared) | 0.8771 |
| MSE | 4,318,768.05 |
| RMSE | 2,078.16 |
| MAE | 1,478.92 |
| Max Depth | 5 |

**Top 10 Feature Importances:**

| Feature | Importance |
|---------|-----------|
| price_per_weight | 0.6975 |
| brand_avg_price | 0.1936 |
| curbweight | 0.0953 |
| avg_mpg | 0.0059 |
| carlength | 0.0018 |
| boreratio | 0.0015 |
| symboling | 0.0011 |
| enginesize | 0.0009 |
| highwaympg | 0.0008 |
| horsepower | 0.0007 |

> **Interpretation:** The Decision Tree (depth=5) achieves a solid R² of 0.8771 (87.71%), but significantly higher error values compared to Linear Regression. The model is almost entirely driven by `price_per_weight` (69.7% importance), followed by `brand_avg_price` (19.4%), confirming these engineered features are highly predictive.

---

### 📊 Model Comparison Summary

| Model | R² | MSE | RMSE | MAE |
|-------|----|-----|------|-----|
| Linear Regression | 0.9936 | 226,113 | 475.51 | 348.81 |
| Decision Tree (depth=5) | 0.8771 | 4,318,768 | 2,078.16 | 1,478.92 |

> **Conclusion:** Linear Regression clearly outperforms the Decision Tree on all metrics for this dataset. The very high R² (0.99) of Linear Regression suggests a strong linear relationship between the engineered features and car price. The Decision Tree, while interpretable and capable at 87.7% R², is more prone to overfitting due to the tree splits and produces considerably larger prediction errors (~4x higher RMSE). For production deployment, Linear Regression is the preferred model unless non-linearity or outlier robustness is a priority.

---

---

## Notebook 06 — Model Training, Evaluation & Interpretability

> **PyCaret Integration for Rapid Prototyping — `06_3Models_Training`**

### Overview

This notebook is **Step 6 of 7** in the car price estimation ML pipeline. Its declared purpose is to train multiple regression models using a PyCaret-style AutoML comparison workflow (implemented via scikit-learn pipelines), tune the best-performing model, evaluate it on a held-out test set, and generate interpretability artifacts using SHAP and LIME. All experiments were intended to be tracked via MLflow.

- **Inputs:** Preprocessed Train/Test Parquet tables (`car_price_train`, `car_price_test`) produced by Notebook 05.
- **Outputs:** Saved model artifact (`final_model.pkl`) + evaluation metrics + interpretation plots.

---

### 1. Data Loading & Split

| Property | Value |
|----------|-------|
| Training set | 152 rows |
| Test set | 39 rows (approx. 80/20 split) |
| Total feature columns | 38 |
| Regression target | `price` |
| Numerical features | 25 |
| Categorical features | 13 |

---

### 2. Preprocessing Pipeline

A `ColumnTransformer`-based preprocessing scheme was constructed:

- **Numerical features:** Median imputation (`SimpleImputer`) → Standard scaling (`StandardScaler`)
- **Categorical features:** Most-frequent imputation → One-Hot Encoding (`OneHotEncoder`, `handle_unknown='ignore'`)

---

### 3. Model Comparison (PyCaret-style AutoML)

Three `sklearn` Pipeline models were evaluated via **5-fold cross-validated RMSE**:

| Model | CV RMSE (USD) |
|-------|--------------|
| Ridge Regression (α=10.0) | $908.44 ✅ **Best** |
| Gradient Boosting (n=100) | $930.08 |
| Random Forest (n=100) | $1,150.98 |

> Ridge Regression was selected as the best base model.

---

### 4. Hyperparameter Tuning

A `GridSearchCV` (3-fold CV) was applied to the Ridge model with α ∈ {1.0, 10.0, 100.0}:

| Parameter | Value |
|-----------|-------|
| Best alpha | 1.0 |
| Tuned CV RMSE | $542.34 |
| Untuned CV RMSE | $908.44 |
| Improvement | ~40% reduction |

---

### 5. Final Test Set Evaluation

The tuned Ridge pipeline was fitted on the full training set and evaluated on the 39-sample test set:

| Metric | Value |
|--------|-------|
| MAE | $733.38 |
| RMSE | $1,187.45 |
| R² | 0.9714 (97.14% variance explained) |
| MAPE | 5.58% |

> The model achieves excellent generalization — R² of 0.97 on the test set indicates strong predictive power, and a MAPE of ~5.6% is well within acceptable bounds for dealership pricing guidance.

> **Note:** The Business Summary section at the bottom of the notebook references earlier/alternative model run metrics (MAE $1,376, RMSE $1,915, R² 0.9536, MAPE 10.02%) that correspond to a Random Forest result — likely an earlier experimental run. The definitive, current execution results are those reported above for the tuned Ridge model.

---

### 6. Model Persistence & MLflow Logging

MLflow tracking was attempted (`mlflow.set_experiment`, `mlflow.start_run`), but failed due to Databricks Free Edition limitations — the `spark.mlflow.modelRegistryUri` configuration is unavailable in this tier (`CONFIG_NOT_AVAILABLE / SQLSTATE: 42K0I`). As a fallback, the model was serialized locally via `joblib`:

```
/Workspace/Users/hande.gabrali@gmail.com/car-price-estimation/models/final_model.pkl
```

---

### 7. SHAP — Global Feature Importance

A SHAP `Explainer` was applied to the Ridge model's inner estimator on the transformed training data (82 features post-OHE). Two plots were generated and persisted:

- `shap_summary.png` — beeswarm plot showing value distribution and directional impact of the top 15 features
- `shap_importance.png` — bar chart of mean absolute SHAP values

**Key SHAP findings:**

- `curbweight` and `enginesize` are the strongest positive price predictors
- `horsepower`, `carlength`, `carwidth` are significant positive drivers
- `citympg` and `highwaympg` show negative correlation with price (economy vs. luxury trade-off)

---

### 8. LIME — Local Prediction Explanations

A `LimeTabularExplainer` (regression mode) was used to explain three individual test-set predictions (indices 0, 19, 38):

| Sample | Actual Price | Predicted Price |
|--------|-------------|----------------|
| #0 | $8,238 | $8,343 |
| #19 | $11,048 | $10,674 |
| #38 | $16,845 | $17,367 |

LIME plots were saved as `lime_sample_0.png`, `lime_sample_19.png`, `lime_sample_38.png`. The close alignment between actual and predicted values across the price range confirms model stability.

---

### 9. Notable Issues & Observations

| Issue | Description |
|-------|-------------|
| `threadpoolctl` AttributeError | A known harmless compatibility warning on certain Databricks runtimes (`NoneType.split` on `get_version()`). |
| `SimpleImputer` UserWarning | The `car_id` column contains no observed (non-null) values in the training set; median imputation was silently skipped. This column is an identifier and should be dropped in upstream preprocessing. |
| Py4J / gRPC connection resets | The Spark driver connection was temporarily dropped during GridSearchCV (a common occurrence on Free Edition serverless clusters due to idle timeouts). Computation completed successfully despite these transient errors. |
| MLflow registry unavailable | Free Edition Databricks does not expose `spark.mlflow.modelRegistryUri`; the pipeline gracefully falls back to local `joblib` serialization. |
| Business Summary metric discrepancy | Cell 12 reports Random Forest metrics (R²=0.95, MAPE=10%) rather than the final tuned Ridge metrics (R²=0.97, MAPE=5.58%), indicating the summary was written during an earlier experiment iteration and was not updated after the final run. |

---

### Summary

The notebook successfully implements a rapid model prototyping workflow: automated multi-model comparison → hyperparameter tuning → held-out evaluation → explainability (SHAP + LIME) → model serialization. The final **Ridge Regression model (α=1.0)** achieves an **R² of 0.9714** and **MAPE of 5.58%** on the test set, making it production-ready for car price estimation tasks within the defined error tolerance.

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

---

### ⚠️ Known Compatibility Issues & Workarounds

#### 1. PyCaret incompatibility with Python 3.12

PyCaret (2024 release) does **not** support Python 3.12.x due to unresolved dependency conflicts in its dependency tree (e.g., `scikit-learn`, `numba`, `catboost` pinning issues). The library **only runs stably on Python 3.11.x**.

**Workaround applied in this project:**
The Databricks cluster runtime was configured to use **Python 3.11** at the project level, effectively downgrading from the default Python 3.12.3 environment. This was achieved by selecting a DBR (Databricks Runtime) version that ships with Python 3.11 (e.g., DBR 14.x LTS), rather than the latest Python 3.12-based runtime.

> **Note for reproducibility:** If you encounter `ImportError` or version conflict errors when installing PyCaret, verify that your cluster is running **Python 3.11.x** (`python --version`). Do **not** use Python 3.12.x with PyCaret.

#### 2. SHAP API change — `TreeExplainer` instead of `Explainer`

In newer versions of SHAP (0.44+), using the generic `shap.Explainer` on tree-based models (Random Forest, Gradient Boosting, XGBoost) can raise `TypeError` or produce incorrect masker/background behavior.

**Workaround applied in this project:**
`shap.TreeExplainer` was used explicitly instead of the generic `shap.Explainer`. `TreeExplainer` is optimized for tree-based models, uses fast exact computation (no sampling), and is the correct API for sklearn `Pipeline` objects wrapping ensemble estimators.

```python
# ✅ Correct — used in this project
explainer = shap.TreeExplainer(final_model.named_steps['model'])

# ❌ Avoid — generic Explainer causes issues with tree models in newer SHAP
# explainer = shap.Explainer(final_model)
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

![Databricks Workflow Runs](https://github.com/user-attachments/assets/112009ad-98fb-4a33-9859-03efda202bb7)

#### Databricks Job Run History – Technical Analysis

The image above shows the **Run History** of a Databricks Workflow consisting of 7 sequential tasks:
`Gathering → Cleaning → EDA → Feature_Engineering → Preprocessing → Model_Training → Valuation`

**Color Legend:**

| Color | Meaning |
|-------|---------|
| 🟩 Green (solid) | Succeeded |
| 🟥 Dark Red / Hatched | Failed |
| 🌸 Light Pink | Skipped (upstream task failed) |

**Problems Identified:**

1. **Cascading Task Failures (Upstream Dependency Failures)** — In the early runs, `Gathering` and `Cleaning` tasks failed (dark red/hatched), causing all downstream tasks to be skipped (light pink). This is a classic upstream dependency failure cascade — when a parent task fails, all child tasks are blocked and marked as skipped.

2. **Intermittent / Flaky Task Failures** — Tasks like `Preprocessing` and `Model_Training` show a repeated pattern of alternating failures and successes, indicating non-deterministic or environment-sensitive errors — possibly caused by cluster startup issues, data availability timing (race conditions), or resource contention (OOM errors, timeouts).

3. **Increasing Run Duration Over Time** — The bar chart (Run total duration) shows a progressive increase from fast-failing runs to stabilization around 3m 59s – 7m 59s. Early runs failed fast (short duration = early exit on failure); later runs ran longer as more tasks succeeded — consistent with iterative debugging.

4. **Partial Pipeline Success in Middle Runs** — Several runs show a mixed state where some tasks succeed (green) while others fail (hatched), indicating that fixes were applied incrementally per task.

**Solutions Applied (Inferred from Run History):**

- **Bug fixes applied task-by-task:** The gradual shift from failed → succeeded (left to right in the timeline) across each task row indicates iterative debugging and patching.
- **Dependency/data source fixes for Gathering & Cleaning:** Once these foundational tasks turned green, all downstream tasks were unblocked.
- **Cluster/environment stabilization:** The reduction in `Preprocessing` and `Model_Training` failures over time suggests cluster configuration or library dependency issues were resolved.
- **Final stable run (rightmost green bar):** The last run shows a fully successful end-to-end pipeline execution — all tasks from `Gathering` through `Valuation` completed without error.

> **Summary:** The pipeline went through a debugging and stabilization lifecycle: starting with critical upstream failures causing full pipeline skips, progressing through intermittent per-task failures during incremental fixes, and ultimately reaching a fully green end-to-end run after resolving data ingestion, environment, and task-level code issues.

![Databricks Job Run History](https://github.com/user-attachments/assets/e1542541-5795-411f-80da-22069c64774f)

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
