# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Model Training, Evaluation & Interpretability
# MAGIC 
# MAGIC ## Purpose
# MAGIC Train multiple regression models with PyCaret, tune the best model,
# MAGIC evaluate on the test set, and explain predictions using SHAP and LIME.
# MAGIC All experiments are tracked in MLflow.
# MAGIC 
# MAGIC **Pipeline Position:** Step 6 of 7
# MAGIC **Input:** Train/Test Parquet files from notebook 05
# MAGIC **Output:** Saved model artifact + evaluation metrics + interpretation plots

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# %pip install pycaret==3.3.2 shap lime lightgbm xgboost --quiet
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports & Configuration

# COMMAND ----------

import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # headless for Databricks
import shap
import lime
import lime.lime_tabular
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyCaret regression
from pycaret.regression import (
    setup as pycaret_setup,
    compare_models,
    tune_model,
    finalize_model,
    predict_model,
    save_model,
    pull,
    get_config
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

TRAIN_PATH      = '/dbfs/tmp/car_price/train_data.parquet'
TEST_PATH       = '/dbfs/tmp/car_price/test_data.parquet'
MODEL_PATH      = '/dbfs/tmp/car_price/final_model'
TARGET_COL      = 'price'
MLFLOW_EXP_NAME = '/Users/car-price-estimation/car_price_pipeline'
RANDOM_STATE    = 42

logger.info('06_3Models_Training started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Train & Test Data

# COMMAND ----------

train_df = pd.read_parquet(TRAIN_PATH)
test_df  = pd.read_parquet(TEST_PATH)
logger.info('Train: %d rows | Test: %d rows', len(train_df), len(test_df))
print(f'Train shape: {train_df.shape} | Test shape: {test_df.shape}')

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Initialise PyCaret

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXP_NAME)

env = pycaret_setup(
    data              = train_df,
    target            = TARGET_COL,
    session_id        = RANDOM_STATE,
    normalize         = True,
    normalize_method  = 'zscore',
    transformation    = True,
    transformation_method = 'yeo-johnson',
    remove_multicollinearity = True,
    multicollinearity_threshold = 0.9,
    log_experiment    = True,
    experiment_name   = MLFLOW_EXP_NAME,
    verbose           = False
)
logger.info('PyCaret environment initialised')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare Models (PyCaret AutoML)

# COMMAND ----------

logger.info('Starting model comparison...')

# Compare a curated set of regression models
include_models = ['lr', 'ridge', 'lasso', 'en', 'dt', 'rf', 'gbr', 'xgboost', 'lightgbm']

top_models = compare_models(
    include     = include_models,
    n_select    = 3,
    sort        = 'RMSE',
    verbose     = True
)

# Pull the comparison results
comparison_df = pull()
logger.info('Model comparison complete.')
print('\n=== MODEL COMPARISON RESULTS ===')
display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tune Best Model

# COMMAND ----------

best_model_raw = top_models[0]
logger.info('Tuning best model: %s', type(best_model_raw).__name__)

tuned_model = tune_model(
    best_model_raw,
    n_iter        = 50,
    optimize      = 'RMSE',
    search_library = 'optuna',
    search_algorithm = 'tpe',
    verbose        = True
)

tuning_results = pull()
logger.info('Tuning complete.')
print('\n=== TUNED MODEL RESULTS ===')
display(tuning_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Finalise & Evaluate on Test Set

# COMMAND ----------

# Finalise: retrain on 100% training data
final_model = finalize_model(tuned_model)

# Predict on held-out test set
test_predictions = predict_model(final_model, data=X_test)
y_pred = test_predictions['prediction_label']

# Metrics
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

logger.info('MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%', mae, rmse, r2, mape)
print(f'\n=== TEST SET METRICS ===')
print(f'MAE:  ${mae:,.2f}')
print(f'RMSE: ${rmse:,.2f}')
print(f'R2:   {r2:.4f}')
print(f'MAPE: {mape:.2f}%')

# COMMAND ----------

# Actual vs Predicted plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.4, color='#2196F3', s=20)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Price (USD)')
ax.set_ylabel('Predicted Price (USD)')
ax.set_title(f'Actual vs Predicted Prices (R2={r2:.3f})', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('/dbfs/tmp/car_price/actual_vs_predicted.png', dpi=120, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log to MLflow

# COMMAND ----------

with mlflow.start_run(run_name='06_Final_Model') as run:
    mlflow.log_param('model_type',   type(final_model).__name__)
    mlflow.log_param('random_state', RANDOM_STATE)
    mlflow.log_metric('test_mae',  mae)
    mlflow.log_metric('test_rmse', rmse)
    mlflow.log_metric('test_r2',   r2)
    mlflow.log_metric('test_mape', mape)
    mlflow.sklearn.log_model(
        final_model,
        artifact_path='car_price_model',
        registered_model_name='CarPriceRegressor'
    )
    mlflow.log_artifact('/dbfs/tmp/car_price/actual_vs_predicted.png')
    logger.info('Model logged to MLflow. Run ID: %s', run.info.run_id)
    print('MLflow Run ID:', run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save Model (PyCaret)

# COMMAND ----------

save_model(final_model, MODEL_PATH)
logger.info('Model saved to: %s', MODEL_PATH)
print(f'Model saved to: {MODEL_PATH}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. SHAP - Global Feature Importance

# COMMAND ----------

logger.info('Computing SHAP values...')

# Get transformed training data from PyCaret
X_train_transformed = get_config('X_train_transformed')

# Build SHAP explainer
try:
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_train_transformed)
    shap_type = 'TreeExplainer'
except Exception:
    explainer = shap.KernelExplainer(
        final_model.predict,
        shap.sample(X_train_transformed, 100)
    )
    shap_values = explainer.shap_values(X_train_transformed.iloc[:200])
    shap_type = 'KernelExplainer'

logger.info('SHAP explainer type used: %s', shap_type)

# SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train_transformed, show=False, max_display=15)
plt.title('SHAP Summary Plot - Feature Impact on Price Prediction', fontsize=13)
plt.tight_layout()
plt.savefig('/dbfs/tmp/car_price/shap_summary.png', dpi=120, bbox_inches='tight')
plt.show()

# COMMAND ----------

# SHAP Bar Plot (Global Importance)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_transformed, plot_type='bar', show=False, max_display=15)
plt.title('SHAP Global Feature Importance', fontsize=13)
plt.tight_layout()
plt.savefig('/dbfs/tmp/car_price/shap_importance.png', dpi=120, bbox_inches='tight')
plt.show()

logger.info('SHAP plots saved.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. LIME - Local Prediction Explanations

# COMMAND ----------

logger.info('Building LIME explainer...')

# Get column names and types
feature_names  = X_train_transformed.columns.tolist()
categorical_idx = [
    X_train_transformed.columns.get_loc(c)
    for c in X_train_transformed.select_dtypes(include='object').columns
]

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data       = X_train_transformed.values,
    feature_names       = feature_names,
    categorical_features = categorical_idx,
    mode                = 'regression',
    random_state        = RANDOM_STATE,
    verbose             = False
)

# Explain 3 sample predictions
X_test_transformed = get_config('X_test_transformed')
sample_indices = [0, len(X_test_transformed) // 2, len(X_test_transformed) - 1]

for idx in sample_indices:
    instance = X_test_transformed.iloc[idx].values
    explanation = lime_explainer.explain_instance(
        instance,
        final_model.predict,
        num_features=10
    )
    actual_price    = y_test.iloc[idx]
    predicted_price = y_pred.iloc[idx]
    
    print(f'\n--- LIME Explanation for Test Sample #{idx} ---')
    print(f'Actual Price:    ${actual_price:,.0f}')
    print(f'Predicted Price: ${predicted_price:,.0f}')
    print(f'Error:           ${abs(actual_price - predicted_price):,.0f}')
    print('\nTop 10 Feature Contributions:')
    for feat, weight in explanation.as_list():
        direction = 'raises' if weight > 0 else 'lowers'
        print(f'  {feat}: {direction} price by ${abs(weight):,.2f}')
    
    # Save LIME plot
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(12, 6)
    plt.title(f'LIME Explanation - Sample #{idx} (Actual=${actual_price:,.0f})', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'/dbfs/tmp/car_price/lime_sample_{idx}.png', dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()

logger.info('LIME explanations complete.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Business Summary
# MAGIC 
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Model | Best PyCaret model (auto-selected) |
# MAGIC | MAE | Average $ error on test set |
# MAGIC | RMSE | Root mean squared error |
# MAGIC | R² | Proportion of variance explained |
# MAGIC | MAPE | Mean absolute percentage error |
# MAGIC 
# MAGIC **Key SHAP Findings:**
# MAGIC - Car age and mileage are typically the strongest negative predictors
# MAGIC - Make/brand and segment are the strongest positive differentiators
# MAGIC - Engine volume and condition also contribute meaningfully

# COMMAND ----------

dbutils.notebook.exit(MODEL_PATH)
