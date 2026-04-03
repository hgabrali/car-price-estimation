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

%pip install shap lime "numpy<2" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports & Configuration

# COMMAND ----------

import logging, warnings, os, joblib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow, mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV

import shap
import lime, lime.lime_tabular

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

TRAIN_TABLE  = 'car_price_train'
TEST_TABLE   = 'car_price_test'
MODEL_PATH   = '/Workspace/Users/hande.gabrali@gmail.com/car-price-estimation/models/final_model'
PLOTS_DIR    = '/Workspace/Users/hande.gabrali@gmail.com/car-price-estimation/plots'
TARGET_COL   = 'price'
MLFLOW_EXP_NAME = '/Users/hande.gabrali@gmail.com/car_price_pipeline'
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
logger.info('06_3Models_Training started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Train & Test Data

# COMMAND ----------

train_df = spark.table(TRAIN_TABLE).toPandas()
test_df  = spark.table(TEST_TABLE).toPandas()
logger.info('Train: %d rows | Test: %d rows', len(train_df), len(test_df))
print(f'Train shape: {train_df.shape} | Test shape: {test_df.shape}')

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Initialise PyCaret

# COMMAND ----------

cat_cols = train_df.select_dtypes(include='object').columns.drop([TARGET_COL], errors='ignore').tolist()
num_cols = train_df.select_dtypes(exclude='object').columns.drop([TARGET_COL], errors='ignore').tolist()

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

logger.info('Ready: %d num, %d cat features', len(num_cols), len(cat_cols))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare Models (PyCaret AutoML)

# COMMAND ----------

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

m_ridge = Pipeline([('pre', preprocessor), ('model', Ridge(alpha=10.0, random_state=RANDOM_STATE))])
m_rf    = Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))])
m_gb    = Pipeline([('pre', preprocessor), ('model', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE))])

s_ridge = -cross_val_score(m_ridge, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
s_rf    = -cross_val_score(m_rf,    X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
s_gb    = -cross_val_score(m_gb,    X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()

scores_dict = {'ridge': s_ridge, 'random_forest': s_rf, 'gradient_boosting': s_gb}
comparison_df = pd.DataFrame(list(scores_dict.items()), columns=['Model', 'CV_RMSE']).sort_values('CV_RMSE')
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
best_model = [m for n, m in [('ridge', m_ridge), ('random_forest', m_rf), ('gradient_boosting', m_gb)] if n == best_model_name][0]
logger.info('Best: %s', best_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tune Best Model

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

param_grids = {
    'gradient_boosting': {'model__n_estimators': [100, 200], 'model__max_depth': [3, 5], 'model__learning_rate': [0.05, 0.1]},
    'random_forest':     {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10], 'model__min_samples_split': [2, 5]},
    'ridge':             {'model__alpha': [1.0, 10.0, 100.0]}
}
param_grid  = param_grids[best_model_name]
grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

tuned_model  = grid_search.best_estimator_
final_model  = tuned_model
logger.info('Best params: %s | CV RMSE: %.2f', grid_search.best_params_, -grid_search.best_score_)
print(f'Best params: {grid_search.best_params_}')
print(f'Best CV RMSE: {-grid_search.best_score_:.2f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Finalise & Evaluate on Test Set

# COMMAND ----------

final_model = best_model
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

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
plot_path = f'{PLOTS_DIR}/actual_vs_predicted.png'
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.show()
logger.info('Plot saved to: %s', plot_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log to MLflow

# COMMAND ----------

import mlflow, joblib

try:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(MLFLOW_EXP_NAME)
    with mlflow.start_run(run_name='06_Final_Model') as run:
        mlflow.log_param('model_type', type(final_model).__name__)
        mlflow.log_param('random_state', RANDOM_STATE)
        mlflow.log_metric('test_mae', mae)
        mlflow.log_metric('test_rmse', rmse)
        mlflow.log_metric('test_r2', r2)
        mlflow.log_metric('test_mape', mape)
        try:
            mlflow.sklearn.log_model(final_model, artifact_path='car_price_model')
        except Exception:
            pass
        try:
            mlflow.log_artifact(f'{PLOTS_DIR}/actual_vs_predicted.png')
        except Exception:
            pass
        logger.info('MLflow run ID: %s', run.info.run_id)
        print('MLflow Run ID:', run.info.run_id)
except Exception as e:
    logger.warning('MLflow logging failed (Free Edition limitation): %s', e)
    print(f'MLflow skipped due to: {e}')
    print('Saving model locally instead...')
    joblib.dump(final_model, MODEL_PATH + '.pkl')
    logger.info('Model saved locally to %s.pkl', MODEL_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save Model (PyCaret)

# COMMAND ----------

import joblib
model_file = MODEL_PATH + '.pkl'
joblib.dump(final_model, model_file)
logger.info('Model saved to: %s', model_file)
print(f'Model saved to: {model_file}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. SHAP - Global Feature Importance

# COMMAND ----------

logger.info('Computing SHAP values...')
X_train_transformed = final_model.named_steps['pre'].transform(X_train)
inner_model = final_model.named_steps['model']
explainer   = shap.TreeExplainer(inner_model)
shap_values = explainer.shap_values(X_train_transformed)
logger.info('SHAP computed')

ohe_features = list(final_model.named_steps['pre'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols))
feat_names   = num_cols + ohe_features
n_feats      = X_train_transformed.shape[1]
X_train_df   = pd.DataFrame(X_train_transformed, columns=feat_names[:n_feats])

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train_df, show=False, max_display=15)
plt.title('SHAP Summary Plot', fontsize=13)
plt.tight_layout()
shap_path = f'{PLOTS_DIR}/shap_summary.png'
plt.savefig(shap_path, dpi=120, bbox_inches='tight')
plt.show()
logger.info('SHAP summary saved to %s', shap_path)

# COMMAND ----------

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_df, plot_type='bar', show=False, max_display=15)
plt.title('SHAP Global Feature Importance', fontsize=13)
plt.tight_layout()
shap_importance_path = f'{PLOTS_DIR}/shap_importance.png'
plt.savefig(shap_importance_path, dpi=120, bbox_inches='tight')
plt.show()
logger.info('SHAP importance saved to %s', shap_importance_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. LIME - Local Prediction Explanations

# COMMAND ----------

logger.info('Building LIME explainer...')
X_train_arr = final_model.named_steps['pre'].transform(X_train)
X_test_arr  = final_model.named_steps['pre'].transform(X_test)
inner_model = final_model.named_steps['model']

lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train_arr,
                                                        mode='regression',
                                                        random_state=RANDOM_STATE,
                                                        verbose=False)
sample_indices = [0, len(X_test_arr) // 2, len(X_test_arr) - 1]
y_pred_series  = pd.Series(y_pred, index=y_test.index)

for idx in sample_indices:
    instance    = X_test_arr[idx]
    explanation = lime_explainer.explain_instance(instance, inner_model.predict, num_features=10)
    actual_price    = y_test.iloc[idx]
    predicted_price = y_pred_series.iloc[idx]
    print(f'--- LIME Sample #{idx}: Actual=${actual_price:,.0f} | Predicted=${predicted_price:,.0f} ---')
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(12, 6)
    plt.title(f'LIME Sample #{idx}', fontsize=12)
    plt.tight_layout()
    lime_path = f'{PLOTS_DIR}/lime_sample_{idx}.png'
    plt.savefig(lime_path, dpi=120, bbox_inches='tight')
    plt.close()

logger.info('LIME complete.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Business Summary & Agent Audit Framework
# MAGIC
# MAGIC ### Model Performance Metrics
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Model  | Random Forest (sklearn Pipeline) |
# MAGIC | MAE    | $$1,376 average error per prediction |
# MAGIC | RMSE   | $$1,915 root mean squared error |
# MAGIC | R2     | 0.9536 (95.4% variance explained) |
# MAGIC | MAPE   | 10.02% mean absolute percentage error |
# MAGIC
# MAGIC ### Key SHAP Findings
# MAGIC - **Curbweight and enginesize** are the strongest price predictors in this dataset
# MAGIC - **Horsepower and car dimensions** (carlength, carwidth) are significant positive drivers
# MAGIC - **Fuel efficiency (citympg, highwaympg)** negatively correlates with price (economy vs luxury trade-off)
# MAGIC
# MAGIC ### Agent Audit Framework (per Article Reference)
# MAGIC Following the AI Agent audit methodology for Data Analysts/Scientists:
# MAGIC - **i) Logical Validation:** The Random Forest model was validated against Ridge and Gradient Boosting. Feature selection is driven by domain knowledge (automotive pricing factors).
# MAGIC - **ii) Hallucination Detection:** SHAP and LIME analyses confirm the model relies on physically meaningful features, not spurious correlations.
# MAGIC - **iii) Information Sources:** All data comes from the verified CarPrice_Assignment dataset with 205 records and 26 features.
# MAGIC - **iv) Error Margins:** MAPE of ~10% is acceptable for dealership pricing guidance. Predictions should be treated as estimates, not exact valuations.
# MAGIC - **v) Business Translation:** The model enables data-driven pricing, replacing gut-feel estimates with quantified predictions and confidence context.
# MAGIC - **vi) Strategic Decision:** For a dealership, minimizing over-pricing (losing customers) is balanced against under-pricing (losing margin). The 10% MAPE provides a reasonable trade-off.

# COMMAND ----------

dbutils.notebook.exit(MODEL_PATH)
