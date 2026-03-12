# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Preprocessing Dataset
# MAGIC 
# MAGIC ## Purpose
# MAGIC Prepare data for modelling: define features and target, split train/test,
# MAGIC and initialise the PyCaret regression environment. Log preprocessing steps
# MAGIC with MLflow.
# MAGIC 
# MAGIC **Pipeline Position:** Step 5 of 7
# MAGIC **Input:** `/tmp/car_price/featured_data.parquet`
# MAGIC **Output:** Train/Test Parquet files + PyCaret environment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies (if not already installed)

# COMMAND ----------

# %pip install pycaret==3.3.2 --quiet
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
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# PyCaret regression
from pycaret.regression import setup as pycaret_setup, get_config

# Paths
INPUT_PATH      = '/dbfs/tmp/car_price/featured_data.parquet'
TRAIN_PATH      = '/tmp/car_price/train_data.parquet'
TEST_PATH       = '/tmp/car_price/test_data.parquet'
MLFLOW_EXP_NAME = '/Users/car-price-estimation/car_price_pipeline'

TARGET_COL = 'price'
TEST_SIZE  = 0.2
RANDOM_STATE = 42

logger.info('05_Preprocessing_Dataset started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Featured Data

# COMMAND ----------

df = pd.read_parquet(INPUT_PATH)
logger.info('Loaded featured data: %d rows x %d cols', *df.shape)
print(f'Input shape: {df.shape}')
print(f'Target column: {TARGET_COL}')
print(f'Target stats:\n{df[TARGET_COL].describe().round(2)}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Select Features

# COMMAND ----------

# Drop derived log_price to avoid leakage - keep original price as target
COLS_TO_DROP = ['log_price']
df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])

# Define feature columns
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]
logger.info('Using %d features: %s', len(FEATURE_COLS), FEATURE_COLS)
print(f'Features ({len(FEATURE_COLS)}): {FEATURE_COLS}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train/Test Split

# COMMAND ----------

X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_df = X_train.copy()
train_df[TARGET_COL] = y_train.values

test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values

logger.info('Train size: %d rows | Test size: %d rows', len(train_df), len(test_df))
print(f'Train: {train_df.shape} | Test: {test_df.shape}')

# Persist splits
train_df.to_parquet(f'/dbfs{TRAIN_PATH}', index=False)
test_df.to_parquet(f'/dbfs{TEST_PATH}', index=False)
logger.info('Train/Test splits saved.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Initialise PyCaret Environment

# COMMAND ----------

# Set MLflow experiment
mlflow.set_experiment(MLFLOW_EXP_NAME)

with mlflow.start_run(run_name='05_Preprocessing') as run:
    logger.info('MLflow run started: %s', run.info.run_id)
    
    # PyCaret setup - let PyCaret handle internal preprocessing
    pycaret_env = pycaret_setup(
        data             = train_df,
        target           = TARGET_COL,
        session_id       = RANDOM_STATE,
        train_size       = 1.0,           # already split; pass full train set
        normalize        = True,
        normalize_method = 'zscore',
        transformation   = True,          # Yeo-Johnson to reduce skewness
        transformation_method = 'yeo-johnson',
        remove_multicollinearity = True,
        multicollinearity_threshold = 0.9,
        fix_imbalance    = False,         # regression task
        feature_selection = False,        # handled manually
        log_experiment   = True,
        experiment_name  = MLFLOW_EXP_NAME,
        verbose          = False
    )
    
    # Log preprocessing parameters
    mlflow.log_param('train_size',    len(train_df))
    mlflow.log_param('test_size',     len(test_df))
    mlflow.log_param('n_features',    len(FEATURE_COLS))
    mlflow.log_param('target_col',    TARGET_COL)
    mlflow.log_param('random_state',  RANDOM_STATE)
    mlflow.log_param('normalize',     True)
    mlflow.log_param('transformation','yeo-johnson')
    
    logger.info('PyCaret environment initialised successfully.')
    print('MLflow Run ID:', run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verify PyCaret Config

# COMMAND ----------

X_transformed = get_config('X_train_transformed')
print(f'Transformed training features shape: {X_transformed.shape}')
print(f'Transformed feature names: {X_transformed.columns.tolist()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary

# COMMAND ----------

print('=== PREPROCESSING SUMMARY ===')
print(f'Total samples:      {len(df):,}')
print(f'Training samples:   {len(train_df):,} ({(1-TEST_SIZE)*100:.0f}%)')
print(f'Test samples:       {len(test_df):,} ({TEST_SIZE*100:.0f}%)')
print(f'Feature count:      {len(FEATURE_COLS)}')
print(f'Target:             {TARGET_COL}')
print(f'Train Parquet:      {TRAIN_PATH}')
print(f'Test Parquet:       {TEST_PATH}')
print(f'MLflow Experiment:  {MLFLOW_EXP_NAME}')

# COMMAND ----------

dbutils.notebook.exit(TRAIN_PATH)
