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

# MAGIC %pip install pycaret==3.3.2 "numpy<2" --quiet

# COMMAND ----------

dbutils.library.restartPython()

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
FEAT_TABLE      = 'car_price_featured'
TRAIN_TABLE     = 'car_price_train'
TEST_TABLE      = 'car_price_test'
MLFLOW_EXP_NAME = '/Users/hande.gabriali@gmail.com/car_price_pipeline'

TARGET_COL = 'price'
TEST_SIZE  = 0.2
RANDOM_STATE = 42

logger.info('05_Preprocessing_Dataset started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Featured Data

# COMMAND ----------

# Try loading from pipeline Delta table; fall back to CSV if table doesn't exist
FEAT_TABLE = 'car_price_featured'
try:
    df = spark.table(FEAT_TABLE).toPandas()
    logger.info('Loaded from Delta table: %s', FEAT_TABLE)
except Exception:
    logger.warning('Delta table %s not found, loading from CSV as fallback', FEAT_TABLE)
    CSV_PATH = "/Workspace/Users/hande.gabrali@gmail.com/car-price-estimation/Dataset/CarPrice_Assignment.csv"
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[\s/]+', '_', regex=True)

    # Add log_price for modelling
    df['log_price'] = np.log1p(df['price'])
    logger.info('Loaded featured data: %d rows x %d cols', *df.shape)
print(f'Input shape: {df.shape}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Select Features

# COMMAND ----------

# Drop non-predictive & leakage columns
COLS_TO_DROP = ['log_price', 'car_id', 'carname']
df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
logger.info('Dropped columns: %s', COLS_TO_DROP)

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

print(f'Train: {train_df.shape} | Test: {test_df.shape}')

spark.createDataFrame(train_df).write.mode('overwrite').option("mergeSchema", "true").saveAsTable(TRAIN_TABLE)
spark.createDataFrame(test_df).write.mode('overwrite').option("mergeSchema", "true").saveAsTable(TEST_TABLE)
logger.info('Train/Test splits saved to Delta tables.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Initialise PyCaret Environment

# COMMAND ----------

pycaret_env = pycaret_setup(data=train_df, target=TARGET_COL, session_id=RANDOM_STATE, normalize=True, verbose=False)
logger.info('PyCaret environment initialised')

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
print(f'Total samples:    {len(df):,}')
print(f'Training samples: {len(train_df):,} ({(1-TEST_SIZE)*100:.0f}%)')
print(f'Test samples:     {len(test_df):,} ({TEST_SIZE*100:.0f}%)')
print(f'Feature count:    {len(FEATURE_COLS)}')
print(f'Target:           {TARGET_COL}')
print(f'Train Table:      {TRAIN_TABLE}')
print(f'Test Table:       {TEST_TABLE}')
print(f'MLflow Experiment: {MLFLOW_EXP_NAME}')

# COMMAND ----------

dbutils.notebook.exit(TRAIN_TABLE)
