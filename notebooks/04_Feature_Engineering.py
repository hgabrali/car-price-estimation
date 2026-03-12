# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Feature Engineering
# MAGIC 
# MAGIC ## Purpose
# MAGIC Create new features and transform existing ones to improve model performance.
# MAGIC 
# MAGIC **Pipeline Position:** Step 4 of 7
# MAGIC **Input:** `/tmp/car_price/clean_data.parquet`
# MAGIC **Output:** `/tmp/car_price/featured_data.parquet`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports & Configuration

# COMMAND ----------

import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

INPUT_PATH  = '/dbfs/tmp/car_price/clean_data.parquet'
OUTPUT_PATH = '/tmp/car_price/featured_data.parquet'
CURRENT_YEAR = datetime.now().year
logger.info('04_Feature_Engineering started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Clean Data

# COMMAND ----------

df = pd.read_parquet(INPUT_PATH)
logger.info('Loaded clean data: %d rows x %d cols', *df.shape)
print(f'Input shape: {df.shape}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature: Car Age

# COMMAND ----------

def add_car_age(df, current_year=CURRENT_YEAR):
    '''Derive car age from model year.'''
    if 'year' in df.columns:
        df['car_age'] = current_year - df['year']
        df['car_age'] = df['car_age'].clip(lower=0)
        logger.info('Created car_age feature (current_year=%d)', current_year)
    return df

df = add_car_age(df)
print('Car age stats:')
print(df['car_age'].describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature: Mileage Bins

# COMMAND ----------

def add_mileage_bins(df):
    '''Bin mileage into ordered categories.'''
    if 'mileage' in df.columns:
        bins   = [0, 20000, 50000, 100000, 150000, 200000, float('inf')]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']
        df['mileage_bin'] = pd.cut(df['mileage'], bins=bins, labels=labels, right=True)
        df['mileage_bin'] = df['mileage_bin'].astype(str)
        logger.info('Created mileage_bin feature')
    return df

df = add_mileage_bins(df)
print('Mileage bin distribution:')
print(df['mileage_bin'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature: Age Group

# COMMAND ----------

def add_age_group(df):
    '''Bin car age into descriptive groups.'''
    if 'car_age' in df.columns:
        bins   = [0, 2, 5, 10, 15, 20, float('inf')]
        labels = ['Brand New', 'Nearly New', 'Recent', 'Used', 'Old', 'Classic']
        df['age_group'] = pd.cut(df['car_age'], bins=bins, labels=labels, right=True)
        df['age_group'] = df['age_group'].astype(str)
        logger.info('Created age_group feature')
    return df

df = add_age_group(df)
print('Age group distribution:')
print(df['age_group'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature: Price Per Year (Interaction)

# COMMAND ----------

def add_interaction_features(df):
    '''Create meaningful interaction features.'''
    # Mileage per year of car age (usage intensity)
    if 'mileage' in df.columns and 'car_age' in df.columns:
        df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)
        logger.info('Created mileage_per_year feature')
    # Engine volume efficiency proxy
    if 'volume' in df.columns:
        df['volume_log'] = np.log1p(df['volume'])
        logger.info('Created volume_log feature')
    return df

df = add_interaction_features(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Handle Rare Categories

# COMMAND ----------

def collapse_rare_categories(df, col, threshold=20, replacement='Other'):
    '''Replace categories with fewer than threshold occurrences.'''
    if col not in df.columns:
        return df
    counts = df[col].value_counts()
    rare   = counts[counts < threshold].index
    df[col] = df[col].where(~df[col].isin(rare), replacement)
    logger.info('Collapsed %d rare values in %s into Other', len(rare), col)
    return df

for col_name in ['make', 'model', 'color', 'drive_unit']:
    if col_name in df.columns:
        df = collapse_rare_categories(df, col_name, threshold=20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log-Transform Target Variable

# COMMAND ----------

# Log1p transform helps with right-skewed price distribution
df['log_price'] = np.log1p(df['price'])
logger.info('Created log_price feature (log1p transform of price)')

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['price'], bins=50, color='#2196F3', edgecolor='white')
axes[0].set_title('Price (Original)')
axes[1].hist(df['log_price'], bins=50, color='#4CAF50', edgecolor='white')
axes[1].set_title('Price (Log-transformed)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Feature Summary

# COMMAND ----------

print('=== FEATURE ENGINEERING SUMMARY ===')
print(f'Final shape: {df.shape}')
print(f'New features created:')
new_features = ['car_age', 'mileage_bin', 'age_group', 'mileage_per_year', 'volume_log', 'log_price']
for feat in new_features:
    if feat in df.columns:
        print(f'  - {feat}: {df[feat].dtype}')

print(f'\nAll columns ({len(df.columns)}):')
print(df.columns.tolist())
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save Enhanced Dataset

# COMMAND ----------

df.to_parquet(f'/dbfs{OUTPUT_PATH}', index=False)
logger.info('Featured data saved to: %s', OUTPUT_PATH)

# Also overwrite Delta table
spark_df = spark.createDataFrame(df)
spark_df.write.mode('overwrite').saveAsTable('car_price_featured')
logger.info('Delta table car_price_featured created/updated.')

print(f'Saved featured dataset: {df.shape[0]:,} rows x {df.shape[1]} columns')
dbutils.notebook.exit(OUTPUT_PATH)
