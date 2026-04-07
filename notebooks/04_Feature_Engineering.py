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
# MAGIC

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

CLEAN_TABLE   = 'car_price_clean'
FEAT_TABLE    = 'car_price_featured'
CURRENT_YEAR = datetime.now().year
logger.info('04_Feature_Engineering started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Clean Data

# COMMAND ----------

df = spark.table(CLEAN_TABLE).toPandas()
logger.info('Loaded clean data: %d rows x %d cols', *df.shape)
print(f'Input shape: {df.shape}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature: Power-to-Weight Ratio & Engine Efficiency

# COMMAND ----------

# Power-to-weight ratio (horsepower per unit of curbweight)
if 'horsepower' in df.columns and 'curbweight' in df.columns:
    df['power_to_weight'] = df['horsepower'] / df['curbweight']
    logger.info('Created power_to_weight feature')

# Engine displacement efficiency (horsepower per engine cc)
if 'horsepower' in df.columns and 'enginesize' in df.columns:
    df['hp_per_cc'] = df['horsepower'] / (df['enginesize'] + 1)
    logger.info('Created hp_per_cc feature')

# Fuel efficiency composite (average of city and highway mpg)
if 'citympg' in df.columns and 'highwaympg' in df.columns:
    df['avg_mpg'] = (df['citympg'] + df['highwaympg']) / 2
    logger.info('Created avg_mpg feature')

print('New feature stats:')
for feat in ['power_to_weight', 'hp_per_cc', 'avg_mpg']:
    if feat in df.columns:
        print(f'  {feat}: mean={df[feat].mean():.4f}, std={df[feat].std():.4f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature: Horsepower Bins & Engine Size Category

# COMMAND ----------

# Horsepower bins
if 'horsepower' in df.columns:
    bins = [0, 70, 100, 150, 200, float('inf')]
    labels = ['Low', 'Medium', 'High', 'Very High', 'Premium']
    df['hp_bin'] = pd.cut(df['horsepower'], bins=bins, labels=labels, right=True).astype(str)
    logger.info('Created hp_bin feature')

# Engine size category
if 'enginesize' in df.columns:
    bins = [0, 100, 150, 200, float('inf')]
    labels = ['Small', 'Medium', 'Large', 'Very Large']
    df['engine_category'] = pd.cut(df['enginesize'], bins=bins, labels=labels, right=True).astype(str)
    logger.info('Created engine_category feature')

print('HP bin distribution:')
print(df['hp_bin'].value_counts() if 'hp_bin' in df.columns else 'hp_bin not created')
print('\nEngine category distribution:')
print(df['engine_category'].value_counts() if 'engine_category' in df.columns else 'engine_category not created')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature: Car Body & Drive Type Aggregations

# COMMAND ----------

# Brand mean price encoding (target-based feature)
if 'brand' in df.columns:
    brand_avg_price = df.groupby('brand')['price'].transform('mean')
    df['brand_avg_price'] = brand_avg_price
    logger.info('Created brand_avg_price feature')

# Car body popularity score
if 'carbody' in df.columns:
    body_counts = df['carbody'].value_counts(normalize=True)
    df['carbody_popularity'] = df['carbody'].map(body_counts)
    logger.info('Created carbody_popularity feature')

# Drive wheel frequency encoding
if 'drivewheel' in df.columns:
    dw_counts = df['drivewheel'].value_counts(normalize=True)
    df['drivewheel_freq'] = df['drivewheel'].map(dw_counts)
    logger.info('Created drivewheel_freq feature')

print('Encoding features stats:')
for feat in ['brand_avg_price', 'carbody_popularity', 'drivewheel_freq']:
    if feat in df.columns:
        print(f'  {feat}: mean={df[feat].mean():.4f}, std={df[feat].std():.4f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature: Interaction & Composite Features

# COMMAND ----------

# Compression per cylinder (interaction feature)
if 'compressionratio' in df.columns and 'cylindernumber' in df.columns:
    cyl_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}
    df['cyl_num'] = df['cylindernumber'].map(cyl_map).fillna(4)
    df['compression_per_cyl'] = df['compressionratio'] / df['cyl_num']
    logger.info('Created compression_per_cyl feature')

# Bore-stroke ratio (engine geometry)
if 'boreratio' in df.columns and 'stroke' in df.columns:
    df['bore_stroke_ratio'] = df['boreratio'] / (df['stroke'] + 0.001)
    logger.info('Created bore_stroke_ratio feature')

# Price per curb weight (value metric)
if 'price' in df.columns and 'curbweight' in df.columns:
    df['price_per_weight'] = df['price'] / df['curbweight']
    logger.info('Created price_per_weight feature')

print('Interaction feature stats:')
for feat in ['compression_per_cyl', 'bore_stroke_ratio', 'price_per_weight']:
    if feat in df.columns:
        print(f'  {feat}: mean={df[feat].mean():.4f}, std={df[feat].std():.4f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Handle Rare Categories

# COMMAND ----------

def collapse_rare_categories(df, col, threshold=5, replacement='other'):
    '''Replace categories with fewer than threshold occurrences.'''
    if col not in df.columns:
        return df
    counts = df[col].value_counts()
    rare = counts[counts < threshold].index
    df[col] = df[col].where(~df[col].isin(rare), replacement)
    logger.info('Collapsed %d rare values in %s into other', len(rare), col)
    return df

# Collapse rare categories in the actual dataset columns
for col_name in ['brand', 'enginetype', 'fuelsystem', 'cylindernumber']:
    if col_name in df.columns:
        df = collapse_rare_categories(df, col_name, threshold=5)

# Print distributions of collapsed columns
for col_name in ['brand', 'enginetype', 'fuelsystem', 'cylindernumber']:
    if col_name in df.columns:
        print(f'\n{col_name} distribution after collapsing:')
        print(df[col_name].value_counts())

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

spark_df = spark.createDataFrame(df)
spark_df.write.mode('overwrite').saveAsTable(FEAT_TABLE)
logger.info("Delta table '%s' created/updated.", FEAT_TABLE)
print(f'Saved featured dataset: {df.shape[0]:,} rows x {df.shape[1]} columns')
print(f'Saved to Delta table: {FEAT_TABLE}')
dbutils.notebook.exit(FEAT_TABLE)
