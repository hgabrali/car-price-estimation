# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Data Cleaning
# MAGIC 
# MAGIC ## Purpose
# MAGIC Read the raw Parquet file produced by notebook 01, perform comprehensive data
# MAGIC cleaning using Pandas, and persist the cleaned dataset for EDA and feature
# MAGIC engineering.
# MAGIC 
# MAGIC **Pipeline Position:** Step 2 of 7
# MAGIC 
# MAGIC **Input:**  `/tmp/car_price/raw_data.parquet`
# MAGIC 
# MAGIC **Output:** `/tmp/car_price/clean_data.parquet`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports & Configuration

# COMMAND ----------

import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
INPUT_PATH  = "/dbfs/tmp/car_price/raw_data.parquet"
OUTPUT_PATH = "/tmp/car_price/clean_data.parquet"
DBFS_OUTPUT = f"dbfs:{OUTPUT_PATH}"

CURRENT_YEAR = datetime.now().year
logger.info("02_Data_Cleaning started")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Raw Data

# COMMAND ----------

df = pd.read_parquet(INPUT_PATH)
logger.info("Loaded raw data: %d rows x %d cols", *df.shape)
print(f"Raw shape: {df.shape}")
print(df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Standardise Column Names

# COMMAND ----------

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip spaces, replace spaces/slashes with underscores."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df

df = standardise_columns(df)
print("Standardised columns:", df.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fix Data Types

# COMMAND ----------

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns; ensure strings are stripped."""
    
    # Numeric columns expected
    numeric_cols = ["price", "year", "mileage", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.info("Coerced '%s' to numeric.", col)
    
    # String columns: strip whitespace
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    
    return df

df = fix_dtypes(df)
print("Dtypes after fix:")
print(df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Handle Missing Values

# COMMAND ----------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop missing values with documented strategy."""
    
    print("=== Missing values before ===")
    print(df.isnull().sum())
    
    # Replace 'nan' strings with actual NaN
    df.replace("nan", np.nan, inplace=True)
    df.replace("None", np.nan, inplace=True)
    
    # Drop rows where target (price) is missing
    before = len(df)
    df = df.dropna(subset=["price"])
    logger.info("Dropped %d rows with missing price.", before - len(df))
    
    # Numeric: median imputation
    for col in ["mileage", "volume", "year"]:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info("Imputed '%s' with median=%.2f", col, median_val)
    
    # Categorical: mode imputation
    cat_cols = ["make", "model", "condition", "fuel_type",
                "color", "transmission", "drive_unit", "segment"]
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info("Imputed '%s' with mode='%s'", col, mode_val)
    
    print("=== Missing values after ===")
    print(df.isnull().sum())
    return df

df = handle_missing(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Remove Duplicates

# COMMAND ----------

before = len(df)
df = df.drop_duplicates()
logger.info("Removed %d duplicate rows.", before - len(df))
print(f"Shape after deduplication: {df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Outlier Treatment (IQR Method)

# COMMAND ----------

def remove_outliers_iqr(df: pd.DataFrame, col: str,
                        lower_q: float = 0.01,
                        upper_q: float = 0.99) -> pd.DataFrame:
    """Remove rows outside [lower_q, upper_q] quantile range."""
    q_low  = df[col].quantile(lower_q)
    q_high = df[col].quantile(upper_q)
    before = len(df)
    df = df[(df[col] >= q_low) & (df[col] <= q_high)]
    logger.info("Outlier removal on '%s': removed %d rows (kept %d).",
                col, before - len(df), len(df))
    return df

# Apply to target and key numeric features
for col_name in ["price", "mileage", "volume"]:
    if col_name in df.columns:
        df = remove_outliers_iqr(df, col_name)

print(f"Shape after outlier removal: {df.shape}")
print(df[["price", "mileage", "volume"]].describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Domain-Specific Cleaning

# COMMAND ----------

# Year sanity check: keep only reasonable production years
if "year" in df.columns:
    valid_mask = df["year"].between(1990, CURRENT_YEAR)
    removed = (~valid_mask).sum()
    df = df[valid_mask]
    logger.info("Removed %d rows with implausible year values.", removed)

# Segment column: often manually entered; normalise case & trim
if "segment" in df.columns:
    df["segment"] = df["segment"].str.title().str.strip()
    # Collapse rare segments (< 10 occurrences) into 'Other'
    seg_counts = df["segment"].value_counts()
    rare_segs  = seg_counts[seg_counts < 10].index
    df["segment"] = df["segment"].where(~df["segment"].isin(rare_segs), "Other")
    logger.info("Rare segments collapsed: %s", rare_segs.tolist())

# Standardise fuel type values
if "fuel_type" in df.columns:
    df["fuel_type"] = df["fuel_type"].str.title()

# Standardise transmission values
if "transmission" in df.columns:
    df["transmission"] = df["transmission"].str.title()

print("Unique segments:", df["segment"].unique() if "segment" in df.columns else "N/A")
print("Unique fuel types:", df["fuel_type"].unique() if "fuel_type" in df.columns else "N/A")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Final Validation

# COMMAND ----------

print("=== CLEANED DATASET SUMMARY ===")
print(f"Shape: {df.shape}")
print("\nMissing values:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)
print("\nPrice statistics:")
print(df["price"].describe().round(2))
df.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save Cleaned Data

# COMMAND ----------

# Save to DBFS as Parquet
df.to_parquet(f"/dbfs{OUTPUT_PATH}", index=False)
logger.info("Cleaned data saved to: %s", DBFS_OUTPUT)

# Also save as Delta table for easy SQL access
spark_df = spark.createDataFrame(df)
spark_df.write.mode("overwrite").saveAsTable("car_price_clean")
logger.info("Delta table 'car_price_clean' created/updated.")

print(f"Cleaned dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
dbutils.notebook.exit(OUTPUT_PATH)
