# Databricks notebook source
import pandas as pd
CSV_PATH = "/Workspace/Users/hande.gabrali@gmail.com/car-price-estimation/Dataset/CarPrice_Assignment.csv"
pdf = pd.read_csv(CSV_PATH)
sdf = spark.createDataFrame(pdf)
sdf.write.mode("overwrite").saveAsTable("car_price_assignment")
print("car_price_assignment created:", pdf.shape[0], "rows x", pdf.shape[1], "cols")

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Data Gathering
# MAGIC
# MAGIC ## Purpose
# MAGIC Load the `car_price_assignment` dataset from the Databricks table using PySpark,
# MAGIC display schema and basic statistics, then persist it as a Parquet file on DBFS for
# MAGIC downstream notebooks.
# MAGIC
# MAGIC **Pipeline Position:** Step 1 of 7
# MAGIC
# MAGIC **Output:** `/dbfs/tmp/car_price/raw_data.parquet`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup & Library Imports

# COMMAND ----------

import logging
from datetime import datetime

# PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

logger.info("01_Data_Gathering notebook started at %s", datetime.now())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

# Configuration
TABLE_NAME   = "car_price_assignment"
# Using Delta tables instead of DBFS (DBFS disabled in Serverless)
RAW_TABLE    = "car_price_raw"
logger.info("Configuration set. Using Delta tables for storage.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Data with PySpark

# COMMAND ----------

logger.info("Reading table: %s", TABLE_NAME)

try:
    sdf = spark.table(TABLE_NAME)
    row_count = sdf.count()
    col_count = len(sdf.columns)
    logger.info("Successfully loaded %d rows x %d columns", row_count, col_count)
except Exception as e:
    logger.error("Failed to load table '%s': %s", TABLE_NAME, str(e))
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Schema & Basic Statistics

# COMMAND ----------

print("=" * 60)
print("SCHEMA")
print("=" * 60)
sdf.printSchema()

# COMMAND ----------

print("=" * 60)
print("BASIC STATISTICS (numerical columns)")
print("=" * 60)
display(sdf.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Missing Value Overview

# COMMAND ----------

print("=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)
import pandas as pd
pdf = sdf.toPandas()
print(pdf.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Preview First Rows

# COMMAND ----------

print(f"Dataset shape: {row_count:,} rows x {col_count} columns")
print("Column names:", sdf.columns)
display(sdf.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Persist as Parquet - Delta Table yapabilirisn

# COMMAND ----------

logger.info("Writing raw data to Delta table: %s", RAW_TABLE)
sdf.write.mode("overwrite").saveAsTable(RAW_TABLE)
logger.info("Delta table write complete.")
print(f"Raw data saved to Delta table: {RAW_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Pass Output Path to Next Notebook

# COMMAND ----------

dbutils.notebook.exit(RAW_TABLE)
