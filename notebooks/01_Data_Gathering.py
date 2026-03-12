# Databricks notebook source
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
TABLE_NAME      = "car_price_assignment"
OUTPUT_PATH     = "/tmp/car_price/raw_data.parquet"
DBFS_OUTPUT     = f"dbfs:{OUTPUT_PATH}"

dbutils.fs.mkdirs("/tmp/car_price/")
logger.info("Output directory ensured: /tmp/car_price/")

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
missing_counts = sdf.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in sdf.columns
])
display(missing_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Preview First Rows

# COMMAND ----------

print(f"Dataset shape: {row_count:,} rows x {col_count} columns")
print("Column names:", sdf.columns)
display(sdf.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Persist as Parquet

# COMMAND ----------

logger.info("Writing Parquet to: %s", DBFS_OUTPUT)
(
    sdf
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(DBFS_OUTPUT)
)
logger.info("Parquet write complete.")

file_list = dbutils.fs.ls("/tmp/car_price/")
print("Files written:", [f.name for f in file_list])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Pass Output Path to Next Notebook

# COMMAND ----------

dbutils.notebook.exit(OUTPUT_PATH)
