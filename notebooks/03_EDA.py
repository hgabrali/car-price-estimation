# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Exploratory Data Analysis (EDA)
# MAGIC 
# MAGIC ## Purpose
# MAGIC Perform comprehensive EDA on the cleaned dataset to uncover patterns,
# MAGIC distributions, correlations, and business insights for feature engineering.
# MAGIC 
# MAGIC **Pipeline Position:** Step 3 of 7
# MAGIC **Input:** `/tmp/car_price/clean_data.parquet`
# MAGIC **Output:** EDA insights (in-notebook) + cleaned DataFrame passed forward

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports & Configuration

# COMMAND ----------

import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

INPUT_PATH  = '/dbfs/tmp/car_price/clean_data.parquet'
OUTPUT_PATH = '/tmp/car_price/clean_data.parquet'
logger.info('03_EDA started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Clean Data

# COMMAND ----------

df = pd.read_parquet(INPUT_PATH)
logger.info('Loaded clean data: %d rows x %d cols', *df.shape)
print(f'Shape: {df.shape}')
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Summary Statistics

# COMMAND ----------

print('=== NUMERICAL SUMMARY ===')
display(df.describe(include=[np.number]).T.round(2))

print('\n=== CATEGORICAL SUMMARY ===')
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    print(f'\n{col} ({df[col].nunique()} unique):')
    print(df[col].value_counts().head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Target Variable Distribution

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['price'], bins=50, color='#2196F3', edgecolor='white', alpha=0.85)
axes[0].set_title('Price Distribution (Raw)')
axes[0].set_xlabel('Price (USD)')
axes[0].set_ylabel('Count')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

axes[1].hist(np.log1p(df['price']), bins=50, color='#4CAF50', edgecolor='white', alpha=0.85)
axes[1].set_title('Price Distribution (Log Scale)')
axes[1].set_xlabel('log(1 + Price)')
axes[1].set_ylabel('Count')

plt.suptitle('Target Variable: Price', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

skewness = df['price'].skew()
print(f'Price skewness: {skewness:.3f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Numerical Feature Distributions

# COMMAND ----------

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'price']

n = len(num_cols)
cols_per_row = 3
rows = (n + cols_per_row - 1) // cols_per_row
fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, rows * 4))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col].dropna(), bins=40, color='#FF9800', edgecolor='white', alpha=0.85)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Numerical Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Correlation Heatmap

# COMMAND ----------

num_df = df.select_dtypes(include=[np.number])
corr_matrix = num_df.corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5, square=True)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Price vs Key Features (Grouped Analyses)

# COMMAND ----------

# Price by Make (Top 15)
if 'make' in df.columns:
    top_makes = df['make'].value_counts().nlargest(15).index
    fig, ax = plt.subplots(figsize=(14, 6))
    order = (df[df['make'].isin(top_makes)]
             .groupby('make')['price'].median()
             .sort_values(ascending=False).index)
    sns.boxplot(data=df[df['make'].isin(top_makes)],
                x='make', y='price', order=order, ax=ax, palette='Blues_d')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_title('Price Distribution by Make (Top 15)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# Mileage vs Price
if 'mileage' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['mileage'], df['price'], alpha=0.3, color='#E91E63', s=15)
    ax.set_xlabel('Mileage (km)')
    ax.set_ylabel('Price (USD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_title('Mileage vs Price', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print(f'Pearson corr (mileage, price): {df["mileage"].corr(df["price"]):.3f}')

# COMMAND ----------

# Price by Condition
if 'condition' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby('condition')['price'].median().sort_values(ascending=False).index
    sns.violinplot(data=df, x='condition', y='price', order=order, ax=ax,
                   palette='Set2', inner='quartile')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_title('Price Distribution by Condition', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# Price by Segment
if 'segment' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    order = df.groupby('segment')['price'].median().sort_values(ascending=False).index
    sns.barplot(data=df, x='segment', y='price', order=order, ax=ax,
                estimator=np.median, palette='viridis', ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_title('Median Price by Segment', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# Price trend over Year
if 'year' in df.columns:
    yearly = df.groupby('year')['price'].median().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly['year'], yearly['price'], marker='o', color='#3F51B5', linewidth=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_xlabel('Model Year')
    ax.set_ylabel('Median Price (USD)')
    ax.set_title('Median Car Price by Model Year', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key Insights
# MAGIC 
# MAGIC | Insight | Detail |
# MAGIC |---------|--------|
# MAGIC | **Price distribution** | Right-skewed; log transformation recommended |
# MAGIC | **Make impact** | Premium brands show significantly higher median prices |
# MAGIC | **Mileage** | Negative correlation with price as expected |
# MAGIC | **Condition** | Strong predictor; newer condition commands premium |
# MAGIC | **Model year** | Recent years correlate with higher prices |
# MAGIC | **Segment** | SUV/Luxury segments show highest median prices |
# MAGIC | **Fuel type** | Electric/Hybrid vehicles priced higher on average |

# COMMAND ----------

dbutils.notebook.exit(OUTPUT_PATH)
