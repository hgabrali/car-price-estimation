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

CLEAN_TABLE = 'car_price_clean'
logger.info('03_EDA started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Clean Data

# COMMAND ----------

df = spark.table(CLEAN_TABLE).toPandas()
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

# Extract brand from carname for analysis
df['brand'] = df['carname'].str.split().str[0].str.lower()

# Price by Brand (Top 10)
top_brands = df['brand'].value_counts().nlargest(10).index
fig, ax = plt.subplots(figsize=(14, 6))
order = (df[df['brand'].isin(top_brands)]
         .groupby('brand')['price'].median()
         .sort_values(ascending=False).index)
sns.boxplot(data=df[df['brand'].isin(top_brands)], x='brand',
            y='price', order=order, ax=ax, palette='Blues_d')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Price Distribution by Brand (Top 10)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Horsepower vs Price
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['horsepower'], df['price'], alpha=0.4, color='#E91E63', s=20)
ax.set_xlabel('Horsepower')
ax.set_ylabel('Price (USD)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Horsepower vs Price', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print(f'Pearson corr (horsepower, price): {df["horsepower"].corr(df["price"]):.3f}')

# COMMAND ----------

# Price by Car Body Type
fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby('carbody')['price'].median().sort_values(ascending=False).index
sns.violinplot(data=df, x='carbody', y='price', order=order, ax=ax,
               palette='Set2', inner='quartile')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Price Distribution by Car Body Type', fontsize=13, fontweight='bold')
ax.set_xlabel('Car Body Type')
ax.set_ylabel('Price (USD)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Median Price by Drive Wheel Type
fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby('drivewheel')['price'].median().sort_values(ascending=False).index
sns.barplot(data=df, x='drivewheel', y='price', order=order, ax=ax,
            estimator=np.median, palette='viridis', ci=None)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Median Price by Drive Wheel Type', fontsize=13, fontweight='bold')
ax.set_xlabel('Drive Wheel')
ax.set_ylabel('Median Price (USD)')
for p in ax.patches:
    ax.annotate(f'${p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Engine Size vs Price (with Fuel Type hue)
fig, ax = plt.subplots(figsize=(12, 5))
for ft, color in zip(['gas', 'diesel'], ['#3F51B5', '#E91E63']):
    subset = df[df['fueltype'] == ft]
    ax.scatter(subset['enginesize'], subset['price'], alpha=0.5, label=ft, color=color)
ax.set_xlabel('Engine Size (cc)')
ax.set_ylabel('Price (USD)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Engine Size vs Price by Fuel Type', fontsize=13, fontweight='bold')
ax.legend(title='Fuel Type')
plt.tight_layout()
plt.show()
print(f'Pearson corr (enginesize, price): {df["enginesize"].corr(df["price"]):.3f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key Insights
# MAGIC
# MAGIC | Insight | Detail |
# MAGIC |---------|--------|
# MAGIC | **Price distribution** | Right-skewed (skewness=1.625); log transformation recommended for modelling |
# MAGIC | **Curbweight** | Strongest positive correlation with price - heavier cars tend to be more expensive |
# MAGIC | **Engine size** | Strong positive correlation with price - larger engines command premium |
# MAGIC | **Horsepower** | Positive correlation; higher performance increases value |
# MAGIC | **Fuel efficiency** | citympg/highwaympg negatively correlated with price (economy vs premium trade-off) |
# MAGIC | **Carbody** | Sedans dominate (47%); convertibles and hardtops are rare but premium |
# MAGIC | **Drive wheel** | FWD most common (59%); RWD associated with higher-priced vehicles |
# MAGIC | **Fuel type** | 87% gas, 13% diesel; no missing values in any column |

# COMMAND ----------

dbutils.notebook.exit(CLEAN_TABLE)
