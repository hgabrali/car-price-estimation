# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Valuation (Car Price Prediction)
# MAGIC 
# MAGIC ## Purpose
# MAGIC Load the trained model and provide a production-ready prediction function
# MAGIC for valuing new car listings. Accepts car attributes and returns predicted
# MAGIC price with confidence context.
# MAGIC 
# MAGIC **Pipeline Position:** Step 7 of 7
# MAGIC **Input:** Saved model from notebook 06 + new car attributes
# MAGIC **Output:** Predicted prices for new listings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports & Configuration

# COMMAND ----------

import logging
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pycaret.regression import load_model, predict_model
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH   = '/dbfs/tmp/car_price/final_model'
CURRENT_YEAR = datetime.now().year

# Widget for JSON input (used when called from Databricks Workflow)
dbutils.widgets.text('car_json', '{}', 'Car Attributes (JSON)')

logger.info('07_Valuation notebook started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Trained Model

# COMMAND ----------

logger.info('Loading model from: %s', MODEL_PATH)
model = load_model(MODEL_PATH)
logger.info('Model loaded successfully: %s', type(model).__name__)
print(f'Model type: {type(model).__name__}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prediction Function

# COMMAND ----------

def predict_price(car_features: dict) -> dict:
    '''
    Predict the price of a used car given its attributes.
    
    Parameters
    ----------
    car_features : dict
        Dictionary of car attributes. Required keys depend on the model,
        but typically include: make, model, year, condition, mileage,
        fuel_type, volume, color, transmission, drive_unit, segment.
    
    Returns
    -------
    dict with keys:
        - predicted_price : float  (USD)
        - car_age          : int
        - input_features   : dict  (the features used after engineering)
    '''
    # ── Feature Engineering (mirrors notebook 04) ──────────────────
    features = car_features.copy()
    
    # Car age
    if 'year' in features:
        features['car_age'] = CURRENT_YEAR - int(features['year'])
    
    # Mileage bin
    if 'mileage' in features:
        mileage = float(features['mileage'])
        if   mileage < 20000:  features['mileage_bin'] = 'Very Low'
        elif mileage < 50000:  features['mileage_bin'] = 'Low'
        elif mileage < 100000: features['mileage_bin'] = 'Medium'
        elif mileage < 150000: features['mileage_bin'] = 'High'
        elif mileage < 200000: features['mileage_bin'] = 'Very High'
        else:                  features['mileage_bin'] = 'Extreme'
    
    # Age group
    car_age = features.get('car_age', 0)
    if   car_age <= 2:  features['age_group'] = 'Brand New'
    elif car_age <= 5:  features['age_group'] = 'Nearly New'
    elif car_age <= 10: features['age_group'] = 'Recent'
    elif car_age <= 15: features['age_group'] = 'Used'
    elif car_age <= 20: features['age_group'] = 'Old'
    else:               features['age_group'] = 'Classic'
    
    # Interaction features
    if 'mileage' in features and 'car_age' in features:
        features['mileage_per_year'] = float(features['mileage']) / (car_age + 1)
    if 'volume' in features:
        features['volume_log'] = np.log1p(float(features['volume']))
    
    # ── Predict ──────────────────────────────────────────────────────
    input_df = pd.DataFrame([features])
    predictions = predict_model(model, data=input_df)
    predicted_price = float(predictions['prediction_label'].iloc[0])
    predicted_price = max(predicted_price, 0)  # no negative prices
    
    return {
        'predicted_price': round(predicted_price, 2),
        'car_age':          car_age,
        'input_features':   features
    }

logger.info('predict_price function defined.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sample Predictions (Hypothetical Listings)

# COMMAND ----------

sample_cars = [
    {
        'name':         'Budget Hatchback',
        'make':         'Ford',
        'model':        'Fiesta',
        'year':         2018,
        'condition':    'Good',
        'mileage':      65000,
        'fuel_type':    'Petrol',
        'volume':       1400,
        'color':        'White',
        'transmission': 'Manual',
        'drive_unit':   'FWD',
        'segment':      'Compact'
    },
    {
        'name':         'Mid-Range SUV',
        'make':         'Toyota',
        'model':        'RAV4',
        'year':         2020,
        'condition':    'Excellent',
        'mileage':      30000,
        'fuel_type':    'Hybrid',
        'volume':       2500,
        'color':        'Silver',
        'transmission': 'Automatic',
        'drive_unit':   'AWD',
        'segment':      'SUV'
    },
    {
        'name':         'Premium Sedan',
        'make':         'BMW',
        'model':        '5 Series',
        'year':         2022,
        'condition':    'Excellent',
        'mileage':      15000,
        'fuel_type':    'Petrol',
        'volume':       3000,
        'color':        'Black',
        'transmission': 'Automatic',
        'drive_unit':   'RWD',
        'segment':      'Luxury'
    },
    {
        'name':         'High-Mileage Workhorse',
        'make':         'Volkswagen',
        'model':        'Golf',
        'year':         2015,
        'condition':    'Fair',
        'mileage':      180000,
        'fuel_type':    'Diesel',
        'volume':       1600,
        'color':        'Grey',
        'transmission': 'Manual',
        'drive_unit':   'FWD',
        'segment':      'Compact'
    }
]

results = []
for car in sample_cars:
    car_input = {k: v for k, v in car.items() if k != 'name'}
    result = predict_price(car_input)
    results.append({
        'Listing Name':      car['name'],
        'Make':              car['make'],
        'Model':             car['model'],
        'Year':              car['year'],
        'Condition':         car['condition'],
        'Mileage (km)':      car['mileage'],
        'Fuel':              car['fuel_type'],
        'Car Age (yrs)':     result['car_age'],
        'Predicted Price ($)': f"${result['predicted_price']:,.0f}"
    })

results_df = pd.DataFrame(results)
print('\n=== DEALERSHIP VALUATION RESULTS ===')
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Batch Prediction from Widget Input

# COMMAND ----------

# Read car attributes from widget (for Workflow automation)
car_json_str = dbutils.widgets.get('car_json')

if car_json_str and car_json_str != '{}':
    try:
        car_features = json.loads(car_json_str)
        widget_result = predict_price(car_features)
        print(f'\n=== WIDGET PREDICTION ===')
        print(f'Input: {json.dumps(car_features, indent=2)}')
        print(f'Predicted Price: ${widget_result["predicted_price"]:,.2f}')
        print(f'Car Age: {widget_result["car_age"]} years')
        dbutils.notebook.exit(json.dumps(widget_result))
    except Exception as e:
        logger.error('Widget prediction failed: %s', str(e))
        print(f'Widget input error: {e}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Integration Guide
# MAGIC 
# MAGIC ### Option A: Databricks Workflow (Batch)
# MAGIC 1. Add this notebook as the final task in your job
# MAGIC 2. Pass car attributes via the `car_json` widget parameter
# MAGIC 3. The notebook returns the prediction as the notebook exit value
# MAGIC 
# MAGIC ### Option B: MLflow Model Serving (REST API)
# MAGIC ```python
# MAGIC # Register model in MLflow (done in notebook 06)
# MAGIC # Then enable Model Serving in Databricks:
# MAGIC # Serving > Create Endpoint > Select 'CarPriceRegressor'
# MAGIC # Call via REST:
# MAGIC import requests
# MAGIC url = 'https://<workspace>.azuredatabricks.net/serving-endpoints/CarPriceRegressor/invocations'
# MAGIC headers = {'Authorization': 'Bearer <token>', 'Content-Type': 'application/json'}
# MAGIC payload = {'dataframe_records': [car_features]}
# MAGIC response = requests.post(url, headers=headers, json=payload)
# MAGIC print(response.json())
# MAGIC ```
# MAGIC 
# MAGIC ### Option C: Direct Notebook Call
# MAGIC ```python
# MAGIC result = dbutils.notebook.run(
# MAGIC     '07_Valuation',
# MAGIC     timeout_seconds=120,
# MAGIC     arguments={'car_json': json.dumps(car_features)}
# MAGIC )
# MAGIC price = json.loads(result)['predicted_price']
# MAGIC ```

# COMMAND ----------

logger.info('07_Valuation completed successfully.')
dbutils.notebook.exit('Valuation complete')
