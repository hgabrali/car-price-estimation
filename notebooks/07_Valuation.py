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

# MAGIC %pip install "numpy<2" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import logging
import warnings
warnings.filterwarnings('ignore')

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = '/Workspace/Users/hande.gabrali@gmail.com/car-price-estimation/models/final_model.pkl'

# Widget for JSON input (used when called from Databricks Workflow)
dbutils.widgets.text('car_json', '{}', 'Car Attributes (JSON)')

logger.info('07_Valuation notebook started')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Trained Model

# COMMAND ----------

import os
logger.info('Loading model from: %s', MODEL_PATH)
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logger.info('Model loaded successfully: %s', type(model).__name__)
    print(f'Model type: {type(model).__name__}')
else:
    logger.warning('Model file not found at %s. Run notebook 06 first.', MODEL_PATH)
    print(f'WARNING: Model not found at {MODEL_PATH}. Please run notebook 06 to train and save the model.')
    model = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prediction Function

# COMMAND ----------

def predict_price(car_features: dict) -> dict:
    """
    Predict the price of a car given its attributes.

    Parameters
    ----------
    car_features : dict
        Dictionary of car attributes matching the CarPrice_Assignment dataset schema.
        Expected keys: symboling, fueltype, aspiration, doornumber, carbody,
        drivewheel, enginelocation, wheelbase, carlength, carwidth, carheight,
        curbweight, enginetype, cylindernumber, enginesize, fuelsystem,
        boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg

    Returns
    -------
    dict with keys:
        - predicted_price : float (USD)
        - input_features  : dict (the features used for prediction)
    """
    if model is None:
        return {'predicted_price': 0.0, 'input_features': car_features,
                'error': 'Model not loaded. Run notebook 06 first.'}

    input_df = pd.DataFrame([car_features])

    # Use the model's pipeline (includes PyCaret preprocessing)
    try:
        prediction = model.predict(input_df)
        predicted_price = float(prediction[0])
        predicted_price = max(predicted_price, 0)
    except Exception as e:
        logger.error('Prediction failed: %s', str(e))
        return {'predicted_price': 0.0, 'input_features': car_features,
                'error': str(e)}

    return {
        'predicted_price': round(predicted_price, 2),
        'input_features': car_features
    }

logger.info('predict_price function defined.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sample Predictions (Hypothetical Listings)

# COMMAND ----------

# Sample cars based on actual CarPrice_Assignment dataset schema
sample_cars = [
    {
        'name': 'Economy Hatchback (Toyota-like)',
        'symboling': 1, 'fueltype': 'gas', 'aspiration': 'std',
        'doornumber': 'four', 'carbody': 'hatchback', 'drivewheel': 'fwd',
        'enginelocation': 'front', 'wheelbase': 95.7, 'carlength': 158.7,
        'carwidth': 63.6, 'carheight': 54.5, 'curbweight': 2050,
        'enginetype': 'ohc', 'cylindernumber': 'four', 'enginesize': 97,
        'fuelsystem': '2bbl', 'boreratio': 3.19, 'stroke': 3.03,
        'compressionratio': 9.0, 'horsepower': 70, 'peakrpm': 4800,
        'citympg': 31, 'highwaympg': 37
    },
    {
        'name': 'Mid-Range Sedan (Honda-like)',
        'symboling': 0, 'fueltype': 'gas', 'aspiration': 'std',
        'doornumber': 'four', 'carbody': 'sedan', 'drivewheel': 'fwd',
        'enginelocation': 'front', 'wheelbase': 102.4, 'carlength': 175.0,
        'carwidth': 66.5, 'carheight': 54.1, 'curbweight': 2750,
        'enginetype': 'ohc', 'cylindernumber': 'four', 'enginesize': 130,
        'fuelsystem': 'mpfi', 'boreratio': 3.33, 'stroke': 3.47,
        'compressionratio': 9.5, 'horsepower': 110, 'peakrpm': 5500,
        'citympg': 26, 'highwaympg': 31
    },
    {
        'name': 'Premium Sedan (BMW-like)',
        'symboling': 0, 'fueltype': 'gas', 'aspiration': 'std',
        'doornumber': 'four', 'carbody': 'sedan', 'drivewheel': 'rwd',
        'enginelocation': 'front', 'wheelbase': 108.7, 'carlength': 186.7,
        'carwidth': 68.3, 'carheight': 56.0, 'curbweight': 3500,
        'enginetype': 'ohc', 'cylindernumber': 'six', 'enginesize': 209,
        'fuelsystem': 'mpfi', 'boreratio': 3.62, 'stroke': 3.39,
        'compressionratio': 8.0, 'horsepower': 182, 'peakrpm': 5400,
        'citympg': 16, 'highwaympg': 22
    },
    {
        'name': 'Diesel Wagon (Peugeot-like)',
        'symboling': 0, 'fueltype': 'diesel', 'aspiration': 'turbo',
        'doornumber': 'four', 'carbody': 'wagon', 'drivewheel': 'fwd',
        'enginelocation': 'front', 'wheelbase': 107.9, 'carlength': 186.7,
        'carwidth': 68.4, 'carheight': 56.7, 'curbweight': 3075,
        'enginetype': 'l', 'cylindernumber': 'four', 'enginesize': 152,
        'fuelsystem': 'idi', 'boreratio': 3.70, 'stroke': 3.52,
        'compressionratio': 21.0, 'horsepower': 95, 'peakrpm': 4150,
        'citympg': 25, 'highwaympg': 28
    }
]

results = []
for car in sample_cars:
    car_input = {k: v for k, v in car.items() if k != 'name'}
    result = predict_price(car_input)
    results.append({
        'Listing': car['name'],
        'Body': car['carbody'],
        'Drive': car['drivewheel'],
        'Engine (cc)': car['enginesize'],
        'HP': car['horsepower'],
        'Curb Wt': car['curbweight'],
        'Predicted Price ($)': f"${result['predicted_price']:,.0f}" if result['predicted_price'] > 0 else 'N/A'
    })

results_df = pd.DataFrame(results)
print('\n=== CAR PRICE VALUATION RESULTS ===')
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
