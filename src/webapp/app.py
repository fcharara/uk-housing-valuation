"""
Flask Web Application — Methodology Section 7
Provides an interface to predict house prices with SHAP explanations.
"""


import json, joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, render_template, request
from pathlib import Path


app = Flask(__name__)


MODEL_DIR = Path('outputs/models')
WEBAPP_DIR = Path(__file__).resolve().parent


# Load model and config
BEST_MODEL = joblib.load(MODEL_DIR / 'XGBoost_C_full_integrated.joblib')


# Load region defaults (pre-computed median demographic/macro values)
with open(WEBAPP_DIR / 'model_artifacts' / 'region_lookup.json') as f:
    REGION_DEFAULTS = json.load(f)




@app.route('/', methods=['GET'])
def index():
    regions = sorted(REGION_DEFAULTS.keys())
    return render_template('index.html', regions=regions)




@app.route('/predict', methods=['POST'])
def predict():
    # Collect user inputs
    property_type = request.form['property_type']
    floor_area = float(request.form['floor_area'])
    num_rooms = int(request.form['num_rooms'])
    energy_rating = request.form['energy_rating']
    region = request.form['region']


    # Fetch region-level defaults
    defaults = REGION_DEFAULTS[region]


    # Build input DataFrame
    input_data = pd.DataFrame([{
        'property_type': property_type,
        'duration_label': 'Freehold',
        'floor_area': floor_area,
        'num_rooms': num_rooms,
        'energy_rating': energy_rating,
        'construction_age_band': 'England and Wales: 1991-2002',
        'is_new_build': 0,
        'year': 2024,
        'quarter': 1,
        **defaults,
    }])


    # Predict
    log_price = BEST_MODEL.predict(input_data)[0]
    predicted_price = np.exp(log_price)


    # SHAP explanation
    preprocessor = BEST_MODEL.named_steps['preprocessor']
    model = BEST_MODEL.named_steps['model']
    X_transformed = preprocessor.transform(input_data)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)


    # Top 5 drivers
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0]
    }).sort_values('shap_value', key=abs, ascending=False).head(5)


    drivers = shap_df.to_dict('records')


    return render_template('result.html',
        predicted_price=f'GBP {predicted_price:,.0f}',
        drivers=drivers,
        region=region,
        property_type=property_type,
    )




if __name__ == '__main__':
    app.run(debug=True, port=5000)
