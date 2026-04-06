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
import webbrowser, threading

app = Flask(__name__)
MODEL_DIR = Path('outputs/models')
WEBAPP_DIR = Path(__file__).resolve().parent

BEST_MODEL = joblib.load(MODEL_DIR / 'XGBoost_C_full_integrated.joblib')

with open(WEBAPP_DIR / 'model_artifacts' / 'region_lookup.json') as f:
    REGION_DEFAULTS = json.load(f)

TYPE_LABELS = {'D': 'Detached', 'S': 'Semi-Detached', 'T': 'Terraced', 'F': 'Flat / Maisonette'}

FEATURE_LABELS = {
    'num__floor_area': 'Floor Area',
    'num__num_rooms': 'Number of Rooms',
    'num__is_new_build': 'New Build',
    'num__year': 'Transaction Year',
    'num__quarter': 'Quarter',
    'num__population_density': 'Population Density',
    'num__population_growth_pct': 'Population Growth',
    'num__median_age': 'Median Age',
    'num__net_migration_total': 'Net Migration',
    'num__housing_pressure_index': 'Housing Pressure Index',
    'num__base_rate_pct': 'BoE Base Rate',
    'num__cpi_yoy_pct': 'CPI Inflation',
    'num__median_household_income': 'Median Income',
    'num__unemployment_rate': 'Unemployment Rate',
    'num__net_additions': 'Net Housing Additions',
    'num__gva_total_millions': 'Regional GVA',
}

@app.route('/', methods=['GET'])
def index():
    regions = sorted(REGION_DEFAULTS.keys())
    return render_template('index.html', regions=regions)

@app.route('/predict', methods=['POST'])
def predict():
    property_type = request.form['property_type']
    floor_area = float(request.form['floor_area'])
    num_rooms = int(request.form['num_rooms'])
    energy_rating = request.form['energy_rating']
    region = request.form['region']

    defaults = REGION_DEFAULTS[region]

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

    log_price = BEST_MODEL.predict(input_data)[0]
    predicted_price = np.exp(log_price)

    preprocessor = BEST_MODEL.named_steps['preprocessor']
    model = BEST_MODEL.named_steps['model']
    X_transformed = preprocessor.transform(input_data)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0]
    }).sort_values('shap_value', key=abs, ascending=False).head(5)

    drivers = []
    for _, row in shap_df.iterrows():
        label = FEATURE_LABELS.get(row['feature'], row['feature'].replace('num__','').replace('cat__','').replace('_',' ').title())
        drivers.append({'feature_label': label, 'shap_value': row['shap_value']})

    return render_template('result.html',
        predicted_price=f'GBP {predicted_price:,.0f}',
        drivers=drivers,
        region=region,
        property_type=property_type,
        property_type_label=TYPE_LABELS.get(property_type, property_type),
    )

if __name__ == '__main__':
    threading.Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=False, port=5000)
