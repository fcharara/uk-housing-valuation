"""
Model Training — Methodology Sections 4.1, 4.2, 4.3
Trains 4 models x 3 feature sets = 12 model variants.
"""


import json, joblib, logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DATA_PATH = Path('data/merged/transactions_enriched.parquet')
OUT_DIR   = Path('outputs/models')
OUT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
TEST_SIZE = 0.2
ANALYSIS_START = 2015


# ── Feature Set Definitions (Methodology Table 2) ─────────────────────


STRUCTURAL_FEATURES = [
    'property_type', 'duration_label', 'floor_area', 'num_rooms',
    'energy_rating', 'construction_age_band', 'is_new_build',
    'year', 'quarter',
]


DEMOGRAPHIC_FEATURES = [
    'population_density', 'population_growth_pct', 'median_age',
    'household_count', 'net_migration_total', 'housing_pressure_index',
]


MACRO_FEATURES = [
    'base_rate_pct', 'cpi_yoy_pct', 'median_household_income',
    'unemployment_rate', 'mortgage_approvals', 'net_additions',
]


FEATURE_SETS = {
    'A_structural': STRUCTURAL_FEATURES,
    'B_structural_demo': STRUCTURAL_FEATURES + DEMOGRAPHIC_FEATURES,
    'C_full_integrated': STRUCTURAL_FEATURES + DEMOGRAPHIC_FEATURES + MACRO_FEATURES,
}


TARGET = 'log_price'


# Identify which columns are categorical vs continuous
CATEGORICAL = ['property_type', 'duration_label', 'energy_rating',
               'construction_age_band']
CONTINUOUS = [f for f in STRUCTURAL_FEATURES + DEMOGRAPHIC_FEATURES + MACRO_FEATURES
              if f not in CATEGORICAL]




def load_and_prepare():
    df = pd.read_parquet(DATA_PATH)
    df = df[df['year'] >= ANALYSIS_START]


    # Outlier removal: 1st and 99th percentile (Methodology 2.3)
    p1, p99 = df['price'].quantile(0.01), df['price'].quantile(0.99)
    df = df[(df['price'] >= p1) & (df['price'] <= p99)]


    # Missing value imputation (Methodology 2.3)
    # Forward-fill lower-frequency variables
    fill_cols = ['median_household_income', 'unemployment_rate',
                 'gva_per_head_gbp', 'net_additions']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df.groupby('region_name')[col].ffill()


    # Median imputation by property_type + laua for remaining NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0 and col != TARGET:
            df[col] = df.groupby(['property_type', 'region_name'])[col].transform(
                lambda x: x.fillna(x.median())
            )


    # Construction age band: create 'missing' category
    if 'construction_age_band' in df.columns:
        df['construction_age_band'] = df['construction_age_band'].fillna('UNKNOWN')


    df.dropna(subset=[TARGET], inplace=True)
    log.info(f'Prepared dataset: {len(df):,} rows')
    return df




def build_preprocessor(feature_list):
    """Build sklearn ColumnTransformer for the given feature list."""
    cat_features = [f for f in feature_list if f in CATEGORICAL]
    num_features = [f for f in feature_list if f not in CATEGORICAL]


    transformers = []
    if num_features:
        transformers.append(('num', StandardScaler(), num_features))
    if cat_features:
        transformers.append((
            'cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            cat_features
        ))


    return ColumnTransformer(transformers, remainder='drop')




def train_all_models(df):
    """Train 4 models x 3 feature sets = 12 variants."""
    results = []


    for fs_name, features in FEATURE_SETS.items():
        available = [f for f in features if f in df.columns]
        missing = set(features) - set(available)
        if missing:
            log.warning(f'{fs_name}: missing features {missing}, skipping them')


        X = df[available]
        y = df[TARGET]


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED
        )


        preprocessor = build_preprocessor(available)


        models = {
            'OLS': LinearRegression(),
            'Lasso': LassoCV(cv=3, random_state=SEED, max_iter=5000),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=15, min_samples_leaf=20,
                random_state=SEED, n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, n_jobs=-1
            ),
        }


        for model_name, model in models.items():
            log.info(f'Training {model_name} with {fs_name}...')


            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model),
            ])


            
            pipe.fit(X_train, y_train)


            # Evaluate on test set
            y_pred = pipe.predict(X_test)


            # Convert back from log space for interpretable metrics
            y_test_prices = np.exp(y_test)
            y_pred_prices = np.exp(y_pred)


            mae  = mean_absolute_error(y_test_prices, y_pred_prices)
            rmse = root_mean_squared_error(y_test_prices, y_pred_prices)
            r2   = r2_score(y_test, y_pred)  # R2 on log scale


            log.info(f'  MAE: GBP{mae:,.0f} | RMSE: GBP{rmse:,.0f} | R2: {r2:.4f}')


            results.append({
                'model': model_name,
                'feature_set': fs_name,
                'mae_gbp': mae,
                'rmse_gbp': rmse,
                'r2': r2,
            })


            # Save model artifact
            artifact_name = f'{model_name}_{fs_name}'
            joblib.dump(pipe, OUT_DIR / f'{artifact_name}.joblib')


    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / 'model_comparison_table.csv', index=False)
    print('\nModel Comparison:')
    print(results_df.to_string(index=False))
    return results_df




def main():
    df = load_and_prepare()
    results = train_all_models(df)
    print(f'\nAll 12 models trained. Artifacts saved to {OUT_DIR}')




if __name__ == '__main__':
    main()
