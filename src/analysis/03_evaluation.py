"""
Model Evaluation and SHAP Analysis — Methodology Section 5
"""


import joblib, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sys, joblib, json
from pathlib import Path
from sklearn.model_selection import train_test_split


MODEL_DIR = Path('outputs/models')
OUT_DIR   = Path('outputs/evaluation')
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path('data/merged/transactions_enriched.parquet')


SEED = 42
ANALYSIS_START = 2015




def load_results():
    return pd.read_csv(MODEL_DIR / 'model_comparison_table.csv')




def plot_model_comparison(results_df):
    """Grouped bar chart comparing MAE, RMSE, R2 across all 12 models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))


    for ax, metric, title in zip(
        axes,
        ['mae_gbp', 'rmse_gbp', 'r2'],
        ['MAE (GBP)', 'RMSE (GBP)', 'R-squared']
    ):
        pivot = results_df.pivot(
            index='model', columns='feature_set', values=metric
        )
        pivot.plot(kind='bar', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Feature Set', fontsize=8)


    plt.tight_layout()
    plt.savefig(OUT_DIR / 'model_comparison_chart.png', dpi=150)
    plt.close()




def pairwise_improvement(results_df):
    """Compute % improvement from A->B and B->C for each model."""
    improvements = []
    for model in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model]
        a = model_results[model_results['feature_set'] == 'A_structural'].iloc[0]
        b = model_results[model_results['feature_set'] == 'B_structural_demo'].iloc[0]
        c = model_results[model_results['feature_set'] == 'C_full_integrated'].iloc[0]


        improvements.append({
            'model': model,
            'mae_improvement_A_to_B_pct': (a['mae_gbp'] - b['mae_gbp']) / a['mae_gbp'] * 100,
            'mae_improvement_B_to_C_pct': (b['mae_gbp'] - c['mae_gbp']) / b['mae_gbp'] * 100,
            'r2_improvement_A_to_B': b['r2'] - a['r2'],
            'r2_improvement_B_to_C': c['r2'] - b['r2'],
        })


    imp_df = pd.DataFrame(improvements)
    imp_df.to_csv(OUT_DIR / 'pairwise_improvement.csv', index=False)
    print('Pairwise improvements:')
    print(imp_df.to_string(index=False))




def shap_analysis():
    """SHAP analysis on the best tree-based model (Methodology Section 5)."""
    # Load the best model (XGBoost full integrated, typically best)
    model_path = MODEL_DIR / 'XGBoost_C_full_integrated.joblib'
    if not model_path.exists():
        model_path = MODEL_DIR / 'RandomForest_C_full_integrated.joblib'
    pipe = joblib.load(model_path)


    # Load and prepare test data
    df = pd.read_parquet(DATA_PATH)
    df = df[df['year'] >= ANALYSIS_START]
    p1, p99 = df['price'].quantile(0.01), df['price'].quantile(0.99)
    df = df[(df['price'] >= p1) & (df['price'] <= p99)]


    # Use C feature set
    sys.path.append('src/analysis')
    from importlib.machinery import SourceFileLoader
    mod = SourceFileLoader('model_training', 'src/analysis/02_model_training.py').load_module()
    FEATURE_SETS = mod.FEATURE_SETS
    TARGET = mod.TARGET
    features = [f for f in FEATURE_SETS['C_full_integrated'] if f in df.columns]
    X = df[features]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


    # Transform features using the pipeline's preprocessor
    preprocessor = pipe.named_steps['preprocessor']
    X_test_transformed = preprocessor.transform(X_test)


    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()


    # SHAP explainer
    model = pipe.named_steps['model']
    explainer = shap.TreeExplainer(model)


    # Use a subsample for SHAP (full dataset too slow)
    sample_size = min(5000, len(X_test_transformed))
    X_shap = X_test_transformed[:sample_size]
    shap_values = explainer.shap_values(X_shap)


    # Global SHAP summary (beeswarm)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'shap_summary_global.png', dpi=150, bbox_inches='tight')
    plt.close()


    # SHAP bar chart grouped by variable category
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_names, 'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    shap_df.to_csv(OUT_DIR / 'shap_feature_importance.csv', index=False)


    plt.figure(figsize=(10, 12))
    plt.barh(shap_df['feature'][:20], shap_df['mean_abs_shap'][:20])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 20 Features by SHAP Importance')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'shap_feature_importance.png', dpi=150)
    plt.close()


    # Partial Dependence Plots
    for feature in ['floor_area', 'base_rate_pct', 'population_density']:
        if feature in feature_names:
            idx = list(feature_names).index(feature)
            shap.dependence_plot(idx, shap_values, X_shap,
                                 feature_names=feature_names, show=False)
            plt.savefig(OUT_DIR / f'pdp_{feature}.png', dpi=150,
                        bbox_inches='tight')
            plt.close()




def main():
    results = load_results()
    plot_model_comparison(results)
    pairwise_improvement(results)
    shap_analysis()
    print(f'\nEvaluation complete. Outputs saved to {OUT_DIR}')




if __name__ == '__main__':
    main()
