"""
Exploratory Data Analysis — Methodology Section 3
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_PATH = Path('data/merged/transactions_enriched.parquet')
OUT_DIR   = Path('outputs/eda')
OUT_DIR.mkdir(parents=True, exist_ok=True)


ANALYSIS_START = 2015




def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df[df['year'] >= ANALYSIS_START]
    return df




def summary_statistics(df):
    """Table of descriptive statistics for all variables."""
    desc = df.describe(include='all').T
    desc.to_csv(OUT_DIR / 'summary_statistics.csv')
    print(f'Summary statistics saved: {len(desc)} variables')




def price_distributions(df):
    """Histograms of raw price and log price."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['price'], bins=100, color='steelblue', edgecolor='white')
    axes[0].set_title('Transaction Price Distribution')
    axes[0].set_xlabel('Price (GBP)')
    axes[1].hist(df['log_price'], bins=100, color='darkorange', edgecolor='white')
    axes[1].set_title('Log Price Distribution')
    axes[1].set_xlabel('Log Price')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'price_distribution.png', dpi=150)
    plt.close()




def price_by_region(df):
    """Box plots of log price by English region."""
    order = df.groupby('region_name')['log_price'].median().sort_values().index
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='region_name', y='log_price', order=order,
                fliersize=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.title('Log Price by Region')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'price_by_region.png', dpi=150)
    plt.close()




def price_by_type(df):
    """Box plots by property type."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='property_type_label', y='log_price')
    plt.title('Log Price by Property Type')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'price_by_type.png', dpi=150)
    plt.close()




def price_time_series(df):
    """Median price over time by region."""
    ts = df.groupby(['year', 'quarter', 'region_name'])['price'].median().reset_index()
    ts['date'] = pd.to_datetime(
        ts['year'].astype(str) + '-' + (ts['quarter'] * 3).astype(str) + '-01'
    )
    plt.figure(figsize=(14, 7))
    for region in ts['region_name'].unique():
        subset = ts[ts['region_name'] == region]
        plt.plot(subset['date'], subset['price'], label=region)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Median Transaction Price by Region Over Time')
    plt.ylabel('Median Price (GBP)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'price_time_series.png', dpi=150)
    plt.close()




def correlation_matrix(df):
    """Pearson correlation heatmap of all numerical variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()


    # Flag high correlations
    high_corr = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            if abs(corr.iloc[i, j]) > 0.85:
                high_corr.append(
                    (corr.index[i], corr.columns[j], corr.iloc[i, j])
                )
    if high_corr:
        print('High correlations (|r| > 0.85):')
        for a, b, r in high_corr:
            print(f'  {a} <-> {b}: {r:.3f}')


    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'correlation_matrix.png', dpi=150)
    plt.close()




def compute_vif(df):
    """Variance Inflation Factors for multicollinearity check."""
    numeric = df.select_dtypes(include=[np.number]).dropna()
    # Exclude target variables and IDs
    exclude = ['price', 'log_price', 'lat', 'long', 'transaction_id']
    cols = [c for c in numeric.columns if c not in exclude]
    X = numeric[cols].copy()
    
    # Remove infinite values and ensure all float
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X.astype(float)
    
    # Sample if too large (VIF on 8M rows is very slow)
    if len(X) > 50000:
        X = X.sample(50000, random_state=42)
    
    vif_data = pd.DataFrame({
        'Variable': cols,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(len(cols))]
    }).sort_values('VIF', ascending=False)

    vif_data.to_csv(OUT_DIR / 'vif_table.csv', index=False)
    print('VIF results (top 10):')
    print(vif_data.head(10).to_string())


def main():
    print('Loading data...')
    df = load_data()
    print(f'Dataset: {len(df):,} rows, {len(df.columns)} columns')


    summary_statistics(df)
    price_distributions(df)
    price_by_region(df)
    price_by_type(df)
    price_time_series(df)
    correlation_matrix(df)
    compute_vif(df)


    print(f'\nEDA complete. Outputs saved to {OUT_DIR}')




if __name__ == '__main__':
    main()
