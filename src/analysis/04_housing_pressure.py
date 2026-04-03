"""
Housing Pressure Index Analysis — Methodology Section 6
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from pathlib import Path


DATA_PATH = Path('data/merged/transactions_enriched.parquet')
OUT_DIR   = Path('outputs/hpi')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# GeoJSON for LA boundaries (download from ONS Open Geography)
LA_GEOJSON_URL = (
    'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/'
    'rest/services/Local_Authority_Districts_December_2023_'
    'Boundaries_UK_BFC/FeatureServer/0/query'
    '?where=1%3D1&outFields=*&outSR=4326&f=geojson'
)




def compute_hpi(df):
    """
    HPI = Population Change / Net Additional Dwellings
    per local authority per year.
    """
    # Need population_growth_pct, net_additions, population at LA level
    la_year = df.groupby(['laua', 'region_name', 'year']).agg({
        'population': 'first',
        'population_growth_pct': 'first',
        'net_additions': 'first',
        'price': 'median',
    }).reset_index()


    la_year['pop_change'] = (
        la_year['population_growth_pct'] / 100 * la_year['population']
    )
    la_year['hpi'] = la_year['pop_change'] / la_year['net_additions'].replace(0, np.nan)


    return la_year




def hpi_choropleth(la_year):
    """Interactive Folium map of mean HPI by local authority."""
    mean_hpi = la_year.groupby('laua')['hpi'].mean().reset_index()


    m = folium.Map(location=[52.5, -1.5], zoom_start=7)


    try:
        folium.Choropleth(
            geo_data=LA_GEOJSON_URL,
            data=mean_hpi,
            columns=['laua', 'hpi'],
            key_on='feature.properties.LAD23CD',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name='Housing Pressure Index (mean)',
            nan_fill_color='white',
        ).add_to(m)
    except Exception as e:
        print(f'Choropleth layer failed: {e}')
        print('Falling back to marker-based map.')


    m.save(str(OUT_DIR / 'hpi_choropleth.html'))
    print('HPI choropleth saved.')




def hpi_time_series(la_year):
    """Time series of HPI for selected local authorities."""
    # Pick a mix of high-pressure and low-pressure LAs
    mean_hpi = la_year.groupby(['laua', 'region_name'])['hpi'].mean()
    top_5 = mean_hpi.nlargest(5).index.get_level_values('laua')
    bot_5 = mean_hpi.nsmallest(5).index.get_level_values('laua')
    selected = list(top_5) + list(bot_5)


    plt.figure(figsize=(14, 7))
    for la in selected:
        subset = la_year[la_year['laua'] == la]
        region = subset['region_name'].iloc[0]
        plt.plot(subset['year'], subset['hpi'], label=f'{la} ({region})', alpha=0.7)


    plt.axhline(y=1, color='red', linestyle='--', label='HPI = 1 (demand = supply)')
    plt.xlabel('Year')
    plt.ylabel('Housing Pressure Index')
    plt.title('HPI Over Time: Highest vs Lowest Pressure LAs')
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'hpi_time_series.png', dpi=150, bbox_inches='tight')
    plt.close()




def main():
    df = pd.read_parquet(DATA_PATH)
    df = df[df['year'] >= 2015]
    la_year = compute_hpi(df)
    la_year.to_csv(OUT_DIR / 'hpi_summary_table.csv', index=False)
    hpi_choropleth(la_year)
    hpi_time_series(la_year)
    print(f'HPI analysis complete. Outputs in {OUT_DIR}')




if __name__ == '__main__':
    main()
