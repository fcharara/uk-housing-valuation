# UK Housing Valuation Model

Multi-dimensional real estate valuation model integrating property attributes,
EPC structural data, regional demographics, and macroeconomic indicators.
England, 2015–2024.

**Research Question:** Does incorporating demographic and macroeconomic variables
improve predictive accuracy over traditional structural-only hedonic pricing models?

**Key Finding:** Adding demographic variables improves R² from 0.46 to 0.64
(+39% relative improvement), while macro variables add marginal further gains.

---

## Results Summary

| Model | A: Structural | B: + Demographic | C: + Macro |
|-------|:---:|:---:|:---:|
| OLS | 0.287 | 0.495 | 0.550 |
| Lasso | 0.286 | 0.495 | 0.549 |
| Random Forest | 0.459 | 0.638 | 0.640 |
| XGBoost | 0.459 | 0.639 | 0.640 |

Trained on 8.07M transactions. Best model: XGBoost with full integrated features (R² = 0.64).

---

## Project Structure
uk-housing-valuation/
├── run_pipeline.py                    # Master data pipeline
├── requirements.txt
├── data/
│   ├── raw/                           # Source files (not in git)
│   │   ├── land_registry/             # HM Land Registry Price Paid CSVs
│   │   ├── epc/                       # EPC certificates (~300 LA folders)
│   │   ├── postcode/                  # ONS Postcode Directory
│   │   ├── macro/                     # GVA, CPI, BoE rate, AWE, ASHE, unemployment
│   │   ├── demographics/              # MYE population estimates
│   │   └── geo/                       # LA boundary GeoJSON
│   ├── processed/                     # Cleaned parquets per source
│   └── merged/
│       └── transactions_enriched.parquet  # Final dataset (27.2M rows, 50 cols)
├── src/
│   ├── config.py                      # Paths, URLs, parameters
│   ├── ingestion/
│   │   ├── land_registry.py           # HM Land Registry download + clean
│   │   ├── epc_data.py                # EPC certificate processing
│   │   ├── postcode_lookup.py         # ONSPD postcode → region mapping
│   │   ├── macro_indicators.py        # BoE rate, GVA, CPI, AWE, housing supply
│   │   └── demographics.py            # Population, migration, Census 2021
│   ├── processing/
│   │   ├── merge_pipeline.py          # Joins all sources + feature engineering
│   │   └── epc_matching_fast.py       # Vectorised EPC-to-transaction matching
│   ├── analysis/
│   │   ├── 01_eda.py                  # Exploratory data analysis
│   │   ├── 02_model_training.py       # 4 models × 3 feature sets = 12 variants
│   │   ├── 03_evaluation.py           # Comparison tables, SHAP, PDPs
│   │   └── 04_housing_pressure.py     # HPI choropleth maps
│   └── webapp/
│       ├── app.py                     # Flask web application
│       ├── templates/                 # HTML templates
│       └── model_artifacts/           # Saved model config
└── outputs/
├── eda/                           # Distribution plots, correlations, VIF
├── models/                        # Model comparison CSV + joblib artifacts
├── evaluation/                    # SHAP plots, PDPs, sensitivity analysis
└── hpi/                           # Choropleth maps, HPI time series

---

## Quick Start

### 1. Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data pipeline (requires manual EPC download)
```bash
# Download EPC data from https://epc.opendatacommunities.org/
# Place certificates.csv files in data/raw/epc/

# Run full pipeline
python run_pipeline.py
```

### 3. EPC matching (postcode + house number, ~8 min)
```bash
python src/processing/epc_matching_fast.py
```

### 4. Analysis pipeline
```bash
python src/analysis/01_eda.py              # EDA plots
python src/analysis/02_model_training.py   # Train 12 models (~3 hours)
python src/analysis/03_evaluation.py       # SHAP + evaluation (~15 min)
python src/analysis/04_housing_pressure.py # HPI maps
```

### 5. Web application
```bash
python src/webapp/app.py                   # Opens at http://127.0.0.1:5000
```

---

## Data Sources

| Dataset | Source | Variables | Granularity |
|---------|--------|-----------|-------------|
| Price Paid Data | HM Land Registry | Price, type, tenure, postcode | Transaction-level |
| EPC Register | DLUHC | Floor area, rooms, energy rating, age band | Property-level |
| ONSPD | ONS Geography | Postcode → region/LA mapping | Postcode-level |
| Base Rate | Bank of England | Bank rate (%) | Monthly |
| CPI | ONS (D7BT) | Consumer price index, YoY % | Monthly |
| Avg Weekly Earnings | ONS (KAB9) | Earnings (£) | Monthly |
| Regional GVA | ONS | Gross value added (£M) | Region × Year |
| Housing Supply | MHCLG Live Table 122 | Net additional dwellings | LA × Year |
| Population Estimates | ONS MYE | Population, density, median age | Region × Year |
| Migration | ONS MYE3 | Net internal migration | Region (cross-section) |
| Unemployment | ONS Model-based | Unemployment rate (%) | LA (cross-section) |
| Median Income | ONS ASHE Table 8.7a | Gross annual pay (£) | Region (cross-section) |
| Census 2021 | ONS/NOMIS | Tenure structure (%) | Region (cross-section) |

All data is Open Government Licence (OGL v3).

---

## Methodology

The project follows a six-stage methodology aligned with the capstone thesis:

1. **Data Collection** — 13 public datasets merged into a single analytical file
2. **Pre-processing** — EPC fuzzy/exact matching, outlier removal (1st/99th percentile), median imputation, log transformation of prices
3. **EDA** — Distribution plots, correlation matrix, VIF multicollinearity check
4. **Model Training** — OLS, Lasso, Random Forest, XGBoost across three feature sets (structural only, + demographic, + macro)
5. **Evaluation** — MAE/RMSE/R² comparison, SHAP analysis (beeswarm, force plots, PDPs), outlier sensitivity check
6. **Deployment** — Flask web app with SHAP-driven price explanations

---

## Feature Sets (Methodology Table 2)

| Set | Features |
|-----|----------|
| **A: Structural** | property_type, tenure, floor_area, num_rooms, energy_rating, construction_age_band, is_new_build, year, quarter |
| **B: + Demographic** | All A + population_density, population_growth_pct, median_age, net_migration_total, housing_pressure_index |
| **C: + Macro** | All B + base_rate_pct, cpi_yoy_pct, median_household_income, unemployment_rate, net_additions, gva_total_millions |

---

## Technical Stack

- **Python 3.11** — pandas, numpy, scikit-learn, XGBoost, SHAP
- **Visualisation** — matplotlib, seaborn, folium
- **Web** — Flask
- **Data formats** — Parquet (compressed), CSV, XLSX, ODS

---

## Author

Bachelor's capstone project, 2024–2025.
