# UK Housing Valuation Model — Data Pipeline

Multi-dimensional real estate valuation model integrating property attributes,
regional demographics, and macroeconomic indicators. England, 1995–2024.

---

## Project Structure

```
uk_housing_model/
├── run_pipeline.py              # Master pipeline runner
├── requirements.txt
├── data/
│   ├── raw/                     # Downloaded source files (not committed to git)
│   │   ├── land_registry/       # pp-YYYY.csv files
│   │   ├── postcode/            # ONSPD_latest.csv
│   │   ├── macro/               # GVA, CPI, BoE rate, AWE xlsx/csv
│   │   └── demographics/        # MYE, migration, census xlsx
│   ├── processed/               # Cleaned parquet files per source
│   └── merged/                  # Final enriched dataset
│       └── transactions_enriched.parquet
├── src/
│   ├── config.py                # Paths, URLs, column names
│   ├── ingestion/
│   │   ├── land_registry.py     # HM Land Registry Price Paid
│   │   ├── postcode_lookup.py   # ONS postcode → region mapping
│   │   ├── macro_indicators.py  # BoE rate, GVA, CPI, AWE, housing supply
│   │   └── demographics.py      # Population, migration, Census 2021
│   └── processing/
│       └── merge_pipeline.py    # Joins all sources + feature engineering
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory analysis (to be created)
└── outputs/                     # Charts, reports, model results
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
# Full pipeline (downloads ~6–8 GB of data)
python run_pipeline.py

# Development mode — 5% sample, much faster
python run_pipeline.py --sample 0.05

# Only re-run the merge step (if data already downloaded)
python run_pipeline.py --steps merge
```

---

## Data Sources

| Dataset | Source | Size | Licence |
|---------|--------|------|---------|
| Price Paid Data | HM Land Registry | ~5 GB | OGL v3 |
| ONS Postcode Directory | ONS Open Geography | ~1 GB | OGL v3 |
| Regional GVA per head | ONS | ~5 MB | OGL v3 |
| CPI (CZMT) | ONS Time Series | ~1 MB | OGL v3 |
| Average Weekly Earnings | ONS Time Series | ~1 MB | OGL v3 |
| BoE Base Rate | Bank of England | ~1 MB | OGL v3 |
| Housing Supply (Table 122) | MHCLG | ~2 MB | OGL v3 |
| Population Estimates (MYE) | ONS | ~5 MB | OGL v3 |
| Net Migration | ONS | ~2 MB | OGL v3 |
| Census 2021 (NOMIS API) | ONS / NOMIS | API | OGL v3 |

All data is Open Government Licence (OGL v3) — free to use and redistribute with attribution.

---

## Manual Downloads (if automated fetch fails)

Some ONS files require manual download due to URL instability. The pipeline
will print clear instructions if any automated download fails.

### ONSPD (Postcode Directory) — most likely to need manual download
1. Go to: https://geoportal.statistics.gov.uk
2. Search: "ONS Postcode Directory"
3. Download latest CSV edition (~1 GB zip)
4. Extract main Data CSV and save to: `data/raw/postcode/ONSPD_latest.csv`

### ONS Regional GVA
1. Go to: https://www.ons.gov.uk/economy/grossdomesticproductgdp/datasets/regionalgrossvalueaddedbalancedbyindustry
2. Download the current edition XLSX
3. Save to: `data/raw/macro/regional_gva.xlsx`

### MHCLG Live Table 122
1. Go to: https://www.gov.uk/government/statistical-data-sets/live-tables-on-net-supply-of-housing
2. Download "Live Table 122"
3. Save to: `data/raw/macro/mhclg_live_table_122.xlsx`

---

## Final Dataset Schema

After `run_pipeline.py`, `data/merged/transactions_enriched.parquet` contains:

### Property-level (from Land Registry + Postcode lookup)
| Column | Description |
|--------|-------------|
| `price` | Sale price (£) |
| `log_price` | Natural log of price (model target) |
| `date_of_transfer` | Transaction date |
| `year`, `month`, `quarter` | Temporal fields |
| `property_type` | D/S/T/F/O |
| `property_type_label` | Detached / Semi-Detached / etc. |
| `is_new_build` | 1 = new build, 0 = established |
| `duration_label` | Freehold / Leasehold |
| `postcode` | Full postcode |
| `region_name` | English region |
| `laua` | Local Authority code |
| `lat`, `long` | Coordinates |

### Macroeconomic (monthly, joined on year+month)
| Column | Description |
|--------|-------------|
| `base_rate_pct` | BoE Bank Rate (%) |
| `cpi_index` | CPI all items index (2015=100) |
| `cpi_yoy_pct` | CPI year-on-year % change |
| `avg_weekly_earnings_gbp` | UK average weekly earnings (£) |

### Regional economic (annual, joined on region+year)
| Column | Description |
|--------|-------------|
| `gva_per_head_gbp` | Regional GVA per head (£) |
| `net_additions` | New dwellings (MHCLG) |

### Demographic (annual, joined on region+year)
| Column | Description |
|--------|-------------|
| `population` | Regional mid-year population |
| `population_growth_pct` | YoY population growth (%) |
| `net_migration_total` | Net migration into region |

### Census 2021 (cross-sectional, joined on region)
| Column | Description |
|--------|-------------|
| `owner_occupier_pct` | % households owner-occupied |
| `private_rental_pct` | % households private rented |
| `census_population_2021` | 2021 Census population |

### Derived / Engineered
| Column | Description |
|--------|-------------|
| `price_to_annual_income` | Price / annual earnings proxy |
| `housing_pressure_index` | Population growth / new supply |
| `decade` | Decade of transaction |
| `season` | Spring / Summer / Autumn / Winter |

---

## Next Steps (after pipeline runs)

1. **EDA**: `notebooks/01_eda.ipynb` — price distributions by region/type/time
2. **Baseline model**: Linear regression / Lasso — property features only
3. **Extended model**: Add macro + demographic variables; compare RMSE/MAE
4. **ML models**: Random Forest, XGBoost
5. **SHAP analysis**: Feature importance by variable group
6. **Web app**: Flask app for interactive predictions
