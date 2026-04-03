# config.py — Central configuration for the UK Housing Valuation Pipeline
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
MERGED_DIR = DATA_DIR / "merged"
OUTPUT_DIR = BASE_DIR / "outputs"

for d in [RAW_DIR, PROC_DIR, MERGED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Geographic scope ───────────────────────────────────────────────────────────
COUNTRY = "England"

# ONS region codes (English regions only)
ENGLISH_REGIONS = {
    "E12000001": "North East",
    "E12000002": "North West",
    "E12000003": "Yorkshire and The Humber",
    "E12000004": "East Midlands",
    "E12000005": "West Midlands",
    "E12000006": "East of England",
    "E12000007": "London",
    "E12000008": "South East",
    "E12000009": "South West",
}

# ── Time range ─────────────────────────────────────────────────────────────────
# Land Registry data available from 1995
START_YEAR = 1995
END_YEAR   = 2024

# ── Data source URLs ───────────────────────────────────────────────────────────

# 1. HM Land Registry — Price Paid Data (full, ~5GB CSV)
#    Individual yearly files are smaller and easier to handle
LR_BASE_URL = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
LR_FULL_URL = f"{LR_BASE_URL}/pp-complete.csv"
LR_YEARLY_URL = lambda year: f"{LR_BASE_URL}/pp-{year}.csv"

# Column names for Land Registry Price Paid Data
LR_COLUMNS = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",       # D=Detached, S=Semi, T=Terraced, F=Flat, O=Other
    "old_new",             # Y=New build, N=Established
    "duration",            # F=Freehold, L=Leasehold
    "paon",                # Primary addressable object name (house number)
    "saon",                # Secondary addressable object name (flat number)
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category",        # A=Standard, B=Additional
    "record_status",       # A=Addition, C=Change, D=Delete
]

# 2. ONS Postcode Directory — maps postcodes to regions/LAs
ONS_PCD_URL = "https://www.arcgis.com/sharing/rest/content/items/dc23a64fa2db4c2fb946a9b0f5291e3b/data"
# Fallback: download from ONS geography portal manually
ONS_PCD_MANUAL = "https://geoportal.statistics.gov.uk/datasets/ons-postcode-directory-latest-centroids"

# 3. ONS Regional Economic Indicators
#    Regional GVA (balanced) — Table 3
ONS_GVA_URL = "https://www.ons.gov.uk/file?uri=/economy/grossdomesticproductgdp/datasets/regionalgrossvalueaddedbalancedbyindustry/current/regionalgvabalancedbyindustry.xlsx"

#    Regional unemployment rate (model-based)
ONS_UNEMP_URL = "https://www.ons.gov.uk/generator?format=csv&uri=/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms"

# 4. Bank of England — Base Rate
BOE_BASE_RATE_URL = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp?Travel=NIxIRxSUx&FromSeries=1&ToSeries=50&DAT=RNG&FD=1&FM=Jan&FY=1975&TD=31&TM=Dec&TY=2025&VFD=Y&html.x=66&html.y=26&C=C02&Filter=N"
# Simpler CSV download:
BOE_BASE_RATE_CSV = "https://www.bankofengland.co.uk/boeapps/database/Bank-Stats.asp?Travel=NIxIRxSUx&C=C02&G0Xtop.x=1"

# 5. ONS Population Estimates — Regional mid-year estimates
ONS_POP_URL = "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland/mid2023/ukpopestimatesmid2023on2021geographyfinal.xlsx"

# 6. ONS Census 2021 — Regional demographic profiles
ONS_CENSUS_BASE = "https://www.nomisweb.co.uk/api/v01/dataset"

# 7. MHCLG Housing Supply — Live Table 122 (net additions)
MHCLG_SUPPLY_URL = "https://assets.publishing.service.gov.uk/media/65f0f5e6da5c3500138d76c0/Live_Table_122.xlsx"

# ── Processing parameters ──────────────────────────────────────────────────────
CHUNK_SIZE        = 500_000   # rows per chunk when reading large CSVs
SAMPLE_FRAC       = None      # Set to e.g. 0.1 for 10% sample during dev; None = full data
PARQUET_COMPRESS  = "snappy"  # Compression for saved parquet files


# ── Analysis scope ──────────────────────────────────────────
ANALYSIS_START_YEAR = 2015   # EPC data availability cutoff
ANALYSIS_END_YEAR   = 2024


# ── EPC Data ────────────────────────────────────────────────
# EPC data must be downloaded manually from:
# https://epc.opendatacommunities.org/
# Register for free, download 'Domestic EPCs' for all of England
# Save the ZIP files to: data/raw/epc/
EPC_RAW_DIR = RAW_DIR / 'epc'


# ── Additional macro URLs ───────────────────────────────────
# BoE mortgage approvals (series AMZJ)
BOE_MORTGAGE_APPROVALS_URL = (
    'https://www.bankofengland.co.uk/boeapps/database/'
    'fromshowcolumns.asp?Travel=NIxIRxSUx&FromSeries=1'
    '&ToSeries=50&DAT=RNG&FD=1&FM=Jan&FY=1993'
    '&TD=31&TM=Dec&TY=2025&VFD=Y&html.x=66&html.y=26'
    '&C=AMZJ&Filter=N'
)


# NOMIS API base for local authority level data
NOMIS_BASE = 'https://www.nomisweb.co.uk/api/v01/dataset'
