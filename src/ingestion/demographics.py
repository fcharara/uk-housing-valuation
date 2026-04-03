# ingestion/demographics.py
"""
Downloads and processes ONS demographic data by English region.

Indicators collected:
  1. Mid-year population estimates by region     (annual, 1991–present)
  2. Internal & international net migration      (annual, by region, 2001–present)
  3. Household projections                       (ONS, England, by region)
  4. Census 2021 key statistics                  (age structure, tenure, density)

All outputs saved as parquet, indexed by [region_name, year].
"""

import sys
import io
import logging
import requests
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, PROC_DIR, ENGLISH_REGIONS, PARQUET_COMPRESS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEMO_RAW_DIR  = RAW_DIR  / "demographics"
DEMO_PROC_DIR = PROC_DIR / "demographics"
DEMO_RAW_DIR.mkdir(parents=True, exist_ok=True)
DEMO_PROC_DIR.mkdir(parents=True, exist_ok=True)

ENGLISH_REGION_NAMES = set(ENGLISH_REGIONS.values())


def _get(url: str, timeout: int = 120) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def _print_manual(name: str, url: str, dest: Path, notes: str = ""):
    print(f"\n{'='*60}")
    print(f"MANUAL DOWNLOAD REQUIRED: {name}")
    print(f"{'='*60}")
    print(f"URL : {url}")
    print(f"Save: {dest}")
    if notes:
        print(f"Note: {notes}")
    print(f"{'='*60}\n")


# ── 1. Mid-Year Population Estimates ──────────────────────────────────────────

def fetch_population_estimates(overwrite: bool = False) -> pd.DataFrame:
    """
    ONS mid-year population estimates (MYE) by English region, 1991–latest.
    Returns: [region_name, year, population, population_growth_pct]
    """
    out_path = DEMO_PROC_DIR / "population_estimates.parquet"
    if out_path.exists() and not overwrite:
        log.info("Population estimates: already processed.")
        return pd.read_parquet(out_path)

    # ONS MYE time series — regional totals
    # Table MYE5 contains historical MYE by region
    url = (
        "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration"
        "/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland"
        "/mid2023/ukpopestimatesmid2023on2021geographyfinal.xlsx"
    )
    raw_path = DEMO_RAW_DIR / "mye_regional.xlsx"

    log.info("Fetching mid-year population estimates...")
    try:
        r = _get(url)
        with open(raw_path, "wb") as f:
            f.write(r.content)

        xls = pd.ExcelFile(raw_path)
        # Find MYE5 or similar sheet with regional breakdown
        target_sheet = next(
            (s for s in xls.sheet_names if "MYE5" in s.upper() or "REGION" in s.upper()),
            xls.sheet_names[0]
        )

        raw = pd.read_excel(raw_path, sheet_name=target_sheet, header=None)
        _parse_and_save_mye(raw, out_path)

    except Exception as e:
        log.warning(f"Population estimates fetch failed ({e}). Manual download required.")
        _print_manual(
            "ONS Mid-Year Population Estimates",
            "https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland",
            raw_path,
            "Sheet MYE5 or similar; region rows, year columns; total persons"
        )
        # Return empty placeholder so pipeline can continue
        return pd.DataFrame(columns=["region_name", "year", "population", "population_growth_pct"])

    return pd.read_parquet(out_path)


def _parse_and_save_mye(raw: pd.DataFrame, out_path: Path):
    """Parse the wide MYE format and save as tidy parquet."""
    year_row = next(
        (i for i, row in raw.iterrows()
         if sum(1 for v in row if str(v).strip().isdigit() and 1990 < int(str(v).strip()) < 2030) > 3),
        None
    )
    if year_row is None:
        raise ValueError("Cannot locate year header row in MYE data")

    header = raw.iloc[year_row]
    data   = raw.iloc[year_row + 1:].copy()
    data.columns = header

    first_col = data.columns[0]
    data = data.rename(columns={first_col: "region_name"})
    data = data[data["region_name"].isin(ENGLISH_REGION_NAMES)]

    year_cols = [c for c in data.columns if str(c).strip().isdigit() and 1990 < int(str(c).strip()) < 2030]
    df = data[["region_name"] + year_cols].melt(
        id_vars="region_name", var_name="year", value_name="population"
    )
    df["year"]       = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values(["region_name", "year"], inplace=True)

    # Year-on-year population growth %
    df["population_growth_pct"] = (
        df.groupby("region_name")["population"].pct_change() * 100
    )

    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    log.info(f"Population estimates: {len(df):,} rows → {out_path}")


# ── 2. Net Migration by Region ────────────────────────────────────────────────

def fetch_migration(overwrite: bool = False) -> pd.DataFrame:
    """
    ONS internal and international net migration by English region.
    Returns: [region_name, year, net_internal_migration, net_international_migration, net_migration_total]

    Note: Migration data by region is published in the 'Components of Change' series.
    Direct URL download is unreliable; manual download instructions provided.
    """
    out_path = DEMO_PROC_DIR / "migration.parquet"
    if out_path.exists() and not overwrite:
        log.info("Migration: already processed.")
        return pd.read_parquet(out_path)

    raw_path = DEMO_RAW_DIR / "migration_regional.xlsx"

    if not raw_path.exists():
        _print_manual(
            "ONS Regional Net Migration",
            "https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/estimatesofthepopulationforenglandandwales",
            raw_path,
            "Download 'Components of Change' by region; "
            "columns needed: region, year, net_internal_migration, net_international_migration"
        )
        log.warning("Migration data not available — skipping. Place file manually.")
        return pd.DataFrame(columns=["region_name", "year", "net_internal_migration",
                                     "net_international_migration", "net_migration_total"])

    log.info("Processing migration data...")
    try:
        raw = pd.read_excel(raw_path, header=None)
        year_row = next(
            (i for i, row in raw.iterrows()
             if sum(1 for v in row if str(v).strip().isdigit() and 2000 < int(str(v).strip()) < 2030) > 3),
            None
        )
        if year_row is None:
            raise ValueError("Cannot find year header in migration file")

        header = raw.iloc[year_row]
        data   = raw.iloc[year_row + 1:].copy()
        data.columns = header

        first_col = data.columns[0]
        data = data.rename(columns={first_col: "region_name"})
        data = data[data["region_name"].isin(ENGLISH_REGION_NAMES)]

        year_cols = [c for c in data.columns if str(c).strip().isdigit() and 2000 < int(str(c).strip()) < 2030]
        df = data[["region_name"] + year_cols].melt(
            id_vars="region_name", var_name="year", value_name="net_migration_total"
        )
        df["year"]               = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["net_migration_total"] = pd.to_numeric(df["net_migration_total"], errors="coerce")
        df.dropna(inplace=True)

        df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
        log.info(f"Migration: {len(df):,} rows → {out_path}")

    except Exception as e:
        log.warning(f"Migration processing failed: {e}")
        return pd.DataFrame()

    return pd.read_parquet(out_path)


# ── 3. Census 2021 Regional Demographics ─────────────────────────────────────

def fetch_census_2021(overwrite: bool = False) -> pd.DataFrame:
    """
    Fetches key Census 2021 statistics at English region level via NOMIS API.
    Covers: age structure (median age), tenure (owner-occupier %, private rental %),
            household size, population density.

    Returns: [region_name, median_age, owner_occupier_pct, private_rental_pct,
              avg_household_size, population_density_per_km2]

    Note: This is a CROSS-SECTIONAL snapshot (2021 only).
          It is merged to all transaction years as a regional structural control.
    """
    out_path = DEMO_PROC_DIR / "census_2021_regional.parquet"
    if out_path.exists() and not overwrite:
        log.info("Census 2021: already processed.")
        return pd.read_parquet(out_path)

    # NOMIS API — TS007 (Age by single year), TS054 (Tenure)
    # Using the bulk table download approach for reliability
    nomis_base = "https://www.nomisweb.co.uk/api/v01/dataset"

    # Region geography codes for NOMIS (English regions = TYPE480)
    region_geo = "2013265927TYPE480"

    records = {}

    # ── Median age from TS007A ────────────────────────────────────────────────
    log.info("Fetching Census 2021: age structure (TS007A)...")
    try:
        url = (
            f"{nomis_base}/NM_2083_1.data.csv"
            f"?geography={region_geo}"
            f"&c2021_age_92=0"   # total
            f"&measures=20100"
            f"&select=geography_name,obs_value"
        )
        r = _get(url)
        df_age = pd.read_csv(io.StringIO(r.text))
        df_age.columns = df_age.columns.str.lower()
        # median_age not directly in TS007A — store total population for density calc
        for _, row in df_age.iterrows():
            name = row.get("geography_name", "")
            if name in ENGLISH_REGION_NAMES:
                records.setdefault(name, {})["census_population_2021"] = row.get("obs_value")
    except Exception as e:
        log.warning(f"Census age fetch failed: {e}")

    # ── Tenure from TS054 ─────────────────────────────────────────────────────
    log.info("Fetching Census 2021: tenure (TS054)...")
    try:
        url = (
            f"{nomis_base}/NM_2072_1.data.csv"
            f"?geography={region_geo}"
            f"&c2021_tenure_9=0,2,5"  # 0=total, 2=owner-occupied, 5=private rented
            f"&measures=20301"          # % of total
            f"&select=geography_name,c2021_tenure_9_name,obs_value"
        )
        r = _get(url)
        df_tenure = pd.read_csv(io.StringIO(r.text))
        df_tenure.columns = df_tenure.columns.str.lower()

        for _, row in df_tenure.iterrows():
            name   = row.get("geography_name", "")
            tenure = str(row.get("c2021_tenure_9_name", "")).lower()
            val    = row.get("obs_value")
            if name in ENGLISH_REGION_NAMES:
                records.setdefault(name, {})
                if "owned" in tenure or "owner" in tenure:
                    records[name]["owner_occupier_pct"] = val
                elif "private rented" in tenure or "private rental" in tenure:
                    records[name]["private_rental_pct"] = val
    except Exception as e:
        log.warning(f"Census tenure fetch failed: {e}")

    if not records:
        log.warning("No Census 2021 data retrieved. Manual download may be needed.")
        _print_manual(
            "ONS Census 2021 Regional Data",
            "https://www.nomisweb.co.uk/census/2021",
            DEMO_RAW_DIR / "census_2021.csv",
            "Key tables: TS007 (Age), TS054 (Tenure) at English Region level"
        )
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(records, orient="index").reset_index()
    df.rename(columns={"index": "region_name"}, inplace=True)

    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    log.info(f"Census 2021: {len(df)} regions → {out_path}")
    return df

# ── Local Authority Population Density ─────────────────────────────────────────────────────


def compute_population_density(pop_df, area_km2_lookup):
    """
    Derives population_density = population / area_km2
    area_km2_lookup: dict mapping laua code to area in km2
    (from Census 2021 or ONS geography)
    """
    pop_df = pop_df.copy()
    pop_df['area_km2'] = pop_df['laua'].map(area_km2_lookup)
    pop_df['population_density'] = (
        pop_df['population'] / pop_df['area_km2']
    )
    return pop_df




def fetch_la_median_income(overwrite=False):
    """
    NOMIS Annual Survey of Hours and Earnings — median gross annual pay
    by local authority. Returns: [laua, year, median_household_income]
    """
    out_path = DEMO_PROC_DIR / 'la_median_income.parquet'
    if out_path.exists() and not overwrite:
        log.info('LA median income: already processed.')
        return pd.read_parquet(out_path)


    # NOMIS dataset NM_30_1 = ASHE Table 7 (residence-based)
    nomis_base = 'https://www.nomisweb.co.uk/api/v01/dataset'
    url = (
        f'{nomis_base}/NM_30_1.data.csv'
        '?geography=TYPE464'       # local authority districts
        '&variable=18'             # annual pay - gross - median
        '&pay=7'                   # annual
        '&sex=8'                   # total (male + female)
        '&select=DATE,GEOGRAPHY_CODE,OBS_VALUE'
    )


    log.info('Fetching LA median income from NOMIS...')
    try:
        r = _get(url, timeout=180)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = df.columns.str.lower()
        df.rename(columns={
            'date': 'year',
            'geography_code': 'laua',
            'obs_value': 'median_household_income'
        }, inplace=True)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['median_household_income'] = pd.to_numeric(
            df['median_household_income'], errors='coerce'
        )
        df.dropna(inplace=True)
    except Exception as e:
        log.warning(f'LA income fetch failed: {e}')
        return pd.DataFrame()


    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    log.info(f'LA median income: {len(df):,} rows')
    return df


# ── Run all ───────────────────────────────────────────────────────────────────

def run(overwrite: bool = False):
    log.info("=== Demographics: Starting ===")
    fetch_population_estimates(overwrite)
    fetch_migration(overwrite)
    fetch_census_2021(overwrite)
    fetch_la_median_income(overwrite)
    log.info("=== Demographics: Done ===")


if __name__ == "__main__":
    run()
