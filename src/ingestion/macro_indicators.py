# ingestion/macro_indicators.py
"""
Downloads and processes macroeconomic indicators for the housing model.

Indicators collected:
  1. Bank of England Base Rate        (monthly, 1975–present)
  2. ONS Regional GVA per head        (annual, by English region, 1998–present)
  3. ONS Regional Unemployment Rate   (annual, by English region, 2004–present)
  4. ONS CPI Inflation                (monthly, 1989–present)
  5. ONS Average Weekly Earnings      (monthly, UK-wide, 2000–present)

All series are saved as parquet and later joined to the merged dataset
via year/region keys.
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

MACRO_RAW_DIR  = RAW_DIR  / "macro"
MACRO_PROC_DIR = PROC_DIR / "macro"
MACRO_RAW_DIR.mkdir(parents=True, exist_ok=True)
MACRO_PROC_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 60, **kwargs) -> requests.Response:
    r = requests.get(url, timeout=timeout, **kwargs)
    r.raise_for_status()
    return r


# ── 1. Bank of England Base Rate ──────────────────────────────────────────────

def fetch_boe_base_rate(overwrite: bool = False) -> pd.DataFrame:
    """
    Downloads BoE base rate from their public stats API.
    Returns monthly DataFrame with columns: [date, base_rate_pct]
    """
    out_path = MACRO_PROC_DIR / "boe_base_rate.parquet"
    if out_path.exists() and not overwrite:
        log.info("BoE base rate: already processed.")
        return pd.read_parquet(out_path)

    # BoE publishes IUMABEDR (official Bank Rate) as a public CSV
    url = (
        "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
        "?Travel=NIxIRxSUx&FromSeries=1&ToSeries=50&DAT=RNG"
        "&FD=1&FM=Jan&FY=1975&TD=31&TM=Dec&TY=2025"
        "&VFD=Y&html.x=66&html.y=26&C=IUMABEDR&Filter=N"
    )

    log.info("Fetching BoE base rate...")
    try:
        r = _get(url)
        # BoE returns HTML-wrapped CSV; parse the data portion
        lines = r.text.splitlines()
        data_start = next(
            (i for i, l in enumerate(lines) if l.strip().startswith('"Date"') or l.strip().startswith("Date")),
            None
        )
        if data_start is None:
            raise ValueError("Could not locate data header in BoE response")

        csv_text = "\n".join(lines[data_start:])
        df = pd.read_csv(io.StringIO(csv_text))
        df.columns = ["date", "base_rate_pct"]
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["base_rate_pct"] = pd.to_numeric(df["base_rate_pct"], errors="coerce")
        df.dropna(inplace=True)
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df.sort_values("date", inplace=True)

    except Exception as e:
        log.warning(f"BoE live fetch failed ({e}). Using manual download fallback.")
        _print_manual("BoE Base Rate",
                      "https://www.bankofengland.co.uk/monetary-policy/the-interest-rate-bank-rate",
                      MACRO_RAW_DIR / "boe_base_rate.csv",
                      "Columns expected: Date, Value (rate as percent)")
        return pd.DataFrame()

    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    log.info(f"BoE base rate: {len(df):,} rows → {out_path}")
    return df


# ── 2. ONS Regional GVA per head ──────────────────────────────────────────────

def fetch_regional_gva(overwrite: bool = False) -> pd.DataFrame:
    """
    Downloads ONS Regional GVA (Balanced) per head by English ITL1/GOR region.
    Returns: [region_name, year, gva_per_head_gbp]
    """
    out_path = MACRO_PROC_DIR / "regional_gva.parquet"
    if out_path.exists() and not overwrite:
        log.info("Regional GVA: already processed.")
        return pd.read_parquet(out_path)

    url = (
        "https://www.ons.gov.uk/file?uri=/economy/grossdomesticproductgdp/datasets/"
        "regionalgrossvalueaddedbalancedbyindustry/current/"
        "regionalgvabalancedbyindustry.xlsx"
    )
    raw_path = MACRO_RAW_DIR / "regional_gva.xlsx"

    log.info("Fetching Regional GVA...")
    try:
        r = _get(url, timeout=120)
        with open(raw_path, "wb") as f:
            f.write(r.content)

        # Sheet "Table 3" contains per-head GVA; row structure varies by edition
        xls = pd.ExcelFile(raw_path)
        sheet = next((s for s in xls.sheet_names if "3" in s or "per head" in s.lower()), xls.sheet_names[0])
        raw = pd.read_excel(raw_path, sheet_name=sheet, header=None)

        # Find the row containing "England" and year headers
        year_row_idx = next(
            (i for i, row in raw.iterrows() if any(str(v).strip().isdigit() and int(str(v).strip()) > 1990
                                                    for v in row if pd.notna(v))), None
        )
        if year_row_idx is None:
            raise ValueError("Could not find year header row in GVA sheet")

        header = raw.iloc[year_row_idx]
        data   = raw.iloc[year_row_idx + 1:].copy()
        data.columns = header

        # First column is region name
        first_col = data.columns[0]
        data = data.rename(columns={first_col: "region_name"})

        # Filter to English regions only
        english_names = set(ENGLISH_REGIONS.values())
        data = data[data["region_name"].isin(english_names)]

        # Melt years wide → long
        year_cols = [c for c in data.columns if str(c).strip().isdigit() and 1990 < int(str(c).strip()) < 2030]
        df = data[["region_name"] + year_cols].melt(
            id_vars="region_name", var_name="year", value_name="gva_per_head_gbp"
        )
        df["year"]            = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["gva_per_head_gbp"] = pd.to_numeric(df["gva_per_head_gbp"], errors="coerce")
        df.dropna(inplace=True)

    except Exception as e:
        log.warning(f"Regional GVA fetch failed ({e}). Manual download required.")
        _print_manual("ONS Regional GVA",
                      "https://www.ons.gov.uk/economy/grossdomesticproductgdp/datasets/regionalgrossvalueaddedbalancedbyindustry",
                      raw_path,
                      "Table 3 — GVA per head by region; expects region rows, year columns")
        return pd.DataFrame()

    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    log.info(f"Regional GVA: {len(df):,} rows → {out_path}")
    return df


# ── 3. ONS CPI Inflation ──────────────────────────────────────────────────────

def fetch_cpi(overwrite: bool = False) -> pd.DataFrame:
    """
    Downloads UK CPI (all items, index 2015=100) monthly time series from ONS.
    Returns: [date, year, month, cpi_index, cpi_yoy_pct]
    """
    out_path = MACRO_PROC_DIR / "cpi.parquet"
    if out_path.exists() and not overwrite:
        log.info("CPI: already processed.")
        return pd.read_parquet(out_path)

    # ONS time series CZMT = CPI all items index (2015=100)
    url = "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/czmt/mm23"

    log.info("Fetching CPI...")
    try:
        r = _get(url)
        lines = r.text.splitlines()
        # Skip metadata lines; data rows start with a year-month pattern e.g. "1989 JAN"
        data_lines = [l for l in lines if len(l.split(",")) == 2 and l.split(",")[0].strip()[:4].isdigit()]
        df = pd.read_csv(io.StringIO("\n".join(["date,cpi_index"] + data_lines)))
        df["date"]      = pd.to_datetime(df["date"], format="%Y %b", errors="coerce")
        df["cpi_index"] = pd.to_numeric(df["cpi_index"], errors="coerce")
        df.dropna(inplace=True)
        df.sort_values("date", inplace=True)
        df["year"]        = df["date"].dt.year
        df["month"]       = df["date"].dt.month
        df["cpi_yoy_pct"] = df["cpi_index"].pct_change(12) * 100

    except Exception as e:
        log.warning(f"CPI fetch failed ({e}). Manual download required.")
        _print_manual("ONS CPI",
                      "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/czmt/mm23",
                      MACRO_RAW_DIR / "cpi.csv",
                      "Download CSV; expected columns: date (YYYY Mon), value")
        return pd.DataFrame()

    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    log.info(f"CPI: {len(df):,} rows → {out_path}")
    return df


# ── 4. ONS Average Weekly Earnings ────────────────────────────────────────────

def fetch_average_earnings(overwrite: bool = False) -> pd.DataFrame:
    """
    Downloads UK Average Weekly Earnings (total pay, not seasonally adjusted).
    Returns: [date, year, month, avg_weekly_earnings_gbp]
    """
    out_path = MACRO_PROC_DIR / "avg_earnings.parquet"
    if out_path.exists() and not overwrite:
        log.info("Avg earnings: already processed.")
        return pd.read_parquet(out_path)

    # KAB9 = Average Weekly Earnings, total pay, not seasonally adjusted
    url = "https://www.ons.gov.uk/generator?format=csv&uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kab9/emp"

    log.info("Fetching Average Weekly Earnings...")
    try:
        r = _get(url)
        lines = r.text.splitlines()
        data_lines = [l for l in lines if len(l.split(",")) == 2 and l.split(",")[0].strip()[:4].isdigit()]
        df = pd.read_csv(io.StringIO("\n".join(["date,avg_weekly_earnings_gbp"] + data_lines)))
        df["date"]                  = pd.to_datetime(df["date"], format="%Y %b", errors="coerce")
        df["avg_weekly_earnings_gbp"] = pd.to_numeric(df["avg_weekly_earnings_gbp"], errors="coerce")
        df.dropna(inplace=True)
        df.sort_values("date", inplace=True)
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month

    except Exception as e:
        log.warning(f"Avg earnings fetch failed ({e}). Manual download required.")
        _print_manual("ONS AWE",
                      "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kab9/emp",
                      MACRO_RAW_DIR / "avg_earnings.csv",
                      "Download CSV from ONS time series explorer")
        return pd.DataFrame()

    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    log.info(f"Avg earnings: {len(df):,} rows → {out_path}")
    return df


# ── 5. MHCLG Housing Supply (net additions) ───────────────────────────────────

def fetch_housing_supply(overwrite: bool = False) -> pd.DataFrame:
    """
    Downloads MHCLG Live Table 122 — Net additional dwellings by region.
    Returns: [region_name, year, net_additions]
    """
    out_path = MACRO_PROC_DIR / "housing_supply.parquet"
    if out_path.exists() and not overwrite:
        log.info("Housing supply: already processed.")
        return pd.read_parquet(out_path)

    url = "https://assets.publishing.service.gov.uk/media/65f0f5e6da5c3500138d76c0/Live_Table_122.xlsx"
    raw_path = MACRO_RAW_DIR / "mhclg_live_table_122.xlsx"

    log.info("Fetching MHCLG housing supply (Live Table 122)...")
    try:
        r = _get(url, timeout=60)
        with open(raw_path, "wb") as f:
            f.write(r.content)

        xls = pd.ExcelFile(raw_path)
        sheet = xls.sheet_names[0]
        raw = pd.read_excel(raw_path, sheet_name=sheet, header=None)

        # Locate region name column and year headers
        year_row = next(
            (i for i, row in raw.iterrows()
             if sum(1 for v in row if str(v).strip().isdigit() and int(str(v).strip()) > 1990) > 5),
            None
        )
        if year_row is None:
            raise ValueError("Could not find year header row")

        header = raw.iloc[year_row]
        data   = raw.iloc[year_row + 1:].copy()
        data.columns = header

        first_col = data.columns[0]
        data = data.rename(columns={first_col: "region_name"})

        english_names = set(ENGLISH_REGIONS.values())
        data = data[data["region_name"].isin(english_names)]

        year_cols = [c for c in data.columns if str(c).strip().isdigit() and 1990 < int(str(c).strip()) < 2030]
        df = data[["region_name"] + year_cols].melt(
            id_vars="region_name", var_name="year", value_name="net_additions"
        )
        df["year"]          = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["net_additions"] = pd.to_numeric(df["net_additions"], errors="coerce")
        df.dropna(inplace=True)

    except Exception as e:
        log.warning(f"Housing supply fetch failed ({e}). Manual download required.")
        _print_manual(
            "MHCLG Live Table 122",
            "https://www.gov.uk/government/statistical-data-sets/live-tables-on-net-supply-of-housing",
            raw_path,
            "Download 'Live Table 122' Excel; region rows, financial year columns"
        )
        return pd.DataFrame()

    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    log.info(f"Housing supply: {len(df):,} rows → {out_path}")
    return df


# ── Helper ────────────────────────────────────────────────────────────────────

def _print_manual(name: str, url: str, dest: Path, notes: str):
    print(f"\n{'='*60}")
    print(f"MANUAL DOWNLOAD REQUIRED: {name}")
    print(f"{'='*60}")
    print(f"URL : {url}")
    print(f"Save: {dest}")
    print(f"Note: {notes}")
    print(f"{'='*60}\n")


# ── Mortgage Approvals ────────────────────────────────────────────────────────────────────

def fetch_mortgage_approvals(overwrite=False):
    """
    BoE mortgage approvals for house purchase (series AMZJ).
    Returns: [date, year, month, mortgage_approvals]
    """
    out_path = MACRO_PROC_DIR / 'mortgage_approvals.parquet'
    if out_path.exists() and not overwrite:
        log.info('Mortgage approvals: already processed.')
        return pd.read_parquet(out_path)


    url = (
        'https://www.bankofengland.co.uk/boeapps/database/'
        'fromshowcolumns.asp?Travel=NIxIRxSUx&FromSeries=1'
        '&ToSeries=50&DAT=RNG&FD=1&FM=Jan&FY=1993'
        '&TD=31&TM=Dec&TY=2025&VFD=Y&html.x=66&html.y=26'
        '&C=AMZJ&Filter=N'
    )


    log.info('Fetching mortgage approvals...')
    try:
        r = _get(url)
        lines = r.text.splitlines()
        data_start = next(
            (i for i, l in enumerate(lines)
             if l.strip().startswith('"Date"') or l.strip().startswith('Date')),
            None
        )
        if data_start is None:
            raise ValueError('Could not locate data header')


        csv_text = '\n'.join(lines[data_start:])
        df = pd.read_csv(io.StringIO(csv_text))
        df.columns = ['date', 'mortgage_approvals']
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df['mortgage_approvals'] = pd.to_numeric(
            df['mortgage_approvals'], errors='coerce'
        )
        df.dropna(inplace=True)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df.sort_values('date', inplace=True)


    except Exception as e:
        log.warning(f'Mortgage approvals fetch failed: {e}')
        return pd.DataFrame()


    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    log.info(f'Mortgage approvals: {len(df):,} rows')
    return df


# ── Run all ───────────────────────────────────────────────────────────────────

def run(overwrite: bool = False):
    log.info("=== Macro Indicators: Starting ===")
    fetch_boe_base_rate(overwrite)
    fetch_regional_gva(overwrite)
    fetch_cpi(overwrite)
    fetch_average_earnings(overwrite)
    fetch_housing_supply(overwrite)
    fetch_mortgage_approvals(overwrite)
    log.info("=== Macro Indicators: Done ===")


if __name__ == "__main__":
    run()
