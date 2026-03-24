# ingestion/land_registry.py
"""
Downloads and performs initial cleaning of HM Land Registry Price Paid Data.

Source: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com
Licence: Open Government Licence v3.0

Strategy:
  - Download year-by-year (1995–2024) rather than the full 5 GB file
  - Save each year as a compressed parquet for fast downstream loading
  - Log any download failures so they can be retried
"""

import sys
import logging
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, PROC_DIR, LR_YEARLY_URL, LR_COLUMNS, START_YEAR, END_YEAR, CHUNK_SIZE, PARQUET_COMPRESS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LR_RAW_DIR  = RAW_DIR  / "land_registry"
LR_PROC_DIR = PROC_DIR / "land_registry"
LR_RAW_DIR.mkdir(parents=True, exist_ok=True)
LR_PROC_DIR.mkdir(parents=True, exist_ok=True)


# ── Property type mappings ────────────────────────────────────────────────────
PROPERTY_TYPE_MAP = {
    "D": "Detached",
    "S": "Semi-Detached",
    "T": "Terraced",
    "F": "Flat/Maisonette",
    "O": "Other",
}

DURATION_MAP = {
    "F": "Freehold",
    "L": "Leasehold",
    "U": "Unknown",
}


def download_year(year: int, overwrite: bool = False) -> Path | None:
    """Download a single year's CSV from Land Registry. Returns path or None on failure."""
    dest = LR_RAW_DIR / f"pp-{year}.csv"

    if dest.exists() and not overwrite:
        log.info(f"  {year}: already downloaded, skipping.")
        return dest

    url = LR_YEARLY_URL(year)
    log.info(f"  {year}: downloading from {url}")

    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=str(year), leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        log.info(f"  {year}: saved to {dest}")
        return dest
    except Exception as e:
        log.error(f"  {year}: FAILED — {e}")
        if dest.exists():
            dest.unlink()   # remove partial download
        return None


def clean_year(year: int, overwrite: bool = False) -> Path | None:
    """Read raw CSV, clean, and save as parquet. Returns output path or None."""
    raw_path  = LR_RAW_DIR  / f"pp-{year}.csv"
    out_path  = LR_PROC_DIR / f"pp-{year}.parquet"

    if not raw_path.exists():
        log.warning(f"  {year}: raw file not found, skipping clean.")
        return None

    if out_path.exists() and not overwrite:
        log.info(f"  {year}: already processed, skipping.")
        return out_path

    log.info(f"  {year}: cleaning...")
    chunks = []

    try:
        for chunk in pd.read_csv(
            raw_path,
            names=LR_COLUMNS,
            header=None,
            dtype=str,          # read all as str first; cast after cleaning
            chunksize=CHUNK_SIZE,
            encoding="latin-1",
            on_bad_lines="skip",
        ):
            # ── Filter to standard residential transactions only ──────────────
            chunk = chunk[chunk["ppd_category"] == "A"]      # standard price paid only
            chunk = chunk[chunk["record_status"] == "A"]     # additions only (not changes/deletes)
            chunk = chunk[chunk["property_type"].isin(PROPERTY_TYPE_MAP.keys())]

            # ── Cast types ────────────────────────────────────────────────────
            chunk["price"]            = pd.to_numeric(chunk["price"], errors="coerce")
            chunk["date_of_transfer"] = pd.to_datetime(chunk["date_of_transfer"], errors="coerce")

            # ── Drop rows missing critical fields ─────────────────────────────
            chunk.dropna(subset=["price", "date_of_transfer", "postcode"], inplace=True)

            # ── Derived columns ───────────────────────────────────────────────
            chunk["year"]             = chunk["date_of_transfer"].dt.year
            chunk["month"]            = chunk["date_of_transfer"].dt.month
            chunk["quarter"]          = chunk["date_of_transfer"].dt.quarter
            chunk["is_new_build"]     = (chunk["old_new"] == "Y").astype("int8")
            chunk["property_type_label"] = chunk["property_type"].map(PROPERTY_TYPE_MAP)
            chunk["duration_label"]   = chunk["duration"].map(DURATION_MAP)

            # ── Normalise postcode (strip spaces, uppercase) ──────────────────
            chunk["postcode"] = chunk["postcode"].str.upper().str.strip()
            chunk["postcode_area"] = chunk["postcode"].str.extract(r"^([A-Z]{1,2})")

            # ── Drop columns not needed downstream ────────────────────────────
            chunk.drop(columns=["ppd_category", "record_status", "old_new"], inplace=True)

            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)

        # ── Sanity checks ─────────────────────────────────────────────────────
        initial_count = len(df)
        df = df[df["price"].between(10_000, 50_000_000)]   # remove clear outliers
        log.info(f"  {year}: {initial_count:,} → {len(df):,} rows after price filter")

        df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
        log.info(f"  {year}: saved to {out_path}")
        return out_path

    except Exception as e:
        log.error(f"  {year}: clean FAILED — {e}")
        return None


def run(years: range = None, overwrite: bool = False):
    """Download and clean all years. Call this as the main entry point."""
    if years is None:
        years = range(START_YEAR, END_YEAR + 1)

    failed_download = []
    failed_clean    = []

    log.info("=== Land Registry: Download phase ===")
    for year in years:
        result = download_year(year, overwrite=overwrite)
        if result is None:
            failed_download.append(year)

    log.info("=== Land Registry: Clean phase ===")
    for year in years:
        if year in failed_download:
            continue
        result = clean_year(year, overwrite=overwrite)
        if result is None:
            failed_clean.append(year)

    log.info("=== Land Registry: Done ===")
    if failed_download:
        log.warning(f"  Failed downloads: {failed_download}")
    if failed_clean:
        log.warning(f"  Failed cleans: {failed_clean}")
    log.info(f"  Processed parquets in: {LR_PROC_DIR}")


def load_processed(years: range = None) -> pd.DataFrame:
    """Load all processed parquet files into a single DataFrame."""
    if years is None:
        years = range(START_YEAR, END_YEAR + 1)

    paths = [LR_PROC_DIR / f"pp-{y}.parquet" for y in years
             if (LR_PROC_DIR / f"pp-{y}.parquet").exists()]

    if not paths:
        raise FileNotFoundError(f"No processed Land Registry parquets found in {LR_PROC_DIR}")

    log.info(f"Loading {len(paths)} parquet files...")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


if __name__ == "__main__":
    run()
