# ingestion/postcode_lookup.py
"""
Downloads the ONS Postcode Directory and extracts a lean postcode → region lookup.

This is the geographic backbone of the pipeline: it maps every Land Registry
postcode to an English region code and name, enabling demographic/economic
data (which is at region level) to be joined onto property transactions.

Source: ONS Open Geography Portal
Licence: Open Government Licence v3.0

Manual download instructions (if automated download fails):
  1. Go to: https://geoportal.statistics.gov.uk
  2. Search "ONS Postcode Directory"
  3. Download the latest edition (CSV, ~1 GB)
  4. Save to: data/raw/postcode/ONSPD_latest.csv
"""

import sys
import logging
import zipfile
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, PROC_DIR, ENGLISH_REGIONS, CHUNK_SIZE, PARQUET_COMPRESS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PCD_RAW_DIR  = RAW_DIR  / "postcode"
PCD_PROC_DIR = PROC_DIR / "postcode"
PCD_RAW_DIR.mkdir(parents=True, exist_ok=True)
PCD_PROC_DIR.mkdir(parents=True, exist_ok=True)

# Output path for the lean lookup table
LOOKUP_PATH = PCD_PROC_DIR / "postcode_region_lookup.parquet"

# ONS Postcode Directory — latest available bulk download
# Note: The direct URL changes each quarter. Update this URL to the latest release.
# Current URL points to the November 2024 edition.
ONSPD_URL = (
    "https://www.arcgis.com/sharing/rest/content/items/"
    "dc23a64fa2db4c2fb946a9b0f5291e3b/data"
)

# Key column names inside ONSPD (column names are stable across editions)
# pcds  = postcode (with space, e.g. "SW1A 1AA")
# rgn   = region code (9-character ONS code)
# laua  = local authority code
# lat   = latitude
# long  = longitude
# ctry  = country code (E=England, W=Wales, S=Scotland, N=NI)
ONSPD_COLS_NEEDED = ["pcds", "rgn", "oslaua", "lat", "long", "ctry"]

def download_onspd(overwrite: bool = False) -> Path | None:
    """
    Attempt to download the ONSPD zip. Returns the raw CSV path or None.
    If the automated download fails, prints manual instructions.
    """
    raw_csv = PCD_RAW_DIR / "ONSPD_latest.csv"

    if raw_csv.exists() and not overwrite:
        log.info("ONSPD already downloaded.")
        return raw_csv

    log.info(f"Downloading ONSPD from {ONSPD_URL} ...")
    log.info("(This is ~1 GB — may take several minutes)")

    try:
        r = requests.get(ONSPD_URL, stream=True, timeout=300)
        r.raise_for_status()

        content_type = r.headers.get("content-type", "")
        raw_bytes = r.content

        # Try to unzip if it's a zip file
        if "zip" in content_type or raw_bytes[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(BytesIO(raw_bytes)) as z:
                # Find the main data CSV (largest file in the zip)
                csv_files = [f for f in z.namelist() if f.endswith(".csv") and "Data" in f]
                if not csv_files:
                    csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_files:
                    raise ValueError("No CSV found in ONSPD zip")

                target = max(csv_files, key=lambda f: z.getinfo(f).file_size)
                log.info(f"Extracting {target}...")
                with z.open(target) as src, open(raw_csv, "wb") as dst:
                    dst.write(src.read())
        else:
            with open(raw_csv, "wb") as f:
                f.write(raw_bytes)

        log.info(f"ONSPD saved to {raw_csv}")
        return raw_csv

    except Exception as e:
        log.error(f"Automated download failed: {e}")
        _print_manual_instructions(raw_csv)
        return None


def _print_manual_instructions(dest: Path):
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED FOR ONSPD")
    print("="*60)
    print("1. Go to: https://geoportal.statistics.gov.uk")
    print("2. Search: 'ONS Postcode Directory'")
    print("3. Click the latest edition → Download (CSV)")
    print("4. Unzip and place the main Data CSV here:")
    print(f"   {dest}")
    print("="*60 + "\n")


def build_lookup(overwrite: bool = False) -> pd.DataFrame:
    """
    Read ONSPD and produce a lean postcode → region lookup.
    Filters to England only. Saves to parquet.
    """
    if LOOKUP_PATH.exists() and not overwrite:
        log.info("Postcode lookup already built, loading...")
        return pd.read_parquet(LOOKUP_PATH)

    raw_csv = PCD_RAW_DIR / "ONSPD_latest.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"ONSPD CSV not found at {raw_csv}.\n"
            "Run download_onspd() first, or follow the manual download instructions."
        )

    log.info("Building postcode → region lookup...")
    chunks = []

    for chunk in pd.read_csv(
        raw_csv,
        usecols=lambda c: c in ONSPD_COLS_NEEDED,
        dtype=str,
        chunksize=CHUNK_SIZE,
        encoding="latin-1",
    ):
        # Filter to England only
        if "ctry" in chunk.columns:
            chunk = chunk[chunk["ctry"] == "E92000001"]  # England country code

        # Only keep live postcodes that map to an English region
        chunk = chunk[chunk["rgn"].isin(ENGLISH_REGIONS.keys())]

        # Normalise postcode (strip spaces → "SW1A1AA" format for matching)
        chunk["postcode_raw"] = chunk["pcds"].str.upper().str.strip()
        chunk["postcode_nospace"] = chunk["postcode_raw"].str.replace(" ", "", regex=False)

        # Add region name
        chunk["region_code"] = chunk["rgn"]
        chunk["region_name"] = chunk["rgn"].map(ENGLISH_REGIONS)

        chunk = chunk[["postcode_raw", "postcode_nospace", "region_code", "region_name", "oslaua", "lat", "long"]]
        chunk = chunk.rename(columns={"oslaua": "laua"})
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["postcode_nospace"])

    log.info(f"Lookup built: {len(df):,} postcodes across {df['region_name'].nunique()} regions")

    df.to_parquet(LOOKUP_PATH, compression="snappy", index=False)
    log.info(f"Saved to {LOOKUP_PATH}")
    return df


def load_lookup() -> pd.DataFrame:
    """Load the pre-built lookup table."""
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError("Lookup not found. Run build_lookup() first.")
    return pd.read_parquet(LOOKUP_PATH)


def run(overwrite: bool = False):
    download_onspd(overwrite=overwrite)
    build_lookup(overwrite=overwrite)
    log.info("=== Postcode lookup: Done ===")


if __name__ == "__main__":
    run()
