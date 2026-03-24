# processing/merge_pipeline.py
"""
Master merge pipeline: joins Land Registry transactions with
postcode → region lookup, macroeconomic indicators, and demographic data.

Output: data/merged/transactions_enriched.parquet

Join strategy:
  - Postcode lookup   → on postcode (exact join)
  - Macro monthly     → on [year, month]  (base rate, CPI, AWE)
  - Macro annual      → on [region_name, year]  (GVA, unemployment, housing supply)
  - Demographics      → on [region_name, year]  (population, migration)
  - Census 2021       → on [region_name]  (cross-sectional structural controls)
"""

import sys
import logging
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROC_DIR, MERGED_DIR, START_YEAR, END_YEAR, PARQUET_COMPRESS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LR_PROC_DIR = PROC_DIR / "land_registry"
PCD_PATH    = PROC_DIR / "postcode" / "postcode_region_lookup.parquet"
MACRO_DIR   = PROC_DIR / "macro"
DEMO_DIR    = PROC_DIR / "demographics"
OUT_PATH    = MERGED_DIR / "transactions_enriched.parquet"


def load_land_registry(years: range = None, sample_frac: float = None) -> pd.DataFrame:
    if years is None:
        years = range(START_YEAR, END_YEAR + 1)

    paths = sorted(LR_PROC_DIR.glob("pp-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Land Registry parquets in {LR_PROC_DIR}")

    log.info(f"Loading {len(paths)} Land Registry parquets...")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42)
        log.info(f"  Sampled {sample_frac*100:.0f}%: {len(df):,} rows")

    log.info(f"  Loaded: {len(df):,} transactions")
    return df


def load_postcode_lookup() -> pd.DataFrame:
    if not PCD_PATH.exists():
        raise FileNotFoundError(f"Postcode lookup not found at {PCD_PATH}")
    return pd.read_parquet(PCD_PATH)


def attach_region(transactions: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Join region onto each transaction via postcode."""
    log.info("Attaching region via postcode lookup...")

    # Normalise postcode to no-space format for joining
    transactions["postcode_nospace"] = (
        transactions["postcode"].str.upper().str.replace(" ", "", regex=False)
    )
    lookup_slim = lookup[["postcode_nospace", "region_code", "region_name", "laua", "lat", "long"]]

    merged = transactions.merge(lookup_slim, on="postcode_nospace", how="left")

    missing = merged["region_name"].isna().sum()
    pct     = missing / len(merged) * 100
    log.info(f"  Region match: {len(merged) - missing:,} matched, {missing:,} ({pct:.1f}%) unmatched")

    # Drop unmatched (postcodes outside England or not in lookup)
    merged = merged[merged["region_name"].notna()]
    log.info(f"  After dropping unmatched: {len(merged):,} rows")
    return merged


def attach_monthly_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Attach BoE base rate, CPI, and average earnings on [year, month]."""
    log.info("Attaching monthly macro indicators...")

    for name, col_rename in [
        ("boe_base_rate.parquet",  None),
        ("cpi.parquet",            None),
        ("avg_earnings.parquet",   None),
    ]:
        path = MACRO_DIR / name
        if not path.exists():
            log.warning(f"  {name} not found — skipping")
            continue

        macro = pd.read_parquet(path)
        # Keep only the non-date scalar columns + year + month
        drop_cols = [c for c in macro.columns if c not in ["year", "month"] and
                     pd.api.types.is_datetime64_any_dtype(macro[c])]
        macro.drop(columns=drop_cols + ["date"] if "date" in macro.columns else drop_cols, inplace=True, errors="ignore")

        before = df.columns.tolist()
        df = df.merge(macro, on=["year", "month"], how="left")
        new_cols = [c for c in df.columns if c not in before]
        log.info(f"  {name}: joined columns {new_cols}")

    return df


def attach_annual_regional(df: pd.DataFrame) -> pd.DataFrame:
    """Attach GVA, housing supply, population, migration on [region_name, year]."""
    log.info("Attaching annual regional indicators...")

    sources = [
        (MACRO_DIR / "regional_gva.parquet",       "GVA"),
        (MACRO_DIR / "housing_supply.parquet",     "Housing supply"),
        (DEMO_DIR  / "population_estimates.parquet","Population"),
        (DEMO_DIR  / "migration.parquet",           "Migration"),
    ]

    for path, label in sources:
        if not path.exists():
            log.warning(f"  {label} ({path.name}) not found — skipping")
            continue

        data = pd.read_parquet(path)
        # Ensure year is plain int for joining
        data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
        df["year"]   = df["year"].astype("Int64")

        before = set(df.columns)
        df = df.merge(data, on=["region_name", "year"], how="left")
        new_cols = set(df.columns) - before
        log.info(f"  {label}: joined {new_cols}")

    return df


def attach_census_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Census 2021 cross-sectional controls on [region_name]."""
    path = DEMO_DIR / "census_2021_regional.parquet"
    if not path.exists():
        log.warning("  Census 2021 not found — skipping")
        return df

    log.info("Attaching Census 2021 regional controls...")
    census = pd.read_parquet(path)
    before = set(df.columns)
    df = df.merge(census, on="region_name", how="left")
    new_cols = set(df.columns) - before
    log.info(f"  Census 2021: joined {new_cols}")
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer derived features for the model.
    These are documented in detail in features/feature_engineering.py,
    but core ones are computed here on the merged dataset.
    """
    log.info("Computing derived features...")

    # Price-to-income proxy (using AWE as income proxy)
    if "avg_weekly_earnings_gbp" in df.columns:
        df["price_to_annual_income"] = df["price"] / (df["avg_weekly_earnings_gbp"] * 52)

    # Housing pressure index: population growth / net additions
    # High values → demand outpacing supply
    if "population_growth_pct" in df.columns and "net_additions" in df.columns and "population" in df.columns:
        df["housing_pressure_index"] = (
            (df["population_growth_pct"] / 100 * df["population"]) /
            df["net_additions"].replace(0, pd.NA)
        )

    # Log price (target variable for models)
    df["log_price"] = df["price"].apply(lambda x: x if pd.isna(x) else __import__("math").log(x))

    # Decade indicator
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")

    # Season (proxy for listing seasonality)
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    })

    log.info(f"  Final dataset shape: {df.shape}")
    return df


def run(sample_frac: float = None, overwrite: bool = False):
    if OUT_PATH.exists() and not overwrite:
        log.info(f"Merged dataset already exists at {OUT_PATH}. Use overwrite=True to rebuild.")
        return pd.read_parquet(OUT_PATH)

    # ── Load and merge ────────────────────────────────────────────────────────
    transactions = load_land_registry(sample_frac=sample_frac)
    lookup       = load_postcode_lookup()

    df = attach_region(transactions, lookup)
    df = attach_monthly_macro(df)
    df = attach_annual_regional(df)
    df = attach_census_controls(df)
    df = compute_derived_features(df)

    # ── Save ──────────────────────────────────────────────────────────────────
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, compression=PARQUET_COMPRESS, index=False)
    log.info(f"=== Merge complete: {len(df):,} rows, {len(df.columns)} columns → {OUT_PATH} ===")

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MERGED DATASET SUMMARY")
    print("="*60)
    print(f"Rows        : {len(df):,}")
    print(f"Columns     : {len(df.columns)}")
    print(f"Date range  : {df['date_of_transfer'].min().date()} → {df['date_of_transfer'].max().date()}")
    print(f"Regions     : {sorted(df['region_name'].unique())}")
    print(f"\nNulls per column:")
    null_pct = (df.isnull().sum() / len(df) * 100).round(1)
    for col, pct in null_pct[null_pct > 0].items():
        print(f"  {col:<40} {pct:.1f}%")
    print("="*60)

    return df


if __name__ == "__main__":
    run()
