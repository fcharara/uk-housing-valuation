"""
Fast EPC Matching — Vectorised Exact Match
===========================================
Matches Land Registry transactions to EPC records using:
  1. Exact postcode match
  2. Exact house number extraction from address fields
  3. EPC lodgement date within 2 years of transaction date
  4. Closest date match when multiple EPCs match

Runs in ~8 minutes on the full 8M dataset (vs weeks for fuzzy matching).
Match rate: ~53% of 2015+ transactions.

Usage:
    python src/processing/epc_matching_fast.py
    python src/processing/epc_matching_fast.py --sample 0.05
"""
import argparse, logging, time, shutil, re
import pandas as pd, numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("epc_matching.log")])
log = logging.getLogger(__name__)

TRANSACTIONS_PATH = Path("data/merged/transactions_enriched.parquet")
EPC_PATH = Path("data/processed/epc/epc_cleaned.parquet")
OUTPUT_PATH = Path("data/merged/transactions_enriched.parquet")
BACKUP_PATH = Path("data/merged/transactions_enriched_no_epc.parquet")
ANALYSIS_START_YEAR = 2015
DATE_WINDOW_DAYS = 730
EPC_MERGE_COLS = ["floor_area","num_rooms","energy_rating","energy_efficiency_score","construction_age_band"]

def run_matching(sample_frac=None):
    start_time = time.time()
    if not BACKUP_PATH.exists() and TRANSACTIONS_PATH.exists():
        log.info(f"Backing up to {BACKUP_PATH}")
        shutil.copy2(TRANSACTIONS_PATH, BACKUP_PATH)
    source_path = BACKUP_PATH if BACKUP_PATH.exists() else TRANSACTIONS_PATH

    log.info("Loading transactions...")
    tx = pd.read_parquet(source_path)
    log.info(f"  Total rows: {len(tx):,}")
    tx_model = tx[tx["year"] >= ANALYSIS_START_YEAR].copy()
    log.info(f"  Transactions (2015+): {len(tx_model):,}")
    if sample_frac:
        tx_model = tx_model.sample(frac=sample_frac, random_state=42)
        log.info(f"  Sampled {sample_frac*100:.1f}%%: {len(tx_model):,}")

    if "postcode_nospace" not in tx_model.columns:
        tx_model["postcode_nospace"] = tx_model["postcode"].str.upper().str.replace(" ","",regex=False)
    for col in ["saon","paon","street","locality"]:
        if col not in tx_model.columns:
            tx_model[col] = ""

    log.info("  Extracting house numbers from LR addresses...")
    tx_model["paon_str"] = tx_model["paon"].fillna("").astype(str).str.upper().str.strip()
    tx_model["house_num"] = tx_model["paon_str"].str.extract(r'^(\d+)', expand=False)
    tx_has_num = tx_model["house_num"].notna()
    log.info(f"  With house number: {tx_has_num.sum():,} ({tx_has_num.mean()*100:.1f}%%)")

    log.info("Loading EPC data...")
    epc = pd.read_parquet(EPC_PATH)
    epc_cols = [c for c in ["postcode_nospace","epc_address","lodgement_date"]+EPC_MERGE_COLS if c in epc.columns]
    epc = epc[epc_cols].copy()
    log.info(f"  EPC records: {len(epc):,}")

    log.info("  Extracting house numbers from EPC addresses...")
    epc["house_num"] = epc["epc_address"].fillna("").str.extract(r'(\d+)', expand=False)
    epc_has_num = epc["house_num"].notna()
    log.info(f"  With house number: {epc_has_num.sum():,} ({epc_has_num.mean()*100:.1f}%%)")

    log.info("Merging on postcode + house number...")
    tx_merge = tx_model[tx_has_num][["postcode_nospace","house_num","date_of_transfer"]].copy()
    tx_merge["_tx_idx"] = tx_model[tx_has_num].index
    epc_cols_merge = ["postcode_nospace","house_num","lodgement_date"]+[c for c in EPC_MERGE_COLS if c in epc.columns]
    epc_merge = epc[epc_has_num][epc_cols_merge].copy()
    merged = tx_merge.merge(epc_merge, on=["postcode_nospace","house_num"], how="inner")
    log.info(f"  Raw joined rows: {len(merged):,}")

    log.info("Applying date window filter...")
    merged["date_diff"] = (merged["lodgement_date"] - merged["date_of_transfer"]).dt.days.abs()
    merged = merged[merged["date_diff"] <= DATE_WINDOW_DAYS]
    log.info(f"  After date filter: {len(merged):,}")

    log.info("Selecting closest match per transaction...")
    merged = merged.sort_values("date_diff")
    best = merged.drop_duplicates(subset=["_tx_idx"], keep="first")
    log.info(f"  Unique matched: {len(best):,} ({len(best)/len(tx_model)*100:.1f}%%)")

    log.info("Writing results...")
    best = best.set_index("_tx_idx")
    for col in EPC_MERGE_COLS:
        if col in tx.columns: tx = tx.drop(columns=[col])
    for col in EPC_MERGE_COLS:
        tx[col] = best[col] if col in best.columns else pd.NA
    tx.to_parquet(OUTPUT_PATH, compression="snappy", index=False)

    elapsed = (time.time() - start_time) / 60
    log.info(f"\nDone in {elapsed:.1f} minutes! Rows: {len(tx):,}, Cols: {len(tx.columns)}")
    for col in EPC_MERGE_COLS:
        if col in tx.columns:
            n = tx[col].notna().sum()
            log.info(f"  {col}: {n:,} ({n/len(tx)*100:.1f}%%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=None)
    args = parser.parse_args()
    run_matching(sample_frac=args.sample)
