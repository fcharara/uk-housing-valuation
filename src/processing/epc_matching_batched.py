"""
Batched EPC Matching — Memory-Efficient Version
"""
import argparse
import logging
import time
import shutil
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("epc_matching.log")])
log = logging.getLogger(__name__)

TRANSACTIONS_PATH = Path("data/merged/transactions_enriched.parquet")
EPC_PATH = Path("data/processed/epc/epc_cleaned.parquet")
OUTPUT_PATH = Path("data/merged/transactions_enriched.parquet")
BACKUP_PATH = Path("data/merged/transactions_enriched_no_epc.parquet")
SIMILARITY_THRESHOLD = 80
DATE_WINDOW_DAYS = 730
ANALYSIS_START_YEAR = 2015
EPC_MERGE_COLS = ["floor_area", "num_rooms", "energy_rating", "energy_efficiency_score", "construction_age_band"]

def build_lr_address(row):
    parts = []
    for col in ["saon", "paon", "street", "locality"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts).upper()

def match_one_postcode(tx_group, epc_group):
    results = []
    for tx_idx, tx in tx_group.iterrows():
        tx_addr = tx["lr_address"]
        tx_date = tx["date_of_transfer"]
        if not tx_addr or pd.isna(tx_date):
            continue
        date_mask = ((epc_group["lodgement_date"] >= tx_date - pd.Timedelta(days=DATE_WINDOW_DAYS))
            & (epc_group["lodgement_date"] <= tx_date + pd.Timedelta(days=DATE_WINDOW_DAYS)))
        candidates = epc_group[date_mask]
        if candidates.empty:
            continue
        best_score = 0
        best_match = None
        best_date_diff = 99999
        for _, epc_row in candidates.iterrows():
            score = fuzz.token_sort_ratio(tx_addr, epc_row["epc_address"])
            if score < SIMILARITY_THRESHOLD:
                continue
            date_diff = abs((epc_row["lodgement_date"] - tx_date).days)
            if score > best_score or (score == best_score and date_diff < best_date_diff):
                best_score = score
                best_date_diff = date_diff
                best_match = epc_row
        if best_match is not None:
            match_data = {"_tx_idx": tx_idx}
            for col in EPC_MERGE_COLS:
                if col in best_match.index:
                    match_data[col] = best_match[col]
            results.append(match_data)
    return results

def run_matching(sample_frac=None):
    if not BACKUP_PATH.exists():
        log.info(f"Backing up original dataset to {BACKUP_PATH}")
        shutil.copy2(TRANSACTIONS_PATH, BACKUP_PATH)
    log.info("Loading transactions...")
    tx = pd.read_parquet(TRANSACTIONS_PATH)
    tx = tx[tx["year"] >= ANALYSIS_START_YEAR]
    log.info(f"  Transactions (2015+): {len(tx):,}")
    if sample_frac:
        tx = tx.sample(frac=sample_frac, random_state=42)
        log.info(f"  Sampled {sample_frac*100:.1f}%: {len(tx):,}")
    log.info("  Building address strings...")
    tx["lr_address"] = tx.apply(build_lr_address, axis=1)
    if "postcode_nospace" not in tx.columns:
        tx["postcode_nospace"] = tx["postcode"].str.upper().str.replace(" ", "", regex=False)
    log.info("Loading EPC data...")
    epc_cols_needed = ["postcode_nospace", "epc_address", "lodgement_date"] + EPC_MERGE_COLS
    epc = pd.read_parquet(EPC_PATH)
    epc_cols_available = [c for c in epc_cols_needed if c in epc.columns]
    epc = epc[epc_cols_available]
    log.info(f"  EPC records: {len(epc):,}")
    tx_postcodes = set(tx["postcode_nospace"].dropna().unique())
    epc_postcodes = set(epc["postcode_nospace"].dropna().unique())
    shared_postcodes = sorted(tx_postcodes & epc_postcodes)
    log.info(f"  Shared postcodes: {len(shared_postcodes):,}")
    log.info("Grouping by postcode...")
    tx_grouped = tx.groupby("postcode_nospace")
    epc_grouped = epc.groupby("postcode_nospace")
    del epc
    gc.collect()
    all_matches = []
    total_postcodes = len(shared_postcodes)
    matched_count = 0
    total_tx_checked = 0
    start_time = time.time()
    log.info(f"Starting matching across {total_postcodes:,} postcodes...")
    for i, pc in enumerate(shared_postcodes):
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (total_postcodes - i) / rate / 60
            match_rate = matched_count / max(total_tx_checked, 1) * 100
            log.info(f"  Progress: {i:,}/{total_postcodes:,} ({i/total_postcodes*100:.1f}%) | Matched: {matched_count:,} ({match_rate:.1f}%) | ETA: {eta:.0f} min")
        try:
            tx_group = tx_grouped.get_group(pc)
            epc_group = epc_grouped.get_group(pc)
        except KeyError:
            continue
        total_tx_checked += len(tx_group)
        matches = match_one_postcode(tx_group, epc_group)
        all_matches.extend(matches)
        matched_count += len(matches)
    elapsed_total = (time.time() - start_time) / 60
    log.info(f"Matching complete in {elapsed_total:.1f} minutes")
    log.info(f"  Total checked: {total_tx_checked:,}")
    log.info(f"  Total matched: {matched_count:,} ({matched_count/max(total_tx_checked,1)*100:.1f}%)")
    if not all_matches:
        log.warning("No matches found!")
        return
    log.info("Merging EPC columns into transactions...")
    match_df = pd.DataFrame(all_matches).set_index("_tx_idx")
    full_tx = pd.read_parquet(BACKUP_PATH)
    for col in EPC_MERGE_COLS:
        if col in match_df.columns:
            full_tx[col] = match_df[col]
        elif col not in full_tx.columns:
            full_tx[col] = pd.NA
    full_tx.to_parquet(OUTPUT_PATH, compression="snappy", index=False)
    log.info(f"Saved: {len(full_tx):,} rows, {len(full_tx.columns)} columns")
    for col in EPC_MERGE_COLS:
        if col in full_tx.columns:
            coverage = full_tx[col].notna().sum()
            log.info(f"  {col}: {coverage:,} non-null ({coverage/len(full_tx)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=None)
    args = parser.parse_args()
    run_matching(sample_frac=args.sample)
