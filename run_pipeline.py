#!/usr/bin/env python3
# run_pipeline.py
"""
Master pipeline runner for the UK Housing Valuation Model.

Usage:
  python run_pipeline.py                   # Run full pipeline
  python run_pipeline.py --steps ingest    # Only run data ingestion
  python run_pipeline.py --steps merge     # Only run merge step
  python run_pipeline.py --sample 0.05    # Use 5% sample (for dev/testing)
  python run_pipeline.py --overwrite       # Force re-download and reprocess

Pipeline stages:
  1. ingest   — Download raw data from all sources
  2. merge    — Join all datasets into one enriched parquet
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ]
)
log = logging.getLogger(__name__)


def run_ingestion(overwrite: bool = False):
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("STAGE 1: DATA INGESTION")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    from ingestion.land_registry  import run as run_lr
    from ingestion.postcode_lookup import run as run_pcd
    from ingestion.macro_indicators import run as run_macro
    from ingestion.demographics    import run as run_demo
    from ingestion.epc_data import run as run_epc

    log.info("── Step 1/5: Land Registry Price Paid ──")
    run_lr(overwrite=overwrite)

    log.info("── Step 2/5: Postcode → Region Lookup ──")
    run_pcd(overwrite=overwrite)

    log.info("── Step 3/5: Macroeconomic Indicators ──")
    run_macro(overwrite=overwrite)

    log.info("── Step 4/5: Demographic Data ──")
    run_demo(overwrite=overwrite)

    log.info("── Step 5/5: EPC Data ──")
    run_epc(overwrite=overwrite)

    log.info("STAGE 1 COMPLETE")


def run_merge(sample_frac: float = None, overwrite: bool = False):
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("STAGE 2: MERGE & FEATURE ENGINEERING")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    from processing.merge_pipeline import run as run_merge_pipeline
    run_merge_pipeline(sample_frac=sample_frac, overwrite=overwrite)

    log.info("STAGE 2 COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="UK Housing Model Pipeline Runner")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["ingest", "merge", "all"],
        default=["all"],
        help="Which pipeline stages to run (default: all)"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Fraction of Land Registry data to use (e.g. 0.05 for 5%%). "
             "Useful for development. Default: None (full data)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-download and reprocessing of all files"
    )
    args = parser.parse_args()

    steps = args.steps
    if "all" in steps:
        steps = ["ingest", "merge"]

    log.info("╔══════════════════════════════════════╗")
    log.info("║  UK Housing Valuation Model Pipeline  ║")
    log.info("╚══════════════════════════════════════╝")
    log.info(f"Steps    : {steps}")
    log.info(f"Sample   : {args.sample or 'Full dataset'}")
    log.info(f"Overwrite: {args.overwrite}")

    if "ingest" in steps:
        run_ingestion(overwrite=args.overwrite)

    if "merge" in steps:
        run_merge(sample_frac=args.sample, overwrite=args.overwrite)

    log.info("╔══════════════════════════════════════╗")
    log.info("║  Pipeline Complete                    ║")
    log.info("╚══════════════════════════════════════╝")
    log.info("Next step: open notebooks/01_eda.ipynb for exploratory analysis")


if __name__ == "__main__":
    main()
