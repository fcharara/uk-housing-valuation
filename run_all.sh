#!/bin/bash
# ============================================================
# UK Housing Valuation Model — Full Reproducibility Script
# ============================================================
# Runs the complete pipeline from data processing to web app.
# Prerequisites:
#   - Python 3.11 with venv activated
#   - Raw data downloaded to data/raw/ (see README.md)
#   - EPC data from https://epc.opendatacommunities.org/
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
# ============================================================

set -e  # Exit on any error

echo "============================================"
echo "STAGE 1: Data Ingestion & Merge"
echo "============================================"
python run_pipeline.py

echo ""
echo "============================================"
echo "STAGE 2: EPC Matching (~8 minutes)"
echo "============================================"
python src/processing/epc_matching_fast.py

echo ""
echo "============================================"
echo "STAGE 3: Exploratory Data Analysis"
echo "============================================"
python src/analysis/01_eda.py

echo ""
echo "============================================"
echo "STAGE 4: Model Training (~3 hours)"
echo "============================================"
python src/analysis/02_model_training.py

echo ""
echo "============================================"
echo "STAGE 5: Evaluation & SHAP Analysis"
echo "============================================"
python src/analysis/03_evaluation.py

echo ""
echo "============================================"
echo "STAGE 6: Housing Pressure Index Maps"
echo "============================================"
python src/analysis/04_housing_pressure.py

echo ""
echo "============================================"
echo "ALL STAGES COMPLETE"
echo "============================================"
echo "Outputs saved to:"
echo "  outputs/eda/          — EDA plots and statistics"
echo "  outputs/models/       — Trained model artifacts"
echo "  outputs/evaluation/   — SHAP plots, comparisons"
echo "  outputs/hpi/          — Choropleth maps"
echo ""
echo "To launch the web app:"
echo "  python src/webapp/app.py"
