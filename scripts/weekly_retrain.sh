#!/usr/bin/env bash
# Weekly retrain pipeline.
#
# Install with cron (Mondays 03:00 local):
#   0 3 * * 1 /full/path/to/tennis-odds/scripts/weekly_retrain.sh >> /full/path/to/tennis-odds/logs/retrain.log 2>&1
#
# Steps:
#   1. Pull latest match data
#   2. Rebuild features
#   3. Retrain both models (test holdout always excluded)
#   4. Tune ensemble weight
#   5. Fit calibrator
#   6. Run sanity check; bail on failure (so we never deploy a broken model)

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/retrain_${TS}.log"

echo "[$(date)] Weekly retrain starting" | tee -a "$LOG"

uv run python main.py scrape           2>&1 | tee -a "$LOG"
uv run python main.py features         2>&1 | tee -a "$LOG"
uv run python main.py train            2>&1 | tee -a "$LOG"
uv run python main.py train-surface    2>&1 | tee -a "$LOG"
uv run python main.py tune-ensemble    2>&1 | tee -a "$LOG"
uv run python main.py tune-thresholds  2>&1 | tee -a "$LOG"
uv run python main.py calibrate --method isotonic 2>&1 | tee -a "$LOG"

echo "[$(date)] Running sanity check" | tee -a "$LOG"
if ! uv run python main.py sanity 2>&1 | tee -a "$LOG"; then
    echo "[$(date)] SANITY FAILED — model artifacts may be broken" | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] Weekly retrain complete" | tee -a "$LOG"
