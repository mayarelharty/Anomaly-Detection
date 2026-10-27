"""
config.py – Central configuration for Phase 2 (Supervised Anomaly Detection).

Every tuneable choice is a single flag.  Import this module from any other
module to guarantee consistent settings across the pipeline.
"""

import os

# ────────────────────────── paths ──────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "cleaned_corrected_split.xlsx")
SHEET_NAME = "Summary"
HEADER_ROW = 3                          # 0-indexed row that contains column names

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "phase2_anomaly")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────── reproducibility ─────────────────
SEED = 42

# ────────────────────────── split ──────────────────────────
SPLIT_MODE = "chronological"            # "chronological" | "random"
TEST_SIZE  = 0.30

# ────────────────────────── labelling ──────────────────────
THRESHOLD_MODE    = "train_percentile"  # "benchmark" | "train_percentile"
# benchmark thresholds (A < 0.90  OR  P < 0.95  OR  Q < 0.999)
BENCH_A = 0.90
BENCH_P = 0.95
BENCH_Q = 0.999
# percentile thresholds (lower tail, fitted on train only)
TRAIN_PERCENTILE  = 10

# ────────────────────────── features ───────────────────────
FEATURE_MODE = "same_row"               # "same_row" | "history_only"

# ────────────────────────── gate ───────────────────────────
MIN_MINORITY_TRAIN = 50
MIN_MINORITY_TEST  = 10

# ────────────────────────── modelling ──────────────────────
PARAM_GRID = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth":    [None, 10, 20],
}
CV_FOLDS = 5
