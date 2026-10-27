"""
config.py – Central configuration for Phase 2 (Anomaly-Risk Prediction & Feature Enrichment).

Every tuneable choice is a flag. Import this module across the pipeline
to guarantee consistency.
"""

import os

# ────────────────────────── paths ──────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "cleaned_corrected_split.xlsx")
SHEET_NAME = "Summary"
HEADER_ROW = 3                          # 0-indexed row containing column names

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "phase2_anomaly")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────── reproducibility ─────────────────
SEED = 42

# ────────────────────────── split ──────────────────────────
SPLIT_MODE = "chronological"            # "chronological" (70/30 on date boundaries) | "random"
TEST_SIZE  = 0.30

# ────────────────────────── labelling (IQR Rule) ───────────
IQR_K            = 1.5                  # 1.5 (default Tukey fence) | 2.0 | 3.0
IQR_SCOPE        = "machine_model"      # "machine_model" (per group) | "global"
MIN_GROUP_TRAIN  = 20                   # minimum training rows required to compute group fence

# ────────────────────────── gate criteria ──────────────────
MIN_MINORITY_TRAIN = 50                 # minimum training anomalies required to proceed
MIN_MINORITY_TEST  = 10                 # minimum test anomalies required to proceed
MIN_DIM_TRAIN_WARN = 10                 # warn if any single dimension has < 10 training positives

# ────────────────────────── features ───────────────────────
FEATURE_MODE     = "history_only"       # "history_only" (reported) | "same_row" (diagnostic ceiling)

# ────────────────────────── modeling variants ──────────────
PRIMARY_VARIANT  = "smoteenn"           # "smoteenn" (primary) | "balanced" | "reference"
PARAM_GRID = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth":    [None, 10, 20],
}
CV_FOLDS = 5

# ────────────────────────── evaluation ─────────────────────
BOOTSTRAP_ROUNDS = 1000
SEEDS_STABILITY  = [42, 7, 123, 999, 2024]
