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
THRESHOLD_METHOD = "iqr"                # "iqr" (Tukey box plot, default) | "percentile" (Alt A) |
                                        # "oee" (Alt B: single composite OEE fence) | "benchmark" (Alt C: static targets)
AUTO_FALLBACK_PERCENTILE = True         # Alt A: switch to percentile fences if IQR gives < MIN_MINORITY_TRAIN train anomalies
FALLBACK_PERCENTILE = 0.10              # Alt A lower percentile (P10)
BENCHMARK_TARGETS = {                   # Alt C industry targets (OEE target = product)
    "Availability (A)": 0.90,
    "Performance (P)":  0.95,
    "Quality (Q)":      0.999,
}

# ────────────────────────── gate criteria ──────────────────
MIN_MINORITY_TRAIN = 50                 # minimum training rows of the minority class required to proceed
MIN_MINORITY_TEST  = 10                 # minimum test rows of each class required to proceed
MIN_DIM_TRAIN_WARN = 10                 # warn if any single dimension has < 10 training positives

# ────────────────────────── features ───────────────────────
FEATURE_MODE     = "history_only"       # "history_only" (reported) | "same_row" (diagnostic ceiling)
FEATURE_SET      = "spec"               # "spec" (12 lag/rolling/code features) | "extended" (+15 fence-margin,
                                        # trend/volatility and recency features)

# ────────────────────────── modeling variants ──────────────
PRIMARY_VARIANT  = "smoteenn"           # "smoteenn" (primary) | "balanced" | "reference"
TUNING_MODE      = "spec"               # "spec" (3x3 grid, F1) | "extended" (leaf/max_features grid, PR-AUC)
PARAM_GRIDS = {
    "spec": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth":    [None, 10, 20],
    },
    "extended": {
        "clf__n_estimators":     [300],
        "clf__max_depth":        [None, 5, 10],
        "clf__min_samples_leaf": [1, 5, 10, 20],
        "clf__max_features":     ["sqrt", 0.5],
    },
}
GRID_SCORINGS = {"spec": "f1", "extended": "average_precision"}   # PR-AUC is more stable with few positives per fold
PARAM_GRID   = PARAM_GRIDS[TUNING_MODE]
GRID_SCORING = GRID_SCORINGS[TUNING_MODE]
CV_FOLDS = 5

# ────────────────────────── evaluation ─────────────────────
CUTOFF_MODE      = "oof_f1"             # "oof_f1" (spec: F1-optimal cutoff on calibrated OOF; F2 cutoff recorded) |
                                        # "fixed_raw" (DECISION_CUTOFF on raw scores)
DECISION_CUTOFF  = 0.50                 # fixed_raw cutoff; calibrated scores sit near prevalence and rarely reach it
SPEC_DEFAULTS = {"FEATURE_SET": "spec", "TUNING_MODE": "spec", "CUTOFF_MODE": "oof_f1",
                 "THRESHOLD_METHOD": "iqr"}
BOOTSTRAP_ROUNDS = 1000
SEEDS_STABILITY  = [42, 7, 123, 999, 2024]
