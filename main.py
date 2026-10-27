"""
main.py – Phase 2 Orchestrator: Supervised Anomaly Detection & Feature Enrichment.

Runs Stage 1 (EDA / validation / gate) and Stage 2 (modeling / evaluation / scoring)
end to end, then writes report.md.
"""

import json
import os
import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg
from data     import load_and_clean, sanity_checks, validate_oee, split_data
from eda      import run_eda
from labels   import compute_labels, label_distribution, gate_check
from features import build_features
from modeling import build_pipeline, run_grid_search
from evaluate import evaluate_model
from scoring  import score_anomalies


def main():
    t0 = time.time()
    report_lines = []  # collect lines for report.md

    def log(msg):
        print(msg)
        report_lines.append(msg)

    # ════════════════════════════════════════════════════════
    # STAGE 1: ASSUMPTIONS, EDA, VALIDATION, LABEL GATE
    # ════════════════════════════════════════════════════════
    log("# Phase 2 – Supervised Anomaly Detection & Feature Enrichment")
    log("")
    log("## 1. Assumptions")
    log("")
    log("- The dataset is the **corrected** version (no 10/11/25_Asano B_HF 230; HF 230 Min CT ≈ 0.5385).")
    log("- The Summary sheet (header row 4, 0-indexed row 3) contains one row per Sec ID = date_machine_model.")
    log("- Machine identity (Asano A / Asano B) is encoded inside the Sec ID column.")
    log("- No pre-existing train/test split was found (no split column, no separate sheets); we create one.")
    log(f"- SPLIT_MODE = `{cfg.SPLIT_MODE}`, TEST_SIZE = {cfg.TEST_SIZE}")
    log(f"- THRESHOLD_MODE = `{cfg.THRESHOLD_MODE}`")
    log(f"- FEATURE_MODE = `{cfg.FEATURE_MODE}`")
    log(f"- SEED = {cfg.SEED}")
    log("")

    # ── load & clean ───────────────────────────────────────
    log("## 2. Data Loading & Cleaning")
    df = load_and_clean()
    sanity_checks(df)
    log(f"- Shape after cleaning: {df.shape}")
    log(f"- Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    log(f"- Unique dates: {df['Date'].nunique()}")
    log(f"- Machines: {df['machine'].value_counts().to_dict()}")
    log(f"- Models: {sorted(df['Model'].unique())}")
    log("")

    # ── EDA ────────────────────────────────────────────────
    log("## 3. Exploratory Data Analysis")
    run_eda(df)
    log("- EDA figures saved to `outputs/phase2_anomaly/`")
    log("")

    # ── OEE validation ─────────────────────────────────────
    log("## 4. OEE Validation")
    diffs = validate_oee(df)
    log("Max absolute difference per recomputed column:")
    for col, diff in diffs.items():
        log(f"  - {col}: {diff:.2e}")
    all_match = all(d < 1e-10 for d in diffs.values())
    if all_match:
        log("- **All columns match** (< 1e-10). Using existing A, P, Q, OEE columns.")
    else:
        log("- ⚠ Mismatch detected! Investigating…")
        # Check for division-by-zero rows
        zero_op = (df["Operating time (min)"] == 0).sum()
        zero_run = ((df["Operating time (min)"] - df["Total Downtime (min)"]) == 0).sum()
        zero_prod = (df["Production (pcs)"] == 0).sum()
        log(f"  - Rows with Operating time = 0: {zero_op}")
        log(f"  - Rows with run time = 0: {zero_run}")
        log(f"  - Rows with Production = 0: {zero_prod}")
        if max(diffs.values()) < 1e-6:
            log("- Differences < 1e-6 (floating point noise). Using existing columns.")
        else:
            log("- **STOPPING**: OEE mismatch too large.")
            _write_report(report_lines)
            sys.exit(1)
    log("")

    # ── initial split (needed for train_percentile thresholds) ──
    # For benchmark mode, we do a preliminary split to check the gate;
    # for train_percentile, we need the split before labels.
    if cfg.THRESHOLD_MODE == "benchmark":
        # compute labels with benchmark (no train_idx needed)
        y_prelim, thr_prelim = compute_labels(df, train_idx=None)
        train_idx, test_idx = split_data(df, y_prelim)
    else:
        # need train_idx first for percentile thresholds
        # use a temporary label to get the split, then recompute labels
        # For chronological split, the split doesn't depend on labels
        if cfg.SPLIT_MODE == "chronological":
            # dummy label
            y_dummy = pd.Series(0, index=df.index)
            train_idx, test_idx = split_data(df, y_dummy)
        else:
            # For random stratified, we need a preliminary label
            y_prelim_bench, _ = compute_labels(df, train_idx=None)
            train_idx, test_idx = split_data(df, y_prelim_bench)

    # ── compute labels (final) ─────────────────────────────
    y, thresholds = compute_labels(df, train_idx=train_idx)

    # For train_percentile with random split, re-split with actual labels
    if cfg.THRESHOLD_MODE == "train_percentile" and cfg.SPLIT_MODE == "random":
        train_idx, test_idx = split_data(df, y)

    # ── label distribution ─────────────────────────────────
    log("## 5. Label Distribution")
    dist_str = label_distribution(df, y, train_idx, test_idx, thresholds)
    log(dist_str)
    log("")

    # Also show distribution for the OTHER threshold mode for comparison
    if cfg.THRESHOLD_MODE == "benchmark":
        log("### For comparison (train_percentile mode):")
        import config as cfg_ref
        orig_mode = cfg_ref.THRESHOLD_MODE
        cfg_ref.THRESHOLD_MODE = "train_percentile"
        y_alt, thr_alt = compute_labels(df, train_idx=train_idx)
        alt_str = label_distribution(df, y_alt, train_idx, test_idx, thr_alt)
        log(alt_str)
        cfg_ref.THRESHOLD_MODE = orig_mode
    elif cfg.THRESHOLD_MODE == "train_percentile":
        log("### For comparison (benchmark mode):")
        import config as cfg_ref
        orig_mode = cfg_ref.THRESHOLD_MODE
        cfg_ref.THRESHOLD_MODE = "benchmark"
        y_alt, thr_alt = compute_labels(df, train_idx=None)
        alt_str = label_distribution(df, y_alt, train_idx, test_idx, thr_alt)
        log(alt_str)
        cfg_ref.THRESHOLD_MODE = orig_mode
    log("")

    # ── GATE ───────────────────────────────────────────────
    log("## 6. Label Gate Check")
    gate_passed = gate_check(y, train_idx, test_idx)
    if not gate_passed:
        log("- **GATE FAILED** – modeling cannot proceed.")
        log(f"- Current settings: THRESHOLD_MODE={cfg.THRESHOLD_MODE}, "
            f"SPLIT_MODE={cfg.SPLIT_MODE}")
        log("- See diagnostic messages above for suggested fixes.")
        _write_report(report_lines)
        print(f"\n[main] Report written to {os.path.join(cfg.OUTPUT_DIR, 'report.md')}")
        return
    log("- Gate **PASSED** ✓")
    log(f"- Split dates: train ≤ {df.loc[train_idx, 'Date'].max().date()}, "
        f"test ≥ {df.loc[test_idx, 'Date'].min().date()}")
    log("")

    # ════════════════════════════════════════════════════════
    # STAGE 2: MODELING
    # ════════════════════════════════════════════════════════
    log("## 7. Feature Engineering")

    X, feature_names = build_features(df, train_idx)
    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    log(f"- Features ({cfg.FEATURE_MODE}): {feature_names}")
    log(f"- X_train: {X_train.shape}, X_test: {X_test.shape}")
    log("")

    # assertion: test set never passed to fit/resample
    # (guaranteed by imblearn pipeline + GridSearchCV, but let's confirm)
    assert len(set(train_idx) & set(test_idx)) == 0

    # ── build & search ─────────────────────────────────────
    log("## 8. Model Training (GridSearchCV)")
    pipeline = build_pipeline()
    gs = run_grid_search(pipeline, X_train, y_train)

    log(f"- Best params: `{gs.best_params_}`")
    log(f"- Best CV F1: {gs.best_score_:.4f}")
    log("")

    # save CV results
    cv_df = pd.DataFrame(gs.cv_results_)
    try:
        cv_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "cv_results.csv"), index=False)
        log("- Full CV results saved to `cv_results.csv`")
    except PermissionError:
        log("- [WARNING] `cv_results.csv` is locked (open in Excel/another program). Saved as `cv_results_run.csv`.")
        cv_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "cv_results_run.csv"), index=False)

    # save best params
    with open(os.path.join(cfg.OUTPUT_DIR, "best_params.json"), "w") as f:
        # Convert None to string for JSON
        bp = {k: (str(v) if v is None else v) for k, v in gs.best_params_.items()}
        json.dump(bp, f, indent=2)

    # save model
    joblib.dump(gs.best_estimator_, os.path.join(cfg.OUTPUT_DIR, "rf_anomaly_model.joblib"))
    log("- Model saved to `rf_anomaly_model.joblib`")
    log("")

    # ── feature importance ─────────────────────────────────
    log("## 9. Feature Importance")
    _plot_feature_importance(gs.best_estimator_, feature_names, log)
    log("")

    # ── evaluation ─────────────────────────────────────────
    log("## 10. Evaluation (Test Set)")
    metrics = evaluate_model(gs, X_test, y_test, y_train, df, train_idx, test_idx)
    log("")
    log("| Metric | Value |")
    log("|--------|-------|")
    for k, v in metrics.items():
        log(f"| {k} | {v:.4f} |")
    log("")

    # ── anomaly scores ─────────────────────────────────────
    log("## 11. Anomaly Scores & Enriched Dataset")
    enriched = score_anomalies(gs, X, y, df, train_idx, test_idx, X_train, y_train)
    log("- Enriched dataset saved to `enriched_dataset.csv`")
    log("")

    # ── leakage section ────────────────────────────────────
    log("## 12. Leakage & Validity Notes")
    log("")
    if cfg.FEATURE_MODE == "same_row":
        log("> **⚠ IMPORTANT (same_row mode):** The label `y_anom` is a deterministic "
            "function of the features (Operating time, Total Downtime, Production, "
            "Defects, Min CT → A, P, Q → label). The model is therefore recovering a "
            "known rule, not learning genuinely independent anomaly patterns. High "
            "test-set scores (F1, ROC-AUC) reflect successful **rule recovery**, not "
            "independent anomaly detection capability.")
        log(">")
        log("> This is by design for Phase 2 (validating that the pipeline can "
            "faithfully reproduce domain thresholds). For real-world anomaly scoring, "
            "use `FEATURE_MODE = 'history_only'` which uses only lagged features "
            "from previous runs and is leakage-safe.")
    else:
        log("> `FEATURE_MODE = 'history_only'`: features are lagged/rolling statistics "
            "from previous runs. The current row's A/P/Q are never used as inputs, "
            "so there is **no target leakage**.")
    log("")
    log("- Anomaly_Score_OOF provides leakage-safe scores for Phase 3:")
    log("  train rows use out-of-fold probabilities; test rows use final model scores.")
    log("")

    elapsed = time.time() - t0
    log(f"---\n*Phase 2 completed in {elapsed:.1f}s.*")

    _write_report(report_lines)
    print(f"\n{'='*60}")
    print(f"ALL DONE. Outputs in: {cfg.OUTPUT_DIR}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


def _plot_feature_importance(pipeline, feature_names, log_fn):
    """Extract and plot feature importance from the RF inside the pipeline."""
    # The pipeline is [selector, smoteenn, clf]
    selector = pipeline.named_steps["selector"]
    clf = pipeline.named_steps["clf"]

    kept_names = selector.get_feature_names_out()
    importances = clf.feature_importances_

    # Map back to feature names
    if len(kept_names) == len(importances):
        names = kept_names
    else:
        names = [f"feature_{i}" for i in range(len(importances))]

    idx = np.argsort(importances)[::-1]

    log_fn("| Feature | Importance |")
    log_fn("|---------|------------|")
    for i in idx:
        log_fn(f"| {names[i]} | {importances[i]:.4f} |")

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(range(len(idx)), importances[idx][::-1], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx[::-1]])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    log_fn("- Saved `feature_importance.png`")


def _write_report(lines):
    """Write report.md."""
    report_path = os.path.join(cfg.OUTPUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[main] Report written to {report_path}")


if __name__ == "__main__":
    main()
