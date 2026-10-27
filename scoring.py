"""
scoring.py – Feature Enrichment & Leak-Free Anomaly Scoring.

Enriches original dataset with:
- Ground truth binary labels: y_anom, anom_A, anom_P, anom_Q
- Chronological split column: 'train', 'test', or 'dropped_no_history'
- Calibrated Anomaly Scores (OOF for training, final model for test)
- Raw probability scores for all three variants (smoteenn, balanced, reference)
- Explicitly removes optimistic in-sample scores.
"""

import os
import numpy as np
import pandas as pd

import config as cfg


def build_enriched_dataset(df_orig: pd.DataFrame, df_labels_full: pd.DataFrame,
                           valid_train_idx: list, valid_test_idx: list,
                           all_variants_data: dict, output_dir: str):
    """
    Assemble the final leak-free enriched dataset.
    """
    enriched = df_orig.copy()

    # Ground truth labels
    enriched["y_anom"] = df_labels_full["y_anom"].values
    enriched["anom_A"] = df_labels_full["anom_A"].values
    enriched["anom_P"] = df_labels_full["anom_P"].values
    enriched["anom_Q"] = df_labels_full["anom_Q"].values

    # Split assignment
    enriched["split"] = "dropped_no_history"
    enriched.loc[valid_train_idx, "split"] = "train"
    enriched.loc[valid_test_idx,  "split"] = "test"

    # Populate scores for each variant
    for vname, data in all_variants_data.items():
        score_col_cal = f"Anomaly_Score_{vname}" if vname != cfg.PRIMARY_VARIANT else "Anomaly_Score"
        score_col_raw = f"Anomaly_Score_{vname}_raw" if vname != cfg.PRIMARY_VARIANT else "Anomaly_Score_raw"

        # Initialize with NaN
        enriched[score_col_cal] = np.nan
        enriched[score_col_raw] = np.nan

        # Train rows: Walk-forward OOF (warm-up block stays NaN)
        enriched.loc[valid_train_idx, score_col_cal] = data["oof_calibrated"]
        enriched.loc[valid_train_idx, score_col_raw] = data["oof_raw"]

        # Test rows: Final model scores on untouched test set
        enriched.loc[valid_test_idx, score_col_cal] = data["test_cal_scores"]
        enriched.loc[valid_test_idx, score_col_raw] = data["test_raw_scores"]

    # Also keep standard named columns if primary variant is smoteenn
    if cfg.PRIMARY_VARIANT == "smoteenn":
        enriched["Anomaly_Score_smoteenn"] = enriched["Anomaly_Score"]
        enriched["Anomaly_Score_smoteenn_raw"] = enriched["Anomaly_Score_raw"]

    # Assertion: row order strictly equals original row order
    assert list(enriched.index) == list(df_orig.index), "Assertion Error: Enriched row order deviated from original!"

    # Save CSV
    out_path = os.path.join(output_dir, "enriched_dataset.csv")
    try:
        enriched.to_csv(out_path, index=False)
        print(f"[scoring] Saved enriched_dataset.csv ({enriched.shape}) to {out_path} ✓")
    except PermissionError:
        alt_path = os.path.join(output_dir, "enriched_dataset_run.csv")
        enriched.to_csv(alt_path, index=False)
        print(f"[scoring] [WARNING] File locked. Saved to {alt_path} ({enriched.shape})")

    # Diagnostic summary
    primary_data = all_variants_data[cfg.PRIMARY_VARIANT]
    print("\n" + "=" * 60)
    print("ANOMALY SCORE AUDIT (Leak-Free Out-Of-Fold & Test)")
    print("=" * 60)
    print(f"  Warm-up block NaNs (Fold 0 training): {primary_data['warm_up_nans']} rows")
    print(f"  Dropped first-run rows (No history):  {len(df_orig) - len(valid_train_idx) - len(valid_test_idx)} rows")
    print(f"  Valid evaluated Train OOF rows:       {len(valid_train_idx) - primary_data['warm_up_nans']} rows")
    print(f"  Valid evaluated Test rows:            {len(valid_test_idx)} rows")

    return enriched
