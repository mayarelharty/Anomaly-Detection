"""
data.py – Load, clean, validate, and split the OEE dataset.

Public API
----------
load_and_clean()  -> pd.DataFrame   (cleaned, sorted by Date)
validate_oee(df)  -> dict           (max absolute diffs per OEE column)
split_data(df, y) -> (train_idx, test_idx)
"""

import sys
import numpy as np
import pandas as pd

import config as cfg


# ────────────────────────── load & clean ───────────────────
def load_and_clean() -> pd.DataFrame:
    """Read the Summary sheet, coerce types, extract machine, sort by Date."""
    df = pd.read_excel(
        cfg.DATA_PATH,
        sheet_name=cfg.SHEET_NAME,
        header=cfg.HEADER_ROW,
    )

    # ── coerce types ────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = [
        "Operating time (min)", "Production (pcs)",
        "Planned Downtime (min)", "Unplanned Downtime (min)",
        "Total Downtime (min)", "Defects (pcs)",
        "Cycle time – Min CT (min/pc)",
        "Availability (A)", "Performance (P)", "Quality (Q)", "OEE",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── extract machine from Sec ID ─────────────────────────
    df["machine"] = df["Sec ID"].str.extract(r"(Asano [AB])")

    # ── sort by date (stable) then reset index ──────────────
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)

    # ── drop exact duplicates ───────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"[data] Dropped {n_dupes} exact duplicate rows.")

    return df


# ────────────────────────── sanity checks ──────────────────
def sanity_checks(df: pd.DataFrame) -> None:
    """Verify this is the corrected version of the dataset."""
    # 1. No row 10/11/25_Asano B_HF 230
    bad_row = df["Sec ID"].astype(str).str.contains("10/11/25_Asano B_HF 230")
    assert bad_row.sum() == 0, "FAIL: found row 10/11/25_Asano B_HF 230 – not the corrected file!"

    # 2. HF 230 Min CT ≈ 0.5385
    hf230_ct = df.loc[df["Model"] == "HF 230", "Cycle time – Min CT (min/pc)"].unique()
    assert len(hf230_ct) > 0, "FAIL: no HF 230 rows found."
    assert np.allclose(hf230_ct, 0.5384615384615384, atol=1e-4), (
        f"FAIL: HF 230 Min CT = {hf230_ct}, expected ≈ 0.5385"
    )
    print("[data] Sanity checks passed ✓")


# ────────────────────────── OEE validation ──────────────────
def validate_oee(df: pd.DataFrame) -> dict:
    """
    Recompute A, P, Q, OEE from raw columns and compare with stored values.
    Returns {column: max_abs_diff}.
    """
    op   = df["Operating time (min)"]
    td   = df["Total Downtime (min)"]
    prod = df["Production (pcs)"]
    defects = df["Defects (pcs)"]
    ct   = df["Cycle time – Min CT (min/pc)"]

    run_time = op - td  # net run time

    A_calc = run_time / op
    P_calc = (prod * ct) / run_time
    Q_calc = (prod - defects) / prod
    OEE_calc = A_calc * P_calc * Q_calc

    diffs = {
        "Availability (A)": (df["Availability (A)"] - A_calc).abs().max(),
        "Performance (P)":  (df["Performance (P)"]  - P_calc).abs().max(),
        "Quality (Q)":      (df["Quality (Q)"]      - Q_calc).abs().max(),
        "OEE":              (df["OEE"]              - OEE_calc).abs().max(),
    }
    return diffs


# ────────────────────────── split ──────────────────────────
def split_data(df: pd.DataFrame, y: pd.Series):
    """
    Split into train/test indices.

    Chronological: earliest 70 % of *unique dates* → train; rest → test.
    Random:        stratified on y with SEED.

    Returns (train_idx, test_idx) as numpy arrays of integer positions.
    """
    if cfg.SPLIT_MODE == "chronological":
        unique_dates = np.sort(df["Date"].unique())
        n_train_dates = int(len(unique_dates) * (1 - cfg.TEST_SIZE))
        cutoff_date = unique_dates[n_train_dates]        # first test date
        train_idx = df.index[df["Date"] < cutoff_date].values
        test_idx  = df.index[df["Date"] >= cutoff_date].values
        print(f"[split] Chronological: train dates ≤ {pd.Timestamp(unique_dates[n_train_dates-1]).date()}, "
              f"test dates ≥ {pd.Timestamp(cutoff_date).date()}")
    elif cfg.SPLIT_MODE == "random":
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            df.index.values, test_size=cfg.TEST_SIZE,
            stratify=y, random_state=cfg.SEED,
        )
    else:
        sys.exit(f"Unknown SPLIT_MODE: {cfg.SPLIT_MODE}")

    # assertion: no overlap
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap detected!"
    print(f"[split] Train: {len(train_idx)}, Test: {len(test_idx)}")
    return train_idx, test_idx
