"""
labels.py – Compute anomaly labels and label-distribution diagnostics.

Public API
----------
compute_labels(df, train_idx) -> pd.Series
label_distribution(df, y, train_idx, test_idx) -> str   (printable table)
gate_check(y, train_idx, test_idx) -> bool               (True = pass)
"""

import sys
import numpy as np
import pandas as pd

import config as cfg


def _benchmark_thresholds() -> dict:
    return {"A": cfg.BENCH_A, "P": cfg.BENCH_P, "Q": cfg.BENCH_Q}


def _percentile_thresholds(df: pd.DataFrame, train_idx) -> dict:
    """Compute per-dimension lower-tail thresholds on TRAIN only."""
    train = df.loc[train_idx]
    return {
        "A": train["Availability (A)"].quantile(cfg.TRAIN_PERCENTILE / 100),
        "P": train["Performance (P)"].quantile(cfg.TRAIN_PERCENTILE / 100),
        "Q": train["Quality (Q)"].quantile(cfg.TRAIN_PERCENTILE / 100),
    }


def compute_labels(df: pd.DataFrame, train_idx=None) -> tuple:
    """
    Assign y_anom = 1 if A < thr_A OR P < thr_P OR Q < thr_Q else 0.

    Returns (y: pd.Series, thresholds: dict).
    """
    if cfg.THRESHOLD_MODE == "benchmark":
        thr = _benchmark_thresholds()
    elif cfg.THRESHOLD_MODE == "train_percentile":
        assert train_idx is not None, "train_idx required for train_percentile"
        thr = _percentile_thresholds(df, train_idx)
    else:
        sys.exit(f"Unknown THRESHOLD_MODE: {cfg.THRESHOLD_MODE}")

    print(f"[labels] Thresholds ({cfg.THRESHOLD_MODE}): A < {thr['A']:.4f}, "
          f"P < {thr['P']:.4f}, Q < {thr['Q']:.4f}")

    y = (
        (df["Availability (A)"] < thr["A"]) |
        (df["Performance (P)"]  < thr["P"]) |
        (df["Quality (Q)"]      < thr["Q"])
    ).astype(int)

    return y, thr


def label_distribution(df: pd.DataFrame, y: pd.Series,
                       train_idx, test_idx, thresholds: dict) -> str:
    """Build a human-readable label distribution table."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"LABEL DISTRIBUTION  (threshold_mode = {cfg.THRESHOLD_MODE})")
    lines.append(f"Thresholds: A < {thresholds['A']:.4f}, P < {thresholds['P']:.4f}, Q < {thresholds['Q']:.4f}")
    lines.append(f"{'='*60}")

    for name, idx in [("FULL", df.index), ("TRAIN", train_idx), ("TEST", test_idx)]:
        sub_y = y.loc[idx]
        n = len(sub_y)
        n_anom = (sub_y == 1).sum()
        n_norm = (sub_y == 0).sum()
        lines.append(f"\n  {name}: {n} rows  |  anomaly = {n_anom} ({n_anom/n:.1%})  |  normal = {n_norm} ({n_norm/n:.1%})")

    # per-dimension rates (full dataset)
    a_flag = (df["Availability (A)"] < thresholds["A"]).astype(int)
    p_flag = (df["Performance (P)"]  < thresholds["P"]).astype(int)
    q_flag = (df["Quality (Q)"]      < thresholds["Q"]).astype(int)

    lines.append(f"\n  Per-dimension anomaly rates (full dataset):")
    lines.append(f"    A-anomaly: {a_flag.sum()}/{len(df)} = {a_flag.mean():.1%}")
    lines.append(f"    P-anomaly: {p_flag.sum()}/{len(df)} = {p_flag.mean():.1%}")
    lines.append(f"    Q-anomaly: {q_flag.sum()}/{len(df)} = {q_flag.mean():.1%}")

    # overlap patterns
    combo = a_flag * 4 + p_flag * 2 + q_flag  # 3-bit encoding
    labels_map = {
        0: "Normal (none)",
        1: "Q only",
        2: "P only",
        3: "P+Q",
        4: "A only",
        5: "A+Q",
        6: "A+P",
        7: "A+P+Q",
    }
    lines.append(f"\n  Overlap patterns:")
    for k in sorted(labels_map.keys()):
        cnt = (combo == k).sum()
        if cnt > 0:
            lines.append(f"    {labels_map[k]:15s}: {cnt:5d}  ({cnt/len(df):.1%})")

    return "\n".join(lines)


def gate_check(y: pd.Series, train_idx, test_idx) -> bool:
    """
    GATE: minority class in TRAIN >= MIN_MINORITY_TRAIN,
    and each class in TEST >= MIN_MINORITY_TEST.

    Returns True if gate passes, else prints diagnostic and returns False.
    """
    y_train = y.loc[train_idx]
    y_test  = y.loc[test_idx]

    train_counts = y_train.value_counts().to_dict()
    test_counts  = y_test.value_counts().to_dict()

    min_train = min(train_counts.values())
    min_test  = min(test_counts.get(0, 0), test_counts.get(1, 0))

    passed = True
    msgs = []

    if min_train < cfg.MIN_MINORITY_TRAIN:
        msgs.append(
            f"GATE FAIL: minority class in TRAIN has {min_train} rows "
            f"(need >= {cfg.MIN_MINORITY_TRAIN}).\n"
            f"  Train counts: {train_counts}"
        )
        passed = False

    if min_test < cfg.MIN_MINORITY_TEST:
        msgs.append(
            f"GATE FAIL: minority class in TEST has {min_test} rows "
            f"(need >= {cfg.MIN_MINORITY_TEST}).\n"
            f"  Test counts: {test_counts}"
        )
        passed = False

    if not passed:
        print("\n" + "!" * 60)
        print("LABEL GATE DID NOT PASS – modeling will not proceed.")
        print("!" * 60)
        for m in msgs:
            print(m)
        print("\nSuggested fixes:")
        if cfg.THRESHOLD_MODE == "benchmark":
            print("  • Set THRESHOLD_MODE = 'train_percentile' in config.py")
            print("    (benchmark thresholds label almost every row as anomalous)")
        if cfg.SPLIT_MODE == "chronological":
            print("  • Try SPLIT_MODE = 'random' for a stratified split")
        print("  • Lower MIN_MINORITY_TRAIN or MIN_MINORITY_TEST in config.py")
        print("  • Increase TEST_SIZE (currently {})".format(cfg.TEST_SIZE))
        return False

    print(f"[gate] PASSED ✓  Train minority = {min_train}, Test minority = {min_test}")
    return passed
