"""
labels.py – Box-Plot (IQR) Ground Truth Labeling & Gate Verification.

Phase 2 is anomaly-risk prediction validated against threshold-based detection.
Ground truth labels are defined using one-sided lower-tail Tukey IQR fences
learned strictly from the training period.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import config as cfg

DIMS = ["Availability (A)", "Performance (P)", "Quality (Q)"]
DIM_SHORT = {"Availability (A)": "A", "Performance (P)": "P", "Quality (Q)": "Q"}


def compute_fences(train_df: pd.DataFrame, k: float = None, scope: str = None, min_group_train: int = None):
    """
    Compute lower fences for A, P, Q strictly from training rows.
    Returns:
        global_fences: dict of {dim: lower_fence}
        group_fences: dict of {(machine, model): {dim: lower_fence}}
        group_counts: dict of {(machine, model): count}
    """
    k = cfg.IQR_K if k is None else k
    scope = cfg.IQR_SCOPE if scope is None else scope
    min_group_train = cfg.MIN_GROUP_TRAIN if min_group_train is None else min_group_train

    global_fences = {}
    for d in DIMS:
        q1 = train_df[d].quantile(0.25)
        q3 = train_df[d].quantile(0.75)
        iqr = q3 - q1
        global_fences[d] = q1 - k * iqr

    group_fences = {}
    group_counts = {}
    if scope == "machine_model":
        for (m, mod), grp in train_df.groupby(["machine", "Model"]):
            n = len(grp)
            group_counts[(m, mod)] = n
            if n >= min_group_train:
                gf = {}
                for d in DIMS:
                    q1 = grp[d].quantile(0.25)
                    q3 = grp[d].quantile(0.75)
                    iqr = q3 - q1
                    gf[d] = q1 - k * iqr
                group_fences[(m, mod)] = gf

    return global_fences, group_fences, group_counts


def apply_fences(df: pd.DataFrame, global_fences: dict, group_fences: dict, scope: str = None) -> pd.DataFrame:
    """
    Apply pre-computed training fences to a dataset (train or test).
    Returns DataFrame with columns ['anom_A', 'anom_P', 'anom_Q', 'y_anom'].
    """
    scope = cfg.IQR_SCOPE if scope is None else scope
    anom_A = []
    anom_P = []
    anom_Q = []
    y_anom = []

    for _, row in df.iterrows():
        key = (row["machine"], row["Model"])
        if scope == "machine_model" and key in group_fences:
            f = group_fences[key]
        else:
            f = global_fences

        is_a = int(row["Availability (A)"] < f["Availability (A)"])
        is_p = int(row["Performance (P)"] < f["Performance (P)"])
        is_q = int(row["Quality (Q)"] < f["Quality (Q)"])

        anom_A.append(is_a)
        anom_P.append(is_p)
        anom_Q.append(is_q)
        y_anom.append(int(is_a or is_p or is_q))

    out = pd.DataFrame({
        "anom_A": anom_A,
        "anom_P": anom_P,
        "anom_Q": anom_Q,
        "y_anom": y_anom
    }, index=df.index)
    return out


def gate_check(train_labels: pd.DataFrame, test_labels: pd.DataFrame) -> bool:
    """
    Verify label gate criteria:
    - >= 50 training anomalies
    - >= 10 test anomalies
    - Warn if any single dimension has < 10 training positives
    """
    n_train_anom = train_labels["y_anom"].sum()
    n_test_anom = test_labels["y_anom"].sum()

    passed = True
    msgs = []

    if n_train_anom < cfg.MIN_MINORITY_TRAIN:
        msgs.append(f"GATE FAIL: Train anomalies = {n_train_anom} (requires >= {cfg.MIN_MINORITY_TRAIN})")
        passed = False
    if n_test_anom < cfg.MIN_MINORITY_TEST:
        msgs.append(f"GATE FAIL: Test anomalies = {n_test_anom} (requires >= {cfg.MIN_MINORITY_TEST})")
        passed = False

    # Check per-dimension warnings
    for d, short in [("anom_A", "A"), ("anom_P", "P"), ("anom_Q", "Q")]:
        cnt = train_labels[d].sum()
        if cnt < cfg.MIN_DIM_TRAIN_WARN:
            print(f"[gate] WARNING: Dimension {short} has only {cnt} training positives (< {cfg.MIN_DIM_TRAIN_WARN})")

    if not passed:
        print("\n" + "!" * 60)
        print("LABEL GATE HALTED EXECUTION:")
        for m in msgs:
            print(f"  • {m}")
        print("!" * 60)
        return False

    print(f"[gate] PASSED ✓ Train anomalies = {n_train_anom}, Test anomalies = {n_test_anom}")
    return True


def label_distribution_report(df: pd.DataFrame, train_labels: pd.DataFrame, test_labels: pd.DataFrame,
                              train_idx, test_idx, global_fences: dict, group_fences: dict):
    """
    Generate comprehensive label distribution summary, overlap patterns, and quarterly drift.
    """
    lines = []
    lines.append("## Label Distribution (Box-Plot IQR Rule)")
    lines.append(f"- Configuration: `IQR_K = {cfg.IQR_K}`, `IQR_SCOPE = '{cfg.IQR_SCOPE}'`, `MIN_GROUP_TRAIN = {cfg.MIN_GROUP_TRAIN}`")
    lines.append("")

    # Global fences
    lines.append("### Global Training Fences")
    lines.append("| Dimension | Lower Fence | Description |")
    lines.append("|---|---|---|")
    for d in DIMS:
        lines.append(f"| {d} | **{global_fences[d]:.4f}** | $Q_1 - {cfg.IQR_K} \\times \\text{{IQR}}$ on Train |")
    lines.append("")

    # Group fences
    lines.append("### Machine & Model Group Fences (Train)")
    lines.append(f"- Total groups in training: {len(group_fences)} met threshold $\\ge {cfg.MIN_GROUP_TRAIN}$ rows.")
    lines.append("| Machine | Model | Train Rows | Fence A | Fence P | Fence Q | Status |")
    lines.append("|---|---|---|---|---|---|---|")
    train_groups = df.loc[train_idx].groupby(["machine", "Model"])
    for (m, mod), grp in train_groups:
        n = len(grp)
        if (m, mod) in group_fences:
            gf = group_fences[(m, mod)]
            lines.append(f"| {m} | {mod} | {n} | {gf['Availability (A)']:.4f} | {gf['Performance (P)']:.4f} | {gf['Quality (Q)']:.4f} | Group-specific |")
        else:
            lines.append(f"| {m} | {mod} | {n} | {global_fences['Availability (A)']:.4f} | {global_fences['Performance (P)']:.4f} | {global_fences['Quality (Q)']:.4f} | Fallback (< {cfg.MIN_GROUP_TRAIN}) |")
    lines.append("")

    # Anomaly counts and rates
    lines.append("### Anomaly Counts & Rates")
    lines.append("| Partition | Rows | Total Anomalies ($y_{\\text{anom}}=1$) | Anomaly Rate | anom_A | anom_P | anom_Q |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, lbls in [("Train", train_labels), ("Test", test_labels)]:
        n = len(lbls)
        ya = lbls["y_anom"].sum()
        a = lbls["anom_A"].sum()
        p = lbls["anom_P"].sum()
        q = lbls["anom_Q"].sum()
        lines.append(f"| {name} | {n} | **{ya}** | **{ya/n:.2%}** | {a} ({a/n:.2%}) | {p} ({p/n:.2%}) | {q} ({q/n:.2%}) |")
    lines.append("")

    # Overlap between dimensions
    lines.append("### Dimension Overlap Patterns (Combined Dataset)")
    full_labels = pd.concat([train_labels, test_labels]).sort_index()
    overlap_code = full_labels["anom_A"] * 4 + full_labels["anom_P"] * 2 + full_labels["anom_Q"]
    overlap_names = {
        0: "Normal (No Dimension Triggered)",
        1: "Q only",
        2: "P only",
        3: "P + Q",
        4: "A only",
        5: "A + Q",
        6: "A + P",
        7: "A + P + Q (All Three)"
    }
    lines.append("| Pattern | Code | Count | Share |")
    lines.append("|---|---|---|---|")
    for k in sorted(overlap_names.keys()):
        cnt = (overlap_code == k).sum()
        lines.append(f"| {overlap_names[k]} | {k} | {cnt} | {cnt/len(full_labels):.2%} |")
    lines.append("")

    # Quarterly drift in test period
    lines.append("### Test Period Anomaly Rate per Quarter (Drift Analysis)")
    test_df = df.loc[test_idx].copy()
    test_df["y_anom"] = test_labels["y_anom"]
    test_df["Quarter"] = test_df["Date"].dt.to_period("Q").astype(str)
    lines.append("| Quarter | Total Shifts | Anomalies | Anomaly Rate | Notes |")
    lines.append("|---|---|---|---|---|")
    for qtr, grp in test_df.groupby("Quarter"):
        anom_cnt = grp["y_anom"].sum()
        tot_cnt = len(grp)
        lines.append(f"| {qtr} | {tot_cnt} | {anom_cnt} | **{anom_cnt/tot_cnt:.2%}** | Test Period |")
    lines.append("")

    return "\n".join(lines)


def run_sensitivity_table(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Compute sensitivity table: IQR_K in {1.5, 2.0, 3.0} x scope in {global, machine_model}.
    Reports class counts and gate status.
    """
    rows = []
    for scope in ["global", "machine_model"]:
        for k in [1.5, 2.0, 3.0]:
            gf, grp_f, _ = compute_fences(train_df, k=k, scope=scope)
            tr_lbl = apply_fences(train_df, gf, grp_f, scope=scope)
            te_lbl = apply_fences(test_df, gf, grp_f, scope=scope)

            tr_anom = tr_lbl["y_anom"].sum()
            te_anom = te_lbl["y_anom"].sum()
            gate_ok = (tr_anom >= cfg.MIN_MINORITY_TRAIN) and (te_anom >= cfg.MIN_MINORITY_TEST)
            warn = (tr_lbl["anom_A"].sum() < cfg.MIN_DIM_TRAIN_WARN) or \
                   (tr_lbl["anom_P"].sum() < cfg.MIN_DIM_TRAIN_WARN) or \
                   (tr_lbl["anom_Q"].sum() < cfg.MIN_DIM_TRAIN_WARN)

            rows.append({
                "Scope": scope,
                "IQR_K": k,
                "Train Anomalies": tr_anom,
                "Train Rate": f"{tr_anom/len(train_df):.2%}",
                "Test Anomalies": te_anom,
                "Test Rate": f"{te_anom/len(test_df):.2%}",
                "Gate": "PASS" if gate_ok else "FAIL",
                "Dim Warning": "Yes (<10 in A/P/Q)" if warn else "No"
            })
    sens_df = pd.DataFrame(rows)
    return sens_df


def plot_boxplot_with_fences(train_df: pd.DataFrame, global_fences: dict, output_dir: str):
    """
    Extended box plot visualization showing distributions, outliers, and training lower fences.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cols = ["Availability (A)", "Performance (P)", "Quality (Q)"]
    palette = ["#4A90E2", "#50E3C2", "#F5A623"]

    for ax, col, color in zip(axes, cols, palette):
        sns.boxplot(y=train_df[col], ax=ax, color=color, flierprops={'marker':'o', 'markerfacecolor':'crimson', 'markersize':4, 'alpha':0.6})
        fence = global_fences[col]
        ax.axhline(fence, color="red", linestyle="--", linewidth=1.5, label=f"Global Lower Fence ({fence:.3f})")
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_ylabel("Value")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.suptitle(f"Training Period Distributions & Outlier Fences (IQR K = {cfg.IQR_K})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "boxplot_outliers_fences.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"[labels] Saved boxplot with fences to {plot_path}")
