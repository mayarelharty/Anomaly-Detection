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
FENCE_COLS = DIMS + ["OEE"]

METHOD_RULES = {
    "iqr":        "Box plot: $D < Q_1 - K \\times \\text{IQR}$ on any of A, P, Q",
    "percentile": f"Alt A: $D < P_{{{int(cfg.FALLBACK_PERCENTILE * 100)}}}$ on any of A, P, Q",
    "oee":        "Alt B: $\\text{OEE} < Q_1 - K \\times \\text{IQR}$ (single composite fence)",
    "benchmark":  "Alt C: $A < 0.90 \\lor P < 0.95 \\lor Q < 0.999$ (static industry targets)",
}


def _lower_fence(s: pd.Series, k: float, method: str) -> float:
    """Training-only lower fence of one series; Alt A uses a percentile, everything else Tukey IQR."""
    if method == "percentile":
        return float(s.quantile(cfg.FALLBACK_PERCENTILE))
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return float(q1 - k * (q3 - q1))


def compute_fences(train_df: pd.DataFrame, k: float = None, scope: str = None, min_group_train: int = None,
                   method: str = None):
    """
    Compute lower fences for A, P, Q and OEE strictly from training rows.
    method: "iqr" | "percentile" (Alt A) | "oee" (Alt B, IQR fences, label on OEE) | "benchmark" (Alt C, static, global)
    Returns:
        global_fences: dict of {dim: lower_fence}
        group_fences: dict of {(machine, model): {dim: lower_fence}}
        group_counts: dict of {(machine, model): count}
    """
    k = cfg.IQR_K if k is None else k
    scope = cfg.IQR_SCOPE if scope is None else scope
    min_group_train = cfg.MIN_GROUP_TRAIN if min_group_train is None else min_group_train
    method = cfg.THRESHOLD_METHOD if method is None else method

    group_counts = {key: len(grp) for key, grp in train_df.groupby(["machine", "Model"])}

    if method == "benchmark":
        global_fences = dict(cfg.BENCHMARK_TARGETS)
        global_fences["OEE"] = float(np.prod(list(cfg.BENCHMARK_TARGETS.values())))
        return global_fences, {}, group_counts

    global_fences = {d: _lower_fence(train_df[d], k, method) for d in FENCE_COLS}

    group_fences = {}
    if scope == "machine_model":
        for (m, mod), grp in train_df.groupby(["machine", "Model"]):
            if len(grp) >= min_group_train:
                group_fences[(m, mod)] = {d: _lower_fence(grp[d], k, method) for d in FENCE_COLS}

    return global_fences, group_fences, group_counts


def apply_fences(df: pd.DataFrame, global_fences: dict, group_fences: dict, scope: str = None,
                 method: str = None) -> pd.DataFrame:
    """
    Apply pre-computed training fences to a dataset (train or test).
    y_anom is the OR of the dimension flags, except Alt B ("oee") where it is OEE < OEE fence.
    Returns DataFrame with columns ['anom_A', 'anom_P', 'anom_Q', 'y_anom'].
    """
    scope = cfg.IQR_SCOPE if scope is None else scope
    method = cfg.THRESHOLD_METHOD if method is None else method
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
        if method == "oee":
            y_anom.append(int(row["OEE"] < f["OEE"]))
        else:
            y_anom.append(int(is_a or is_p or is_q))

    out = pd.DataFrame({
        "anom_A": anom_A,
        "anom_P": anom_P,
        "anom_Q": anom_Q,
        "y_anom": y_anom
    }, index=df.index)
    return out


def _gate_passes(train_labels: pd.DataFrame, test_labels: pd.DataFrame) -> bool:
    n_tr, n_te = int(train_labels["y_anom"].sum()), int(test_labels["y_anom"].sum())
    return (min(n_tr, len(train_labels) - n_tr) >= cfg.MIN_MINORITY_TRAIN
            and min(n_te, len(test_labels) - n_te) >= cfg.MIN_MINORITY_TEST)


def gate_check(train_labels: pd.DataFrame, test_labels: pd.DataFrame, alternatives: pd.DataFrame = None):
    """
    Verify label gate criteria:
    - >= MIN_MINORITY_TRAIN training rows in the minority class
    - >= MIN_MINORITY_TEST test rows in each class
    - Warn if any single dimension has < 10 training positives
    On failure, recommends the threshold alternatives (from `alternatives`) that would pass.
    Returns (passed, messages).
    """
    n_train_anom = int(train_labels["y_anom"].sum())
    n_test_anom = int(test_labels["y_anom"].sum())
    n_train_norm = len(train_labels) - n_train_anom
    n_test_norm = len(test_labels) - n_test_anom

    passed = True
    msgs = []

    if min(n_train_anom, n_train_norm) < cfg.MIN_MINORITY_TRAIN:
        msgs.append(f"GATE FAIL: Train minority class = {min(n_train_anom, n_train_norm)} "
                    f"(anomalies {n_train_anom}, normal {n_train_norm}; requires >= {cfg.MIN_MINORITY_TRAIN})")
        passed = False
    if min(n_test_anom, n_test_norm) < cfg.MIN_MINORITY_TEST:
        msgs.append(f"GATE FAIL: Test anomalies = {n_test_anom}, normal = {n_test_norm} "
                    f"(requires >= {cfg.MIN_MINORITY_TEST} of each class)")
        passed = False
    if not passed and alternatives is not None:
        ok = alternatives[alternatives["Gate"] == "PASS"]
        if n_train_anom < cfg.MIN_MINORITY_TRAIN and (ok["Method"] == "percentile").any():
            msgs.append("RECOMMENDATION: too few anomalies; use Alternative A (`THRESHOLD_METHOD = 'percentile'`).")
        elif len(ok):
            msgs.append("RECOMMENDATION: switch `THRESHOLD_METHOD` to one of the passing alternatives: "
                        + ", ".join(f"'{m}' ({a})" for m, a in zip(ok["Method"], ok["Alternative"])) + ".")
        else:
            msgs.append("RECOMMENDATION: no documented alternative passes the gate; extend the data window "
                        "or lower MIN_MINORITY_TRAIN / MIN_MINORITY_TEST.")

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
        return False, msgs

    print(f"[gate] PASSED ✓ Train anomalies = {n_train_anom}, Test anomalies = {n_test_anom}")
    return True, msgs


def resolve_labels(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Label train/test with cfg.THRESHOLD_METHOD. If the IQR fences give fewer than
    MIN_MINORITY_TRAIN training anomalies and AUTO_FALLBACK_PERCENTILE is on, fall back to
    Alternative A (percentile fences). Returns (method, global_fences, group_fences, train_labels,
    test_labels, note).
    """
    method, note = cfg.THRESHOLD_METHOD, None
    gf, grpf, _ = compute_fences(train_df, method=method)
    tr = apply_fences(train_df, gf, grpf, method=method)
    te = apply_fences(test_df, gf, grpf, method=method)
    if method == "iqr" and cfg.AUTO_FALLBACK_PERCENTILE and tr["y_anom"].sum() < cfg.MIN_MINORITY_TRAIN:
        note = (f"IQR fences produced {int(tr['y_anom'].sum())} training anomalies (< {cfg.MIN_MINORITY_TRAIN}); "
                f"fell back to Alternative A (P{int(cfg.FALLBACK_PERCENTILE * 100)} percentile fences).")
        method = "percentile"
        gf, grpf, _ = compute_fences(train_df, method=method)
        tr = apply_fences(train_df, gf, grpf, method=method)
        te = apply_fences(test_df, gf, grpf, method=method)
    return method, gf, grpf, tr, te, note


def run_alternatives_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Label counts and gate status for the box-plot rule and the three documented alternatives."""
    rows = []
    for method, name in [("iqr", "Box plot (default)"), ("percentile", "Alternative A"),
                         ("oee", "Alternative B"), ("benchmark", "Alternative C")]:
        gf, grpf, _ = compute_fences(train_df, method=method)
        tr = apply_fences(train_df, gf, grpf, method=method)
        te = apply_fences(test_df, gf, grpf, method=method)
        tr_anom, te_anom = int(tr["y_anom"].sum()), int(te["y_anom"].sum())
        rows.append({
            "Alternative": name,
            "Method": method,
            "Rule": METHOD_RULES[method],
            "Train Anomalies": tr_anom,
            "Train Rate": f"{tr_anom/len(train_df):.2%}",
            "Test Anomalies": te_anom,
            "Test Rate": f"{te_anom/len(test_df):.2%}",
            "Gate": "PASS" if _gate_passes(tr, te) else "FAIL",
        })
    return pd.DataFrame(rows)


def label_distribution_report(df: pd.DataFrame, train_labels: pd.DataFrame, test_labels: pd.DataFrame,
                              train_idx, test_idx, global_fences: dict, group_fences: dict, method: str = None):
    """
    Generate comprehensive label distribution summary, overlap patterns, and quarterly drift.
    """
    method = cfg.THRESHOLD_METHOD if method is None else method
    fence_desc = {
        "iqr":        f"$Q_1 - {cfg.IQR_K} \\times \\text{{IQR}}$ on Train",
        "oee":        f"$Q_1 - {cfg.IQR_K} \\times \\text{{IQR}}$ on Train",
        "percentile": f"$P_{{{int(cfg.FALLBACK_PERCENTILE * 100)}}}$ on Train",
        "benchmark":  "Static industry target",
    }[method]
    label_dims = ["OEE"] if method == "oee" else DIMS
    lines = []
    lines.append("## Label Distribution (Box-Plot IQR Rule)" if method == "iqr"
                 else f"## Label Distribution ({METHOD_RULES[method].split(':')[0]})")
    lines.append(f"- Labelling rule: {METHOD_RULES[method]}.")
    lines.append(f"- Configuration: `THRESHOLD_METHOD = '{method}'`, `IQR_K = {cfg.IQR_K}`, `IQR_SCOPE = '{cfg.IQR_SCOPE}'`, `MIN_GROUP_TRAIN = {cfg.MIN_GROUP_TRAIN}`")
    lines.append("")

    # Global fences
    lines.append("### Global Training Fences")
    lines.append("| Dimension | Lower Fence | Description | Used for $y_{\\text{anom}}$ |")
    lines.append("|---|---|---|---|")
    for d in FENCE_COLS:
        lines.append(f"| {d} | **{global_fences[d]:.4f}** | {fence_desc} | {'Yes' if d in label_dims else 'No'} |")
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
            gf, grp_f, _ = compute_fences(train_df, k=k, scope=scope, method="iqr")
            tr_lbl = apply_fences(train_df, gf, grp_f, scope=scope, method="iqr")
            te_lbl = apply_fences(test_df, gf, grp_f, scope=scope, method="iqr")

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
    Extended box plot visualization showing distributions, outliers, and training lower fences
    for A, P, Q and OEE.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    cols = FENCE_COLS
    palette = ["#4A90E2", "#50E3C2", "#F5A623", "#9B59B6"]

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
