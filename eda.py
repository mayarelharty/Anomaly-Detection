"""
eda.py – Exploratory data analysis and diagnostic plots.

Public API
----------
run_eda(df) -> None   (prints stats, saves figures to OUTPUT_DIR)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg


def run_eda(df: pd.DataFrame) -> None:
    """Run full EDA: shape, dtypes, missing, stats, distributions, correlations."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ── shape & dtypes ─────────────────────────────────────
    print(f"\nShape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")

    # ── missing values ─────────────────────────────────────
    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0]}" if missing.sum() > 0 else "\nNo missing values.")

    # ── duplicates ─────────────────────────────────────────
    print(f"\nExact duplicate rows: {df.duplicated().sum()}")

    # ── infinities ─────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[num_cols]).sum()
    print(f"\nInfinities:\n{inf_count[inf_count > 0]}" if inf_count.sum() > 0 else "\nNo infinities.")

    # ── descriptive statistics ─────────────────────────────
    print(f"\nDescriptive Statistics:\n{df[num_cols].describe().to_string()}")

    # ── counts by Model and machine ────────────────────────
    print(f"\nCounts by Model:\n{df['Model'].value_counts().to_string()}")
    print(f"\nCounts by Machine:\n{df['machine'].value_counts().to_string()}")

    # ── extreme values report ──────────────────────────────
    _report_extremes(df)

    # ── plots ──────────────────────────────────────────────
    _plot_distributions(df)
    _plot_correlation(df)
    _plot_oee_time(df)


def _report_extremes(df: pd.DataFrame):
    """Report potential extreme values without removing them."""
    print("\n--- Extreme value report ---")
    for col in ["Availability (A)", "Performance (P)", "Quality (Q)", "OEE"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        extremes = df[(df[col] < lower) | (df[col] > upper)]
        if len(extremes) > 0:
            print(f"  {col}: {len(extremes)} rows outside 3×IQR [{lower:.4f}, {upper:.4f}]")
        else:
            print(f"  {col}: no extreme outliers (3×IQR)")


def _plot_distributions(df: pd.DataFrame):
    """Histograms of A, P, Q, OEE."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flat, ["Availability (A)", "Performance (P)",
                                    "Quality (Q)", "OEE"]):
        ax.hist(df[col].dropna(), bins=40, edgecolor="black", alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"mean={df[col].mean():.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("Distributions of A, P, Q, OEE", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "eda_distributions.png"), dpi=150)
    plt.close(fig)
    print("[eda] Saved eda_distributions.png")


def _plot_correlation(df: pd.DataFrame):
    """Correlation heatmap of numeric columns."""
    num_cols = [
        "Operating time (min)", "Production (pcs)",
        "Planned Downtime (min)", "Unplanned Downtime (min)",
        "Total Downtime (min)", "Defects (pcs)",
        "Cycle time – Min CT (min/pc)",
        "Availability (A)", "Performance (P)", "Quality (Q)", "OEE",
    ]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "eda_correlation.png"), dpi=150)
    plt.close(fig)
    print("[eda] Saved eda_correlation.png")


def _plot_oee_time(df: pd.DataFrame):
    """OEE over time by machine."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for machine, grp in df.groupby("machine"):
        daily = grp.groupby("Date")["OEE"].mean()
        ax.plot(daily.index, daily.values, alpha=0.6, label=machine, linewidth=0.8)
    ax.set_title("Daily Mean OEE by Machine")
    ax.set_xlabel("Date")
    ax.set_ylabel("OEE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "eda_oee_over_time.png"), dpi=150)
    plt.close(fig)
    print("[eda] Saved eda_oee_over_time.png")
