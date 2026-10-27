"""
evaluate.py – Evaluation metrics, baselines, confusion matrix, and plots.

Public API
----------
evaluate_model(gs, X_test, y_test, y_train, df, train_idx, test_idx)
    -> dict of metrics
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, auc,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

import config as cfg


def evaluate_model(gs, X_test, y_test, y_train, df, train_idx, test_idx) -> dict:
    """Full evaluation on the untouched test set."""
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    metrics = {
        "f1":        float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall":    float(recall_score(y_test, y_pred)),
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "brier":     float(brier_score_loss(y_test, y_prob)),
    }

    # PR-AUC
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    metrics["pr_auc"] = float(auc(rec_curve, prec_curve))

    # ── baselines ──────────────────────────────────────────
    # majority class
    majority = int(y_train.mode().iloc[0])
    y_maj = np.full(len(y_test), majority)
    metrics["baseline_majority_f1"] = float(f1_score(y_test, y_maj, zero_division=0))

    # persistence baseline (chronological only)
    if cfg.SPLIT_MODE == "chronological":
        # previous-run label persistence: use the label of the last train row
        # per machine+model
        y_full = pd.Series(index=df.index, dtype=int)
        y_full.loc[train_idx] = y_train.values
        y_full.loc[test_idx]  = y_test.values

        persist = []
        for idx in test_idx:
            machine = df.loc[idx, "machine"]
            model   = df.loc[idx, "Model"]
            prev = df.loc[
                (df.index < idx) &
                (df["machine"] == machine) &
                (df["Model"] == model)
            ]
            if len(prev) > 0:
                persist.append(y_full.loc[prev.index[-1]])
            else:
                persist.append(majority)
        metrics["baseline_persistence_f1"] = float(f1_score(y_test, persist, zero_division=0))

    # ── classification report ──────────────────────────────
    cls_report = classification_report(y_test, y_pred, digits=4)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("=" * 60)
    print(cls_report)

    # ── confusion matrix ───────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm)

    # ── ROC curve ──────────────────────────────────────────
    _plot_roc(y_test, y_prob, metrics["roc_auc"])

    # ── save metrics ───────────────────────────────────────
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved confusion_matrix.png")


def _plot_roc(y_test, y_prob, roc_auc_val):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved roc_curve.png")
