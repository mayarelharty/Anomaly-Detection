"""
evaluate.py – Evaluation Suite for Test Period & Multi-Variant Comparison.

Evaluates raw and calibrated models:
- F1, Precision, Recall, ROC-AUC, PR-AUC, Brier score, Accuracy
- Precision at matched recall values (0.5, 0.6, 0.7)
- Three competitive baselines (Majority, Persistence, Training Group Base Rate)
- Bootstrap 95% Confidence Intervals (1000 resamples)
- Per-quarter metrics for temporal drift
- Per-dimension breakdown (anom_A, anom_P, anom_Q)
- MDI & Permutation Feature Importance
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

import config as cfg


def evaluate_single_variant(variant_name: str, y_test: pd.Series, raw_scores: np.ndarray,
                            cal_scores: np.ndarray, cutoff: float = None):
    """
    Compute comprehensive metrics for both raw and calibrated versions of a variant
    at the fixed decision cutoff (raw scores are the primary alarm signal).
    """
    cutoff = cfg.DECISION_CUTOFF if cutoff is None else cutoff
    y_true = y_test.values
    prevalence = float(np.mean(y_true))
    total_anomalies = int(np.sum(y_true))

    results = {}

    # Configurations to evaluate:
    configs = [
        ("raw", raw_scores, cutoff, f"Raw @ Cutoff {cutoff:.2f} (Primary)"),
        ("calibrated", cal_scores, cutoff, f"Calibrated @ Cutoff {cutoff:.2f}"),
    ]

    for cfg_key, scores, cut, desc in configs:
        preds = (scores >= cut).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        acc = float(accuracy_score(y_true, preds))
        brier = float(brier_score_loss(y_true, scores))
        roc_auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
        pr_auc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
        n_alarms = int(np.sum(preds))
        mean_score = float(np.mean(scores))
        cm = confusion_matrix(y_true, preds).tolist()
        cr = classification_report(y_true, preds, zero_division=0, output_dict=True)

        results[cfg_key] = {
            "variant": variant_name,
            "description": desc,
            "cutoff": cut,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "accuracy": acc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier": brier,
            "alarms_count": n_alarms,
            "true_anomalies": total_anomalies,
            "mean_score": mean_score,
            "prevalence": prevalence,
            "confusion_matrix": cm,
            "classification_report": cr,
        }

    # Precision at matched recall values
    prec_at_rec = precision_at_matched_recall(y_true, raw_scores, target_recalls=[0.50, 0.60, 0.70])
    results["precision_at_matched_recall"] = prec_at_rec

    return results


def precision_at_matched_recall(y_true: np.ndarray, scores: np.ndarray, target_recalls=(0.50, 0.60, 0.70)):
    """
    Interpolate precision from PR curve at fixed recall targets.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    matched = {}
    for tr in target_recalls:
        valid_p = [p for p, r in zip(precision, recall) if r >= tr]
        matched[f"prec_at_recall_{int(tr*100)}"] = float(max(valid_p)) if valid_p else 0.0
    return matched


def compute_baselines(y_test: pd.Series, X_test: pd.DataFrame, train_group_rates: dict,
                      df_feat_full: pd.DataFrame, valid_test_idx: list):
    """
    Evaluate 3 competitive baselines on the same untouched test rows:
    1. Majority Class (always predict 0)
    2. Previous Run Label (persistence from lag1_y_anom)
    3. Training Group Base Rate (predict anomaly if group training prevalence >= test prevalence)
    """
    y_true = y_test.values
    baselines = {}

    # 1. Majority class
    pred_maj = np.zeros_like(y_true)
    baselines["majority_class"] = {
        "description": "Always Predict Normal (Majority Class)",
        "f1": float(f1_score(y_true, pred_maj, zero_division=0)),
        "precision": float(precision_score(y_true, pred_maj, zero_division=0)),
        "recall": float(recall_score(y_true, pred_maj, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred_maj)),
        "roc_auc": 0.50,
        "brier": float(np.mean((0 - y_true)**2)),
    }

    # 2. Persistence baseline (lag1_y_anom)
    pred_persist = X_test["lag1_y_anom"].fillna(0).astype(int).values
    baselines["persistence_lag1"] = {
        "description": "Previous-Run Label Persistence",
        "f1": float(f1_score(y_true, pred_persist, zero_division=0)),
        "precision": float(precision_score(y_true, pred_persist, zero_division=0)),
        "recall": float(recall_score(y_true, pred_persist, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred_persist)),
        "roc_auc": float(roc_auc_score(y_true, pred_persist)),
        "brier": float(brier_score_loss(y_true, pred_persist)),
    }

    # 3. Per machine+model training base rate
    test_rows = df_feat_full.loc[valid_test_idx]
    pred_base_rate_prob = np.array([train_group_rates.get((r["machine"], r["Model"]), 0.1288) for _, r in test_rows.iterrows()])
    test_prev = np.mean(y_true)
    pred_base_rate = (pred_base_rate_prob >= test_prev).astype(int)

    baselines["training_group_base_rate"] = {
        "description": "Per Machine+Model Training Base Rate",
        "f1": float(f1_score(y_true, pred_base_rate, zero_division=0)),
        "precision": float(precision_score(y_true, pred_base_rate, zero_division=0)),
        "recall": float(recall_score(y_true, pred_base_rate, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred_base_rate)),
        "roc_auc": float(roc_auc_score(y_true, pred_base_rate_prob)),
        "brier": float(brier_score_loss(y_true, pred_base_rate_prob)),
    }

    return baselines


def bootstrap_ci(y_true: np.ndarray, scores: np.ndarray, cutoff: float, n_boot: int = None, seed: int = None):
    """
    Bootstrap 95% confidence intervals (seeded, 1000 resamples).
    """
    n_boot = cfg.BOOTSTRAP_ROUNDS if n_boot is None else n_boot
    seed = cfg.SEED if seed is None else seed
    rng = np.random.RandomState(seed)

    f1_list, prec_list, rec_list, auc_list, pr_auc_list = [], [], [], [], []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        s_b = scores[idx]
        p_b = (s_b >= cutoff).astype(int)

        f1_list.append(f1_score(y_b, p_b, zero_division=0))
        prec_list.append(precision_score(y_b, p_b, zero_division=0))
        rec_list.append(recall_score(y_b, p_b, zero_division=0))
        auc_list.append(roc_auc_score(y_b, s_b))
        pr_auc_list.append(average_precision_score(y_b, s_b))

    ci = {}
    for metric_name, vals in [("f1", f1_list), ("precision", prec_list), ("recall", rec_list),
                              ("roc_auc", auc_list), ("pr_auc", pr_auc_list)]:
        low = float(np.percentile(vals, 2.5))
        high = float(np.percentile(vals, 97.5))
        ci[metric_name] = {"low_95": low, "high_95": high, "ci_str": f"[{low:.4f}, {high:.4f}]"}
    return ci


def quarterly_drift_analysis(df_feat_full: pd.DataFrame, valid_test_idx: list, y_test: pd.Series,
                             scores: np.ndarray, cutoff: float):
    """
    Compute metrics by calendar quarter to show temporal drift.
    """
    test_df = df_feat_full.loc[valid_test_idx].copy()
    test_df["Quarter"] = test_df["Date"].dt.to_period("Q").astype(str)
    test_df["y_true"] = y_test.values
    test_df["score"] = scores
    test_df["y_pred"] = (scores >= cutoff).astype(int)

    q_metrics = {}
    for qtr, grp in test_df.groupby("Quarter"):
        yt = grp["y_true"].values
        yp = grp["y_pred"].values
        sc = grp["score"].values
        cnt = len(grp)
        anom = int(np.sum(yt))
        rate = float(np.mean(yt))
        roc_val = float(roc_auc_score(yt, sc)) if len(np.unique(yt)) > 1 else float("nan")
        q_metrics[qtr] = {
            "quarter": qtr,
            "shifts": cnt,
            "anomalies": anom,
            "anomaly_rate": rate,
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "roc_auc": roc_val,
        }
    return q_metrics


def per_dimension_evaluation(df_labels_full: pd.DataFrame, valid_test_idx: list, pred_binary: np.ndarray):
    """
    Evaluate how well the model catches specific dimension anomalies (anom_A, anom_P, anom_Q).
    """
    test_labels = df_labels_full.loc[valid_test_idx]
    dim_results = {}
    for d, short in [("anom_A", "Availability"), ("anom_P", "Performance"), ("anom_Q", "Quality")]:
        y_dim = test_labels[d].values
        cnt = int(np.sum(y_dim))
        if cnt > 0:
            rec = float(recall_score(y_dim, pred_binary, zero_division=0))
            prec = float(precision_score(y_dim, pred_binary, zero_division=0))
            f1 = float(f1_score(y_dim, pred_binary, zero_division=0))
        else:
            rec, prec, f1 = float("nan"), float("nan"), float("nan")
        dim_results[short] = {
            "test_positives": cnt,
            "recall": rec,
            "precision": prec,
            "f1": f1
        }
    return dim_results


def plot_all_figures(all_variants_data: dict, baselines: dict, feature_cols: list,
                     X_test: pd.DataFrame, y_test: pd.Series, output_dir: str):
    """
    Generate all required publication-ready plots:
    - ROC Curves
    - PR Curves
    - Calibration Curves
    - Confusion Matrix (Primary Variant)
    - Feature Importance (MDI and Permutation)
    - Imbalance Variant Comparison Bar Chart
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true = y_test.values

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for vname, data in all_variants_data.items():
        fpr, tpr, _ = roc_curve(y_true, data["test_raw_scores"])
        auc = roc_auc_score(y_true, data["test_raw_scores"])
        ax.plot(fpr, tpr, label=f"{vname.upper()} Raw (AUC = {auc:.3f})", lw=2)
    # Add persistence baseline
    persist_scores = X_test["lag1_y_anom"].fillna(0).values
    fpr_p, tpr_p, _ = roc_curve(y_true, persist_scores)
    ax.plot(fpr_p, tpr_p, 'k--', label=f"Persistence Baseline (AUC = {roc_auc_score(y_true, persist_scores):.3f})")
    ax.plot([0, 1], [0, 1], ':', color='grey', label="Chance (AUC = 0.50)")
    ax.set_title("Test Period ROC Curves (Multi-Variant Comparison)", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # 2. PR Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for vname, data in all_variants_data.items():
        prec, rec, _ = precision_recall_curve(y_true, data["test_raw_scores"])
        ap = average_precision_score(y_true, data["test_raw_scores"])
        ax.plot(rec, prec, label=f"{vname.upper()} (PR-AUC = {ap:.3f})", lw=2)
    ax.axhline(np.mean(y_true), color='grey', linestyle=':', label=f"Test Prevalence ({np.mean(y_true):.3f})")
    ax.set_title("Test Period Precision-Recall Curves", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=200)
    plt.close()

    # 3. Calibration Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for vname, data in all_variants_data.items():
        prob_true_raw, prob_pred_raw = calibration_curve(y_true, data["test_raw_scores"], n_bins=5)
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, data["test_cal_scores"], n_bins=5)
        ax.plot(prob_pred_raw, prob_true_raw, 'o--', label=f"{vname.upper()} Raw", alpha=0.7)
        ax.plot(prob_pred_cal, prob_true_cal, 's-', label=f"{vname.upper()} Calibrated", lw=2)
    ax.plot([0, 1], [0, 1], 'k:', label="Perfect Calibration")
    ax.set_title("Reliability Diagrams / Calibration Curves", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of True Anomalies")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=200)
    plt.close()

    # 4. Confusion Matrix (Primary Variant: Raw Scores @ Fixed Decision Cutoff)
    prim_data = all_variants_data[cfg.PRIMARY_VARIANT]
    cutoff = prim_data["cutoff"]
    prim_pred = (prim_data["test_raw_scores"] >= cutoff).astype(int)
    cm = confusion_matrix(y_true, prim_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal (0)", "Anomaly (1)"],
                yticklabels=["Normal (0)", "Anomaly (1)"])
    ax.set_title(f"Confusion Matrix: {cfg.PRIMARY_VARIANT.upper()} (Raw, Cutoff = {cutoff:.2f})", fontweight="bold", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual Ground Truth")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # 5. Feature Importance (MDI and Permutation)
    final_rf = prim_data["final_model"].named_steps["clf"]
    selector = prim_data["final_model"].named_steps["selector"]
    retained_names = selector.get_feature_names_out()

    mdi_imp = final_rf.feature_importances_
    perm = permutation_importance(prim_data["final_model"], X_test, y_true, n_repeats=10, random_state=cfg.SEED, scoring="f1")
    perm_imp = perm.importances_mean

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # MDI
    s_idx = np.argsort(mdi_imp)[::-1]
    ax1.barh(range(len(s_idx)), mdi_imp[s_idx][::-1], color="#4A90E2")
    ax1.set_yticks(range(len(s_idx)))
    ax1.set_yticklabels([retained_names[i] for i in s_idx[::-1]])
    ax1.set_title("MDI Feature Importance (Train RF)", fontweight="bold")
    ax1.set_xlabel("Gini Importance")

    # Permutation
    p_idx = np.argsort(perm_imp)[::-1]
    ax2.barh(range(len(p_idx)), perm_imp[p_idx][::-1], color="#50E3C2")
    ax2.set_yticks(range(len(p_idx)))
    ax2.set_yticklabels([feature_cols[i] for i in p_idx[::-1]])
    ax2.set_title("Permutation Importance (Test Set F1 Drop)", fontweight="bold")
    ax2.set_xlabel("Mean Decrease in F1")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=200)
    plt.close()

    print("[evaluate] All plots saved successfully ✓")
