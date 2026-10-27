"""
scoring.py – Anomaly score columns (in-sample, out-of-fold, enriched output).

Public API
----------
score_anomalies(gs, X, y, df, train_idx, test_idx, X_train, y_train)
    -> pd.DataFrame  (enriched dataset)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

import config as cfg
from modeling import build_pipeline


def score_anomalies(gs, X, y, df, train_idx, test_idx, X_train, y_train) -> pd.DataFrame:
    """
    Produce Anomaly_Score and Anomaly_Score_OOF columns.

    - Anomaly_Score:     predict_proba from the final model on ORIGINAL
                         (unbalanced) train and test sets separately.
    - Anomaly_Score_OOF: out-of-fold probabilities for train rows (using
                         the same CV scheme, refit pipeline per fold with
                         best params);  final model scores for test rows.

    Never scores SMOTEENN-resampled data.
    """
    best = gs.best_estimator_
    best_params = gs.best_params_

    # ── Anomaly_Score: final model on original data ────────
    score_train = best.predict_proba(X.loc[train_idx])[:, 1]
    score_test  = best.predict_proba(X.loc[test_idx])[:, 1]

    anomaly_score = pd.Series(np.nan, index=df.index, dtype=float)
    anomaly_score.iloc[train_idx] = score_train
    anomaly_score.iloc[test_idx]  = score_test

    # ── Anomaly_Score_OOF ──────────────────────────────────
    oof_scores = pd.Series(np.nan, index=df.index, dtype=float)

    # out-of-fold for train
    if cfg.SPLIT_MODE == "chronological":
        cv = TimeSeriesSplit(n_splits=cfg.CV_FOLDS)
        # Check fold validity - fall back to stratified if needed
        all_ok = True
        for tr_i, va_i in cv.split(X_train):
            if len(set(y_train.iloc[tr_i])) < 2 or len(set(y_train.iloc[va_i])) < 2:
                all_ok = False
                break
        if not all_ok:
            cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True,
                                 random_state=cfg.SEED)
    else:
        cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True,
                             random_state=cfg.SEED)

    oof_train = np.full(len(train_idx), np.nan)
    for fold_i, (tr_i, va_i) in enumerate(cv.split(X_train, y_train)):
        pipe = build_pipeline()
        pipe.set_params(**best_params)
        pipe.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
        oof_train[va_i] = pipe.predict_proba(X_train.iloc[va_i])[:, 1]
        print(f"  [scoring] OOF fold {fold_i+1}/{cfg.CV_FOLDS} done")

    oof_scores.iloc[train_idx] = oof_train
    # test rows: use final model
    oof_scores.iloc[test_idx] = score_test

    # ── assemble enriched dataset ──────────────────────────
    enriched = df.copy()
    enriched["split"]             = "train"
    enriched.loc[test_idx, "split"] = "test"
    enriched["Anomaly_Score"]     = anomaly_score.values
    enriched["Anomaly_Score_OOF"] = oof_scores.values

    # assertion: row order matches original
    assert list(enriched.index) == list(df.index), "Row order mismatch!"

    # ── report in-sample vs out-of-fold gap ────────────────
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 60)
    print("ANOMALY SCORE SUMMARY")
    print("=" * 60)
    for name, idx, col in [
        ("Train (in-sample)", train_idx, "Anomaly_Score"),
        ("Train (OOF)",       train_idx, "Anomaly_Score_OOF"),
        ("Test",              test_idx,  "Anomaly_Score"),
    ]:
        s = enriched.loc[idx, col].dropna()
        yt = y.loc[s.index]
        if len(s) == 0 or yt.nunique() < 2:
            auc_val = float("nan")
        else:
            auc_val = roc_auc_score(yt, s)
        print(f"  {name:25s}  mean_score={s.mean():.4f}  ROC-AUC={auc_val:.4f}  (n={len(s)})")

    # ── save ───────────────────────────────────────────────
    out_path = os.path.join(cfg.OUTPUT_DIR, "enriched_dataset.csv")
    try:
        enriched.to_csv(out_path, index=False)
        print(f"[scoring] Saved enriched_dataset.csv ({enriched.shape})")
    except PermissionError:
        alt_path = os.path.join(cfg.OUTPUT_DIR, "enriched_dataset_run.csv")
        enriched.to_csv(alt_path, index=False)
        print(f"[scoring] [WARNING] `enriched_dataset.csv` locked. Saved to {alt_path} ({enriched.shape})")

    return enriched
