"""
modeling.py – Multi-Variant Model Training, Walk-Forward OOF, and Calibration.

Implements three imbalance variants:
- Variant A: SMOTEENN with SMOTENC for categorical features (Primary)
- Variant B: Balanced class weights (no resampling)
- Variant C: Reference model (unweighted, no resampling)

Fits isotonic calibration and threshold optimization on walk-forward OOF predictions only.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC

from features import ScaledFeatureSelector
import config as cfg


def build_pipeline_variant(variant_name: str, cat_indices: list = None) -> ImbPipeline:
    """
    Build scikit-learn / imblearn pipeline for specified variant.
    """
    cat_indices = [0, 1, 2, 3] if cat_indices is None else cat_indices
    selector = ScaledFeatureSelector(nzv_threshold=0.01, corr_threshold=0.90)

    if variant_name == "smoteenn":
        smote = SMOTENC(categorical_features=cat_indices, random_state=cfg.SEED)
        resampler = SMOTEENN(smote=smote, random_state=cfg.SEED)
        pipe = ImbPipeline([
            ("selector", selector),
            ("resample", resampler),
            ("clf", RandomForestClassifier(random_state=cfg.SEED, n_jobs=-1)),
        ])
    elif variant_name == "balanced":
        pipe = ImbPipeline([
            ("selector", selector),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=cfg.SEED, n_jobs=-1)),
        ])
    elif variant_name == "reference":
        pipe = ImbPipeline([
            ("selector", selector),
            ("clf", RandomForestClassifier(random_state=cfg.SEED, n_jobs=-1)),
        ])
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    return pipe


def train_and_calibrate_variant(variant_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, cat_indices: list = None):
    """
    Train a model variant:
    1. Run GridSearchCV with walk-forward TimeSeriesSplit(5)
    2. Collect walk-forward Out-Of-Fold (OOF) predictions
    3. Fit IsotonicRegression calibrator on non-NaN OOF predictions
    4. Select optimal decision cutoff on calibrated OOF predictions (F1 and F2)
    5. Score untouched test set with both raw and calibrated model
    """
    t0 = time.time()
    print(f"\n[modeling] Training variant: '{variant_name}'...")
    pipe = build_pipeline_variant(variant_name, cat_indices=cat_indices)
    cv = TimeSeriesSplit(n_splits=cfg.CV_FOLDS)

    # 1. GridSearchCV
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=cfg.PARAM_GRID,
        scoring="f1",
        refit=True,
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)
    best_params = gs.best_params_
    best_cv_f1  = float(gs.best_score_)
    final_model = gs.best_estimator_

    print(f"  [modeling] Best params: {best_params} (CV F1: {best_cv_f1:.4f}) [{time.time()-t0:.1f}s]")

    # 2. Collect walk-forward OOF predictions
    oof_raw = np.full(len(X_train), np.nan)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train)):
        fold_pipe = build_pipeline_variant(variant_name, cat_indices=cat_indices)
        fold_pipe.set_params(**best_params)
        fold_pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_raw[val_idx] = fold_pipe.predict_proba(X_train.iloc[val_idx])[:, 1]

    warm_up_nans = int(np.isnan(oof_raw).sum())
    valid_oof_mask = ~np.isnan(oof_raw)
    y_oof_valid = y_train.iloc[valid_oof_mask].values
    oof_valid_scores = oof_raw[valid_oof_mask]

    # 3. Fit Isotonic Calibrator strictly on OOF predictions
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(oof_valid_scores, y_oof_valid)
    oof_calibrated = np.full(len(X_train), np.nan)
    oof_calibrated[valid_oof_mask] = calibrator.predict(oof_valid_scores)

    # 4. Evaluate the fixed 0.50 decision cutoff on OOF calibrated predictions
    cutoffs = [0.50]
    best_f1, best_cut_f1 = -1.0, 0.5
    best_f2, best_cut_f2 = -1.0, 0.5
    cal_valid_preds = oof_calibrated[valid_oof_mask]

    for c in cutoffs:
        bin_pred = (cal_valid_preds >= c).astype(int)
        f1 = f1_score(y_oof_valid, bin_pred, zero_division=0)
        f2 = fbeta_score(y_oof_valid, bin_pred, beta=2, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_cut_f1 = float(c)
        if f2 > best_f2:
            best_f2 = f2
            best_cut_f2 = float(c)

    oof_auc = float(roc_auc_score(y_oof_valid, oof_valid_scores)) if len(np.unique(y_oof_valid)) > 1 else float("nan")

    # 5. Predict on untouched test set
    test_raw_scores = final_model.predict_proba(X_test)[:, 1]
    test_cal_scores = calibrator.predict(test_raw_scores)

    return {
        "variant": variant_name,
        "final_model": final_model,
        "grid_search": gs,
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "cv_results": pd.DataFrame(gs.cv_results_),
        "calibrator": calibrator,
        "oof_raw": oof_raw,
        "oof_calibrated": oof_calibrated,
        "warm_up_nans": warm_up_nans,
        "oof_auc": oof_auc,
        "best_cut_f1": best_cut_f1,
        "oof_best_f1": best_f1,
        "best_cut_f2": best_cut_f2,
        "oof_best_f2": best_f2,
        "test_raw_scores": test_raw_scores,
        "test_cal_scores": test_cal_scores,
    }
