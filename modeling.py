"""
modeling.py – Build the imblearn pipeline, run GridSearchCV, return fitted model.

Public API
----------
build_pipeline()   -> imblearn.Pipeline
run_grid_search(pipeline, X_train, y_train) -> GridSearchCV (fitted)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

from features import FeatureSelector
import config as cfg


def build_pipeline() -> ImbPipeline:
    """Construct [FeatureSelector -> SMOTEENN -> RandomForest] pipeline."""
    pipe = ImbPipeline([
        ("selector", FeatureSelector(nzv_threshold=0.01, corr_threshold=0.90)),
        ("smoteenn", SMOTEENN(random_state=cfg.SEED)),
        ("clf",      RandomForestClassifier(random_state=cfg.SEED, n_jobs=-1)),
    ])
    return pipe


def _get_cv(y_train):
    """Return the right CV splitter based on SPLIT_MODE."""
    if cfg.SPLIT_MODE == "chronological":
        tscv = TimeSeriesSplit(n_splits=cfg.CV_FOLDS)
        # Verify every fold has both classes; if not, fall back to Stratified
        # with a warning. (TimeSeriesSplit is not class-aware.)
        from sklearn.model_selection import TimeSeriesSplit as TSS
        _tscv = TSS(n_splits=cfg.CV_FOLDS)
        all_ok = True
        for train_i, val_i in _tscv.split(np.arange(len(y_train))):
            if len(set(y_train.iloc[train_i])) < 2 or len(set(y_train.iloc[val_i])) < 2:
                all_ok = False
                break
        if all_ok:
            return tscv
        else:
            print("[modeling] WARNING: TimeSeriesSplit has folds with a single class; "
                  "using StratifiedKFold instead.")
            return StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True,
                                   random_state=cfg.SEED)
    else:
        return StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True,
                               random_state=cfg.SEED)


def run_grid_search(pipeline: ImbPipeline, X_train, y_train):
    """Run GridSearchCV with the configured parameter grid."""
    cv = _get_cv(y_train)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=cfg.PARAM_GRID,
        scoring="f1",
        refit=True,
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"[modeling] Best params: {gs.best_params_}")
    print(f"[modeling] Best CV F1:  {gs.best_score_:.4f}")
    return gs
