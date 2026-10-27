"""
features.py – Feature construction and selection transformers.

Public API
----------
build_features(df, train_idx) -> (X: pd.DataFrame, feature_names: list)
FeatureSelector        – sklearn-compatible transformer for pipeline use
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import config as cfg


# ────────────────────── feature matrix construction ────────
def build_features(df: pd.DataFrame, train_idx=None):
    """
    Build the feature matrix X (numeric-only) according to FEATURE_MODE.

    Returns (X: pd.DataFrame, feature_names: list[str]).
    """
    if cfg.FEATURE_MODE == "same_row":
        cols = [
            "Operating time (min)",
            "Total Downtime (min)",
            "Production (pcs)",
            "Defects (pcs)",
            "Cycle time – Min CT (min/pc)",
        ]
        X = df[cols].copy()

    elif cfg.FEATURE_MODE == "history_only":
        X = _history_features(df)

    else:
        raise ValueError(f"Unknown FEATURE_MODE: {cfg.FEATURE_MODE}")

    print(f"[features] Mode = {cfg.FEATURE_MODE}, shape = {X.shape}")
    return X, list(X.columns)


def _history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per machine+model, lagged A/P/Q/label of the previous run, rolling
    mean of the previous 5 runs (shifted so the current row is never
    included), day of week, month, machine and model as codes.
    """
    df = df.copy()
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"]       = df["Date"].dt.month

    # encode machine & model as integer codes
    df["machine_code"] = df["machine"].astype("category").cat.codes
    df["model_code"]   = df["Model"].astype("category").cat.codes

    group_cols = ["machine", "Model"]
    for col in ["Availability (A)", "Performance (P)", "Quality (Q)"]:
        lag_name = f"lag1_{col}"
        roll_name = f"roll5_{col}"
        df[lag_name]  = df.groupby(group_cols)[col].shift(1)
        df[roll_name] = (
            df.groupby(group_cols)[col]
              .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    keep = [c for c in df.columns if c.startswith(("lag1_", "roll5_"))] + [
        "day_of_week", "month", "machine_code", "model_code",
    ]
    return df[keep].fillna(0)


# ────────────────────── feature selector transformer ───────
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Drops near-zero variance columns (threshold < nzv_threshold) and
    one of each pair with |r| > corr_threshold (keeps the one more
    correlated with y).

    Fitted on training data only; applied identically to test data.
    """
    def __init__(self, nzv_threshold: float = 0.01, corr_threshold: float = 0.90):
        self.nzv_threshold = nzv_threshold
        self.corr_threshold = corr_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        # 1. near-zero variance
        variances = X.var()
        self.nzv_drop_ = list(variances[variances < self.nzv_threshold].index)

        X_kept = X.drop(columns=self.nzv_drop_, errors="ignore")

        # 2. high correlation
        self.corr_drop_ = []
        if y is not None and len(X_kept.columns) > 1:
            corr_matrix = X_kept.corr().abs()
            target_corr = X_kept.corrwith(pd.Series(y, index=X_kept.index)).abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
            )
            for col in upper.columns:
                high_corr_pairs = upper.index[upper[col] > self.corr_threshold].tolist()
                for pair_col in high_corr_pairs:
                    # drop the one less correlated with target
                    if pair_col not in self.corr_drop_:
                        drop_col = (
                            pair_col if target_corr.get(col, 0) >= target_corr.get(pair_col, 0)
                            else col
                        )
                        if drop_col not in self.corr_drop_:
                            self.corr_drop_.append(drop_col)

        self.drop_cols_ = list(set(self.nzv_drop_ + self.corr_drop_))
        self.keep_cols_ = [c for c in X.columns if c not in self.drop_cols_]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        # use positional indexing for safety (SMOTE returns numpy arrays)
        if isinstance(X.columns[0], str):
            return X[[c for c in X.columns if c not in self.drop_cols_]].values
        else:
            # numpy array input – use column positions
            drop_idx = [i for i, c in enumerate(range(X.shape[1]))
                        if c in [int(d) for d in self.drop_cols_ if isinstance(d, (int, np.integer))]]
            keep_idx = [i for i in range(X.shape[1]) if i not in drop_idx]
            if not keep_idx:
                keep_idx = list(range(X.shape[1]))
            return X.iloc[:, keep_idx].values if hasattr(X, "iloc") else X[:, keep_idx]

    def get_feature_names_out(self, input_features=None):
        return self.keep_cols_
