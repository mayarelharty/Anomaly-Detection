"""
features.py – Feature Engineering & Selection for Anomaly Risk Prediction.

Phase 2 uses FEATURE_MODE = "history_only":
The ML predicts next-shift anomaly risk strictly from historical indicators.
No same-row parameters (Operating time, Downtime, Production, Defects, A, P, Q, OEE)
are allowed in X.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import config as cfg

# Explicit list of forbidden columns (current-shift definitions/outcomes)
FORBIDDEN_COLUMNS = [
    "Operating time (min)",
    "Total Downtime (min)",
    "Planned Downtime (min)",
    "Unplanned Downtime (min)",
    "Production (pcs)",
    "Defects (pcs)",
    "Cycle time – Min CT (min/pc)",
    "Availability (A)",
    "Performance (P)",
    "Quality (Q)",
    "OEE",
    "Sec ID",
]


class ScaledFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector that:
    1. Computes variance on standardized/scaled features to avoid penalizing
       naturally bounded variables (e.g. Quality).
    2. Drops features with scaled variance < nzv_threshold (default 0.01).
    3. Identifies feature pairs with |r| > corr_threshold (default 0.90)
       and drops the feature with lower correlation to y.

    Fitted on training data only.
    """
    def __init__(self, nzv_threshold: float = 0.01, corr_threshold: float = 0.90):
        self.nzv_threshold = nzv_threshold
        self.corr_threshold = corr_threshold
        self.keep_columns_ = None
        self.drop_columns_ = []
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            X_df = X.copy()
        else:
            self.feature_names_in_ = [f"feat_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        # 1. Variance on scaled features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        scaled_vars = np.var(X_scaled, axis=0)

        nzv_drop = [col for col, v in zip(self.feature_names_in_, scaled_vars) if v < self.nzv_threshold]

        retained = [c for c in self.feature_names_in_ if c not in nzv_drop]
        corr_drop = []

        # 2. Correlation filter
        if len(retained) > 1 and y is not None:
            y_s = pd.Series(y, index=X_df.index) if not isinstance(y, pd.Series) else y
            corr_mat = X_df[retained].corr().abs()
            target_corr = X_df[retained].corrwith(y_s).abs()

            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            for col in upper.columns:
                high_pairs = upper.index[upper[col] > self.corr_threshold].tolist()
                for pair in high_pairs:
                    if pair not in corr_drop and col not in corr_drop:
                        drop_c = pair if target_corr.get(col, 0) >= target_corr.get(pair, 0) else col
                        corr_drop.append(drop_c)

        self.drop_columns_ = list(set(nzv_drop + corr_drop))
        self.keep_columns_ = [c for c in self.feature_names_in_ if c not in self.drop_columns_]
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            return X[self.keep_columns_].values
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
            return X_df[self.keep_columns_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_columns_)


RECENCY_CAP = 20  # runs_since_anom value when no anomaly seen in the group yet (and upper cap)


def _runs_since_anom(prev_anom: pd.Series) -> pd.Series:
    """Runs back to the most recent prior anomaly (1 = previous run), capped at RECENCY_CAP."""
    out, count = [], RECENCY_CAP
    for v in prev_anom:
        if pd.isna(v):
            out.append(np.nan)
            continue
        count = 1 if v == 1 else min(count + 1, RECENCY_CAP)
        out.append(count)
    return pd.Series(out, index=prev_anom.index)


def build_history_features(df: pd.DataFrame, train_idx, test_idx, y_anom: pd.Series,
                           global_fences: dict, group_fences: dict):
    """
    Construct strictly historical features per machine + model:
    - machine_code, model_code, day_of_week, month, days_since_prev
    - lag-1 of A, P, Q, y_anom
    - rolling-5 (shifted, mean of previous up to 5 runs) of A, P, Q, y_anom
    - fence margins: lag-1 and rolling-5 minimum of A, P, Q minus the group's training fence
    - trend / volatility: lag-1 minus rolling-5 mean, rolling-5 std of A, P, Q
    - recency: runs since last anomaly, anomaly count in previous 10 runs

    Drops rows with no previous run (first run of each machine+model).
    Performs hand verification on lag integrity and asserts no forbidden features.
    """
    df_feat = df.copy()
    df_feat["y_anom"] = y_anom.values

    # Calendar & code features
    df_feat["day_of_week"] = df_feat["Date"].dt.dayofweek
    df_feat["month"]       = df_feat["Date"].dt.month
    df_feat["machine_code"] = df_feat["machine"].astype("category").cat.codes
    df_feat["model_code"]   = df_feat["Model"].astype("category").cat.codes

    group_cols = ["machine", "Model"]
    # Scheduled date of the current run is known in advance; gap to the previous run is history
    df_feat["days_since_prev"] = df_feat.groupby(group_cols)["Date"].diff().dt.days
    metrics_to_lag = [
        ("Availability (A)", "Availability"),
        ("Performance (P)",  "Performance"),
        ("Quality (Q)",      "Quality"),
        ("y_anom",           "y_anom"),
    ]

    for raw_c, short_c in metrics_to_lag:
        # Shift 1: guarantees strictly earlier rows
        df_feat[f"lag1_{short_c}"] = df_feat.groupby(group_cols)[raw_c].shift(1)
        # Shift 1 then rolling 5
        df_feat[f"roll5_{short_c}"] = (
            df_feat.groupby(group_cols)[raw_c]
                   .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    # Fence margins, trend and volatility (fences come from the training period only)
    fence_keys = list(zip(df_feat["machine"], df_feat["Model"]))
    for raw_c, short_c in metrics_to_lag[:3]:
        fence = pd.Series(
            [group_fences.get(k, global_fences)[raw_c] if cfg.IQR_SCOPE == "machine_model" else global_fences[raw_c]
             for k in fence_keys],
            index=df_feat.index,
        )
        roll5_min = df_feat.groupby(group_cols)[raw_c].transform(lambda s: s.shift(1).rolling(5, min_periods=1).min())
        df_feat[f"margin1_{short_c}"]    = df_feat[f"lag1_{short_c}"] - fence
        df_feat[f"margin_min5_{short_c}"] = roll5_min - fence
        df_feat[f"trend_{short_c}"]      = df_feat[f"lag1_{short_c}"] - df_feat[f"roll5_{short_c}"]
        df_feat[f"std5_{short_c}"] = (
            df_feat.groupby(group_cols)[raw_c]
                   .transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())
                   .fillna(0.0)
        )

    # Recency of trouble
    prev_anom = df_feat.groupby(group_cols)["y_anom"].shift(1)
    df_feat["runs_since_anom"] = prev_anom.groupby([df_feat[c] for c in group_cols]).transform(_runs_since_anom)
    df_feat["anom_count10"] = (
        df_feat.groupby(group_cols)["y_anom"]
               .transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    )

    feature_cols = [
        "machine_code", "model_code", "day_of_week", "month",
        "lag1_Availability", "roll5_Availability",
        "lag1_Performance",  "roll5_Performance",
        "lag1_Quality",      "roll5_Quality",
        "lag1_y_anom",       "roll5_y_anom",
        "days_since_prev",
        "margin1_Availability", "margin_min5_Availability", "trend_Availability", "std5_Availability",
        "margin1_Performance",  "margin_min5_Performance",  "trend_Performance",  "std5_Performance",
        "margin1_Quality",      "margin_min5_Quality",      "trend_Quality",      "std5_Quality",
        "runs_since_anom", "anom_count10",
    ]

    # Check for forbidden columns
    for fc in FORBIDDEN_COLUMNS:
        assert fc not in feature_cols, f"CRITICAL: Forbidden column {fc} found in feature matrix!"

    # Identify rows with no previous run
    no_prev_mask = df_feat["lag1_Availability"].isna()
    n_drop_train = int((no_prev_mask & df_feat.index.isin(train_idx)).sum())
    n_drop_test  = int((no_prev_mask & df_feat.index.isin(test_idx)).sum())
    print(f"[features] Dropping runs with no previous history: Train = {n_drop_train}, Test = {n_drop_test}")

    # Valid rows
    valid_train_idx = [i for i in train_idx if not no_prev_mask.loc[i]]
    valid_test_idx  = [i for i in test_idx if not no_prev_mask.loc[i]]

    # Hand verification on sampled rows to verify lag strictness
    _verify_lag_strictness(df_feat, group_cols)

    X_train = df_feat.loc[valid_train_idx, feature_cols].reset_index(drop=True)
    y_train = df_feat.loc[valid_train_idx, "y_anom"].reset_index(drop=True)

    X_test  = df_feat.loc[valid_test_idx, feature_cols].reset_index(drop=True)
    y_test  = df_feat.loc[valid_test_idx, "y_anom"].reset_index(drop=True)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "valid_train_idx": valid_train_idx,
        "valid_test_idx": valid_test_idx,
        "feature_cols": feature_cols,
        "cat_indices": [0, 1, 2, 3],  # machine_code, model_code, day_of_week, month
        "n_drop_train": n_drop_train,
        "n_drop_test": n_drop_test,
        "full_feat_df": df_feat,
    }


def _verify_lag_strictness(df_feat: pd.DataFrame, group_cols):
    """
    Hand-verify sampled rows to ensure lag features match the immediately
    prior run in that (machine, model) group and strictly never the current row.
    """
    for (m, mod), grp in df_feat.groupby(group_cols):
        if len(grp) >= 3:
            # Check row 2 (0-indexed 1) vs row 1 (0-indexed 0)
            idx0 = grp.index[0]
            idx1 = grp.index[1]
            actual_a0 = grp.loc[idx0, "Availability (A)"]
            lagged_a1 = grp.loc[idx1, "lag1_Availability"]
            actual_a1 = grp.loc[idx1, "Availability (A)"]

            assert np.isclose(actual_a0, lagged_a1, equal_nan=True), \
                f"Leakage check failed: group {(m, mod)} lag1 {lagged_a1} != prev {actual_a0}"
            assert not np.isclose(lagged_a1, actual_a1) or np.isclose(actual_a0, actual_a1), \
                f"Suspicious equality between lag and current value in group {(m, mod)}"
    print("[features] Strict lag integrity check passed ✓ (lag-1 strictly matches prior run)")


def build_same_row_diagnostic(df: pd.DataFrame, train_idx, test_idx, y_anom: pd.Series):
    """
    Diagnostic-only same_row feature builder to document rule-recovery ceiling.
    MUST NEVER be used to produce Anomaly_Score.
    """
    cols = [
        "Operating time (min)",
        "Total Downtime (min)",
        "Production (pcs)",
        "Defects (pcs)",
        "Cycle time – Min CT (min/pc)",
    ]
    X_train = df.loc[train_idx, cols].reset_index(drop=True)
    y_train = y_anom.loc[train_idx].reset_index(drop=True)
    X_test  = df.loc[test_idx, cols].reset_index(drop=True)
    y_test  = y_anom.loc[test_idx].reset_index(drop=True)
    return X_train, y_train, X_test, y_test
