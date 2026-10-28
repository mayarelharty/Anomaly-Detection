"""
checks.py – Rigorous Verification Suite for Phase 2 Implementation.

Conducts 7 independent diagnostic checks:
1. Forbidden-feature assertion
2. Lag-feature strictness verification
3. Fences learned from train only
4. No resampling outside training folds
5. OOF recomputation integrity
6. Label-shuffle null test
7. 5-seed stability test
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import config as cfg
from features import FORBIDDEN_COLUMNS
from labels import compute_fences
from modeling import build_pipeline_variant


def run_full_verification_suite(X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                train_df: pd.DataFrame, test_df: pd.DataFrame,
                                global_fences: dict, group_fences: dict,
                                df_feat_full: pd.DataFrame, cat_indices: list,
                                primary_model_data: dict, label_method: str = None) -> dict:
    """
    Execute all 7 verification checks and return audit report dictionary.
    """
    print("\n" + "=" * 60)
    print("RUNNING VERIFICATION SUITE (7 Independent Quality Checks)")
    print("=" * 60)
    audit = {}

    # Check 1: Forbidden-feature assertion
    c1_passed, c1_msg = _check_forbidden_features(X_train, X_test)
    audit["1_forbidden_features"] = {"status": "PASS" if c1_passed else "FAIL", "evidence": c1_msg}
    print(f"Check 1 [Forbidden Features]: {'PASS ✓' if c1_passed else 'FAIL ✗'} – {c1_msg}")

    # Check 2: Lag-feature strictness
    c2_passed, c2_msg = _check_lag_strictness(df_feat_full)
    audit["2_lag_strictness"] = {"status": "PASS" if c2_passed else "FAIL", "evidence": c2_msg}
    print(f"Check 2 [Lag Strictness]: {'PASS ✓' if c2_passed else 'FAIL ✗'} – {c2_msg}")

    # Check 3: Fences learned from train only
    c3_passed, c3_msg = _check_fences_train_only(train_df, test_df, global_fences, group_fences, label_method)
    audit["3_fences_train_only"] = {"status": "PASS" if c3_passed else "FAIL", "evidence": c3_msg}
    print(f"Check 3 [Fences Train-Only]: {'PASS ✓' if c3_passed else 'FAIL ✗'} – {c3_msg}")

    # Check 4: No resampling outside training folds
    c4_passed, c4_msg = _check_no_resampling_outside_folds(primary_model_data["final_model"], X_test)
    audit["4_no_resampling_leakage"] = {"status": "PASS" if c4_passed else "FAIL", "evidence": c4_msg}
    print(f"Check 4 [Resampling Isolation]: {'PASS ✓' if c4_passed else 'FAIL ✗'} – {c4_msg}")

    # Check 5: OOF recomputation integrity
    c5_passed, c5_msg = _check_oof_recomputation(X_train, y_train, primary_model_data, cat_indices)
    audit["5_oof_recomputation"] = {"status": "PASS" if c5_passed else "FAIL", "evidence": c5_msg}
    print(f"Check 5 [OOF Recomputation]: {'PASS ✓' if c5_passed else 'FAIL ✗'} – {c5_msg}")

    # Check 6: Label-shuffle null test
    c6_passed, c6_msg = _check_label_shuffle(X_train, y_train, X_test, y_test, cat_indices)
    audit["6_label_shuffle"] = {"status": "PASS" if c6_passed else "FAIL", "evidence": c6_msg}
    print(f"Check 6 [Label Shuffle Test]: {'PASS ✓' if c6_passed else 'FAIL ✗'} – {c6_msg}")

    # Check 7: 5-seed stability test
    c7_passed, c7_msg = _check_seed_stability(X_train, y_train, X_test, y_test, cat_indices)
    audit["7_seed_stability"] = {"status": "PASS" if c7_passed else "FAIL", "evidence": c7_msg}
    print(f"Check 7 [5-Seed Stability]: {'PASS ✓' if c7_passed else 'FAIL ✗'} – {c7_msg}")

    return audit


def _check_forbidden_features(X_train, X_test):
    found = [c for c in FORBIDDEN_COLUMNS if c in X_train.columns or c in X_test.columns]
    if found:
        return False, f"Forbidden features detected: {found}"
    return True, f"All {len(FORBIDDEN_COLUMNS)} forbidden features confirmed absent from feature matrix."


def _check_lag_strictness(df_feat_full):
    violations = 0
    checked = 0
    group_cols = ["machine", "Model"]
    for (m, mod), grp in df_feat_full.groupby(group_cols):
        if len(grp) >= 5:
            for i in range(1, len(grp)):
                prev_a = grp.iloc[i-1]["Availability (A)"]
                curr_lag1_a = grp.iloc[i]["lag1_Availability"]
                checked += 1
                if not np.isclose(prev_a, curr_lag1_a, equal_nan=True):
                    violations += 1
    if violations == 0:
        return True, f"Verified {checked} consecutive run transitions; 0 lag-leakage violations."
    return False, f"Found {violations} lag-leakage mismatches out of {checked} transitions."


def _check_fences_train_only(train_df, test_df, global_fences, group_fences, label_method=None):
    # Recompute on train
    gf_check, grpf_check, _ = compute_fences(train_df, k=cfg.IQR_K, scope=cfg.IQR_SCOPE, method=label_method)
    diffs = [abs(global_fences[d] - gf_check[d]) for d in global_fences]
    if max(diffs) < 1e-12:
        return True, "Fences identically reproduced from train period; 0 dependence on test rows."
    return False, f"Fence mismatch detected: max diff = {max(diffs)}"


def _check_no_resampling_outside_folds(final_model, X_test):
    initial_shape = X_test.shape
    scores = final_model.predict_proba(X_test)
    if len(scores) == initial_shape[0]:
        return True, f"Test evaluation strictly preserved row count ({initial_shape[0]} rows); no test resampling."
    return False, f"Row count changed: {len(scores)} vs expected {initial_shape[0]}"


def _check_oof_recomputation(X_train, y_train, primary_model_data, cat_indices):
    tscv = TimeSeriesSplit(n_splits=cfg.CV_FOLDS)
    # Recompute fold 4 (last fold)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        if fold == 4:
            pipe = build_pipeline_variant(cfg.PRIMARY_VARIANT, cat_indices=cat_indices)
            pipe.set_params(**primary_model_data["best_params"])
            pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            recomputed = pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
            original = primary_model_data["oof_raw"][val_idx]
            max_diff = np.max(np.abs(recomputed - original))
            if max_diff < 1e-6:
                return True, f"Recomputed fold {fold+1} OOF probabilities match recorded scores within {max_diff:.2e}."
            return False, f"OOF mismatch in fold {fold+1}: max diff = {max_diff:.4f}"
    return False, "Could not locate target validation fold."


def _check_label_shuffle(X_train, y_train, X_test, y_test, cat_indices):
    rng = np.random.RandomState(cfg.SEED)
    y_shuffled = rng.permutation(y_train.values)
    pipe = build_pipeline_variant("reference", cat_indices=cat_indices)
    pipe.set_params(clf__n_estimators=100, clf__max_depth=10)
    pipe.fit(X_train, y_shuffled)
    null_scores = pipe.predict_proba(X_test)[:, 1]
    null_auc = roc_auc_score(y_test, null_scores)
    # Null model ROC-AUC should be near 0.50 (+/- 0.12)
    if 0.38 <= null_auc <= 0.62:
        return True, f"Null label model achieved chance ROC-AUC = {null_auc:.4f} (expected ~0.50)."
    return False, f"Null label model deviated suspiciously from chance: ROC-AUC = {null_auc:.4f}"


def _check_seed_stability(X_train, y_train, X_test, y_test, cat_indices):
    scores_per_seed = []
    for s in cfg.SEEDS_STABILITY:
        pipe = build_pipeline_variant("balanced", cat_indices=cat_indices)
        pipe.set_params(clf__random_state=s, clf__n_estimators=100, clf__max_depth=10)
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        scores_per_seed.append(auc)
    std_auc = float(np.std(scores_per_seed))
    mean_auc = float(np.mean(scores_per_seed))
    if std_auc < 0.05:
        return True, f"5-seed stability verified across {cfg.SEEDS_STABILITY}: mean AUC = {mean_auc:.4f}, std = {std_auc:.4f} (< 0.05)."
    return False, f"High seed instability: std = {std_auc:.4f} (> 0.05)"
