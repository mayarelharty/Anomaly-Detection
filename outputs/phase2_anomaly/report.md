# Phase 2 – Supervised Anomaly Detection & Feature Enrichment

## 1. Assumptions

- The dataset is the **corrected** version (no 10/11/25_Asano B_HF 230; HF 230 Min CT ≈ 0.5385).
- The Summary sheet (header row 4, 0-indexed row 3) contains one row per Sec ID = date_machine_model.
- Machine identity (Asano A / Asano B) is encoded inside the Sec ID column.
- No pre-existing train/test split was found (no split column, no separate sheets); we create one.
- SPLIT_MODE = `chronological`, TEST_SIZE = 0.3
- THRESHOLD_MODE = `train_percentile`
- FEATURE_MODE = `same_row`
- SEED = 42

## 2. Data Loading & Cleaning
- Shape after cleaning: (1921, 15)
- Date range: 2024-01-01 to 2026-06-10
- Unique dates: 546
- Machines: {'Asano A': 1004, 'Asano B': 917}
- Models: ['AD 23', 'AD 27', 'DL 58', 'DL 58GV', 'DL 60', 'DL 63', 'DL 69', 'DL 73', 'FL 58', 'FL 60', 'FL 73', 'FL E63', 'FL E69', 'HF 230', 'HF 230A', 'HF 270', 'HF 270A']

## 3. Exploratory Data Analysis
- EDA figures saved to `outputs/phase2_anomaly/`

## 4. OEE Validation
Max absolute difference per recomputed column:
  - Availability (A): 5.55e-17
  - Performance (P): 2.22e-16
  - Quality (Q): 5.55e-17
  - OEE: 5.55e-17
- **All columns match** (< 1e-10). Using existing A, P, Q, OEE columns.

## 5. Label Distribution

============================================================
LABEL DISTRIBUTION  (threshold_mode = train_percentile)
Thresholds: A < 0.3266, P < 0.6532, Q < 0.8465
============================================================

  FULL: 1921 rows  |  anomaly = 462 (24.0%)  |  normal = 1459 (76.0%)

  TRAIN: 1337 rows  |  anomaly = 331 (24.8%)  |  normal = 1006 (75.2%)

  TEST: 584 rows  |  anomaly = 131 (22.4%)  |  normal = 453 (77.6%)

  Per-dimension anomaly rates (full dataset):
    A-anomaly: 199/1921 = 10.4%
    P-anomaly: 174/1921 = 9.1%
    Q-anomaly: 172/1921 = 9.0%

  Overlap patterns:
    Normal (none)  :  1459  (76.0%)
    Q only         :   113  (5.9%)
    P only         :   141  (7.3%)
    P+Q            :     9  (0.5%)
    A only         :   136  (7.1%)
    A+Q            :    39  (2.0%)
    A+P            :    13  (0.7%)
    A+P+Q          :    11  (0.6%)

### For comparison (benchmark mode):

============================================================
LABEL DISTRIBUTION  (threshold_mode = benchmark)
Thresholds: A < 0.9000, P < 0.9500, Q < 0.9990
============================================================

  FULL: 1921 rows  |  anomaly = 1920 (99.9%)  |  normal = 1 (0.1%)

  TRAIN: 1337 rows  |  anomaly = 1336 (99.9%)  |  normal = 1 (0.1%)

  TEST: 584 rows  |  anomaly = 584 (100.0%)  |  normal = 0 (0.0%)

  Per-dimension anomaly rates (full dataset):
    A-anomaly: 1887/1921 = 98.2%
    P-anomaly: 1606/1921 = 83.6%
    Q-anomaly: 1831/1921 = 95.3%

  Overlap patterns:
    Normal (none)  :     1  (0.1%)
    Q only         :     4  (0.2%)
    P+Q            :    29  (1.5%)
    A only         :    15  (0.8%)
    A+Q            :   295  (15.4%)
    A+P            :    74  (3.9%)
    A+P+Q          :  1503  (78.2%)

## 6. Label Gate Check
- Gate **PASSED** ✓
- Split dates: train ≤ 2025-06-22, test ≥ 2025-06-23

## 7. Feature Engineering
- Features (same_row): ['Operating time (min)', 'Total Downtime (min)', 'Production (pcs)', 'Defects (pcs)', 'Cycle time – Min CT (min/pc)']
- X_train: (1337, 5), X_test: (584, 5)

## 8. Model Training (GridSearchCV)
- Best params: `{'clf__max_depth': None, 'clf__n_estimators': 300}`
- Best CV F1: 0.6357

- Full CV results saved to `cv_results.csv`
- Model saved to `rf_anomaly_model.joblib`

## 9. Feature Importance
| Feature | Importance |
|---------|------------|
| Production (pcs) | 0.4317 |
| Operating time (min) | 0.1983 |
| Total Downtime (min) | 0.1920 |
| Defects (pcs) | 0.1781 |
- Saved `feature_importance.png`

## 10. Evaluation (Test Set)

| Metric | Value |
|--------|-------|
| f1 | 0.7075 |
| precision | 0.6380 |
| recall | 0.7939 |
| accuracy | 0.8527 |
| roc_auc | 0.9147 |
| brier | 0.1003 |
| pr_auc | 0.8288 |
| baseline_majority_f1 | 0.0000 |
| baseline_persistence_f1 | 0.4609 |

## 11. Anomaly Scores & Enriched Dataset
- Enriched dataset saved to `enriched_dataset.csv`

## 12. Leakage & Validity Notes

> **⚠ IMPORTANT (same_row mode):** The label `y_anom` is a deterministic function of the features (Operating time, Total Downtime, Production, Defects, Min CT → A, P, Q → label). The model is therefore recovering a known rule, not learning genuinely independent anomaly patterns. High test-set scores (F1, ROC-AUC) reflect successful **rule recovery**, not independent anomaly detection capability.
>
> This is by design for Phase 2 (validating that the pipeline can faithfully reproduce domain thresholds). For real-world anomaly scoring, use `FEATURE_MODE = 'history_only'` which uses only lagged features from previous runs and is leakage-safe.

- Anomaly_Score_OOF provides leakage-safe scores for Phase 3:
  train rows use out-of-fold probabilities; test rows use final model scores.

---
*Phase 2 completed in 16.0s.*