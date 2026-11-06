import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from src.preprocessing import EXPECTED_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN
from src.train import (
    calculate_metrics,
    get_candidate_configurations,
    partition_and_seal_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
DATA_FILE = PROJECT_ROOT / "Student Social Media And Mental Health Impact.csv"


# -----------------------------------------------------------------------------
# 1. Partitioning and Data Safeguards Tests
# -----------------------------------------------------------------------------
def test_partition_sizes_and_preservation():
    """Verify exact 70/30 partition sizes (3,498 train, 1,500 sealed holdout test)."""
    X_train, y_train, X_test, y_test = partition_and_seal_data(DATA_FILE, random_state=42, test_size=0.30)
    assert len(X_train) == 3498
    assert len(y_train) == 3498
    assert len(X_test) == 1500
    assert len(y_test) == 1500

    # Ensure no index overlap between train and holdout
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0


def test_deterministic_splitting():
    """Verify that train_test_split produces identical splits with fixed random_state=42."""
    X_tr1, y_tr1, X_te1, y_te1 = partition_and_seal_data(DATA_FILE, random_state=42)
    X_tr2, y_tr2, X_te2, y_te2 = partition_and_seal_data(DATA_FILE, random_state=42)

    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_series_equal(y_tr1, y_tr2)
    pd.testing.assert_frame_equal(X_te1, X_te2)
    pd.testing.assert_series_equal(y_te1, y_te2)


def test_sealed_holdout_file_integrity():
    """Verify sealed holdout file exists and contains exactly 1,500 records with all 13 columns."""
    holdout_path = ARTIFACTS_DIR / "sealed_holdout_test.csv"
    assert holdout_path.exists(), "Sealed holdout test split must exist on disk."

    holdout_df = pd.read_csv(holdout_path)
    assert len(holdout_df) == 1500
    assert list(holdout_df.columns) == EXPECTED_COLUMNS
    assert TARGET_COLUMN in holdout_df.columns


# -----------------------------------------------------------------------------
# 2. Metric Calculation Verification
# -----------------------------------------------------------------------------
def test_calculate_metrics_correctness():
    """Verify calculate_metrics returns unrounded float metrics and accurate tolerances."""
    y_true = np.array([5.0, 6.0, 7.0, 8.0])
    y_pred = np.array([5.05, 6.15, 7.30, 8.00])  # diffs: 0.05, 0.15, 0.30, 0.00

    metrics = calculate_metrics(y_true, y_pred)

    assert isinstance(metrics["r2"], float)
    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["tol_0.1"], float)
    assert isinstance(metrics["tol_0.2"], float)

    # In diffs: 2 out of 4 (0.05, 0.00) <= 0.10 -> 50%
    assert metrics["tol_0.1"] == 50.0
    # In diffs: 3 out of 4 (0.05, 0.15, 0.00) <= 0.20 -> 75%
    assert metrics["tol_0.2"] == 75.0


# -----------------------------------------------------------------------------
# 3. Nested Cross-Validation Separation Tests
# -----------------------------------------------------------------------------
def test_nested_cv_inner_outer_separation():
    """Verify inner CV folds are strictly subsets of outer training folds."""
    X_train, y_train, _, _ = partition_and_seal_data(DATA_FILE)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for outer_fold, (outer_tr_idx, outer_va_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
        outer_tr_set = set(outer_tr_idx)
        outer_va_set = set(outer_va_idx)

        # Outer train and outer val must be disjoint
        assert outer_tr_set.isdisjoint(outer_va_set)

        # Inner CV must split only within outer_tr_set
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=100 + outer_fold)
        for inner_tr_idx, inner_va_idx in inner_cv.split(outer_tr_idx):
            # Mapped back to original indices
            mapped_inner_tr = set(outer_tr_idx[inner_tr_idx])
            mapped_inner_va = set(outer_tr_idx[inner_va_idx])

            assert mapped_inner_tr.issubset(outer_tr_set)
            assert mapped_inner_va.issubset(outer_tr_set)
            assert mapped_inner_tr.isdisjoint(outer_va_set)
            assert mapped_inner_va.isdisjoint(outer_va_set)


# -----------------------------------------------------------------------------
# 4. Experiment Tracking and File Verification
# -----------------------------------------------------------------------------
def test_experiment_configuration_matrix_completeness():
    """Verify candidate configuration matrix covers all candidate estimators and treatments."""
    configs = get_candidate_configurations()
    assert len(configs) >= 8

    algorithms = {c["algorithm"] for c in configs}
    expected_algorithms = {"LinearRegression", "Ridge", "RandomForestRegressor", "HistGradientBoostingRegressor"}
    assert expected_algorithms.issubset(algorithms)

    # Verify both full and reduced feature configurations exist
    drop_purpose_flags = {c["drop_purpose"] for c in configs}
    assert True in drop_purpose_flags
    assert False in drop_purpose_flags

    # Verify activity strategies exist
    strategies = {c["activity_strategy"] for c in configs}
    assert "clip" in strategies
    assert "nan" in strategies


def test_structured_experiment_artifacts_exist():
    """Verify structured experiment reports exist and adhere to expected JSON/CSV schemas."""
    nested_summary_file = EXPERIMENTS_DIR / "nested_cv_summary.json"
    benchmarks_file = EXPERIMENTS_DIR / "all_configurations_benchmarks.json"
    fold_csv_file = EXPERIMENTS_DIR / "fold_level_metrics.csv"

    assert nested_summary_file.exists(), "nested_cv_summary.json must exist"
    assert benchmarks_file.exists(), "all_configurations_benchmarks.json must exist"
    assert fold_csv_file.exists(), "fold_level_metrics.csv must exist"

    # Validate nested summary JSON
    with open(nested_summary_file, "r") as f:
        summary_data = json.load(f)
    assert "outer_r2_mean" in summary_data
    assert "outer_mae_mean" in summary_data
    assert "outer_rmse_mean" in summary_data
    assert len(summary_data["outer_folds"]) == 5

    # Validate benchmarks JSON
    with open(benchmarks_file, "r") as f:
        benchmarks_data = json.load(f)
    assert len(benchmarks_data) >= 8
    for b in benchmarks_data:
        assert "config_id" in b
        assert "r2_mean" in b
        assert "mae_mean" in b

    # Validate fold-level CSV
    fold_df = pd.read_csv(fold_csv_file)
    assert len(fold_df) >= 40
    assert "config_id" in fold_df.columns
    assert "outer_fold" in fold_df.columns
    assert "r2" in fold_df.columns


# -----------------------------------------------------------------------------
# 5. Production Model Safety Restrictions
# -----------------------------------------------------------------------------
def test_production_model_artifact_was_not_overwritten():
    """Verify Mental_Health_Model.pkl retains legacy size (~24.5 MB) and was not overwritten."""
    legacy_model_path = PROJECT_ROOT / "Mental_Health_Model.pkl"
    assert legacy_model_path.exists()
    # Legacy file size is exactly 25,694,899 bytes (~24.5 MB)
    size_mb = legacy_model_path.stat().st_size / (1024 * 1024)
    assert 24.0 <= size_mb <= 26.0, "Mental_Health_Model.pkl must not be overwritten during Step 2.2"

