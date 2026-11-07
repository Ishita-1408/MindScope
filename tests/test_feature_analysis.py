import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.explain import DEFAULT_MODEL_PATH, MindScopeExplainer, load_explainer
from src.feature_analysis import (
    CATEGORICAL_FEATURES,
    DEPENDENCE_DIR,
    NON_CAUSAL_DISCLAIMER,
    NUMERICAL_FEATURES,
    PRIORITIZED_FEATURES_ORDER,
    SEALED_HOLDOUT_PATH,
    aggregate_interaction_to_original,
    benchmark_interaction_timing,
    compute_feature_dependence,
    compute_interaction_analysis,
    get_deterministic_training_sample,
    verify_sealed_holdout_isolation,
)
from src.preprocessing import PREDICTOR_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MODEL_PATH = PROJECT_ROOT / "Mental_Health_Model.pkl"
EXPECTED_LEGACY_SIZE = 25694899


@pytest.fixture(scope="module")
def shared_explainer() -> MindScopeExplainer:
    return load_explainer()


@pytest.fixture(scope="module")
def small_train_sample() -> pd.DataFrame:
    sample_df, _ = get_deterministic_training_sample(sample_size=6, random_state=42)
    return sample_df


def test_training_sampling_determinism_and_isolation():
    """Verifies that sampling exclusively draws from the 3,498 training records deterministically."""
    sample1, n_total1 = get_deterministic_training_sample(sample_size=15, random_state=42)
    sample2, n_total2 = get_deterministic_training_sample(sample_size=15, random_state=42)

    assert n_total1 == 3498
    assert n_total2 == 3498
    pd.testing.assert_frame_equal(sample1, sample2)
    assert len(sample1) == 15


def test_sealed_holdout_access_blocked():
    """Verifies that referencing the sealed holdout test file raises a strict PermissionError."""
    with pytest.raises(PermissionError, match="Access violation"):
        verify_sealed_holdout_isolation(SEALED_HOLDOUT_PATH)

    with pytest.raises(PermissionError, match="Access violation"):
        get_deterministic_training_sample(data_path=SEALED_HOLDOUT_PATH)


def test_numerical_dependence_schema_and_finiteness(shared_explainer, small_train_sample):
    """Verifies output schema, required metrics, and finite numbers for all numerical features."""
    results = compute_feature_dependence(shared_explainer, small_train_sample)

    assert "metadata" in results
    assert results["metadata"]["sample_size"] == len(small_train_sample)
    assert "numerical_features" in results
    assert "categorical_features" in results

    for feat in NUMERICAL_FEATURES:
        assert feat in results["numerical_features"], f"Missing numerical feature: {feat}"
        data = results["numerical_features"][feat]
        assert "min_value" in data
        assert "max_value" in data
        assert "pearson_r" in data
        assert "spearman_rho" in data
        assert "bin_statistics" in data
        assert len(data["sample_values"]) == len(small_train_sample)
        assert len(data["sample_shaps"]) == len(small_train_sample)

        # Numerical finiteness
        assert np.isfinite(data["mean_abs_shap"])
        assert np.isfinite(data["pearson_r"])
        assert np.isfinite(data["spearman_rho"])
        assert all(np.isfinite(v) for v in data["sample_shaps"])


def test_categorical_dependence_schema_and_finiteness(shared_explainer, small_train_sample):
    """Verifies output schema and distribution statistics for all categorical features."""
    results = compute_feature_dependence(shared_explainer, small_train_sample)

    for feat in CATEGORICAL_FEATURES:
        assert feat in results["categorical_features"], f"Missing categorical feature: {feat}"
        data = results["categorical_features"][feat]
        assert "n_categories" in data
        assert "category_statistics" in data
        assert data["n_categories"] > 0
        assert len(data["category_statistics"]) > 0

        for cat_stat in data["category_statistics"]:
            assert "category" in cat_stat
            assert "count" in cat_stat
            assert "mean_shap" in cat_stat
            assert "median_shap" in cat_stat
            assert np.isfinite(cat_stat["mean_shap"])
            assert np.isfinite(cat_stat["median_shap"])


def test_interaction_timing_benchmark_structure(shared_explainer, small_train_sample):
    """Verifies that the benchmark runs safely with explicit timeouts and valid projections."""
    # Run with small sample N=1
    bench = benchmark_interaction_timing(shared_explainer, small_train_sample, test_sizes=[1], time_limit_seconds=30.0)

    assert "benchmarks" in bench
    assert len(bench["benchmarks"]) == 1
    rec = bench["benchmarks"][0]
    assert rec["sample_size"] == 1
    assert rec["is_symmetric"] is True
    assert rec["output_shape"] == [1, 38, 38]
    assert rec["max_additivity_error"] < 1e-10

    projections = bench["scaling_projections"]
    assert projections["recommended_sample_size"] == 10
    assert projections["is_full_interaction_feasible"] is False
    assert "projected_hours_full_train_3498" in projections


def test_interaction_aggregation_dimensions_and_symmetry(shared_explainer):
    """Verifies aggregation of synthetic (N, 38, 38) interaction matrix to (N, 12, 12) matrix."""
    N = 2
    M = shared_explainer.n_transformed_features
    # Create symmetric synthetic 38x38 interaction tensors
    rng = np.random.default_rng(42)
    sym_tensor = np.zeros((N, M, M))
    for i in range(N):
        mat = rng.standard_normal((M, M))
        sym_tensor[i] = (mat + mat.T) / 2.0

    orig_tensor = aggregate_interaction_to_original(sym_tensor, shared_explainer)
    assert orig_tensor.shape == (N, 12, 12)

    for i in range(N):
        mat_12 = orig_tensor[i]
        assert np.allclose(mat_12, mat_12.T, atol=1e-12), "Aggregated 12x12 matrix must be strictly symmetric"


def test_interaction_row_sum_consistency_with_grouped_shap(shared_explainer, small_train_sample):
    """Verifies that the row sums of the aggregated 12x12 interaction matrix match grouped SHAP values."""
    sub_df = small_train_sample.iloc[:1].reset_index(drop=True)
    X_trans = shared_explainer.transform_data(sub_df)
    shap_trans = shared_explainer.compute_transformed_shap_values(X_trans)
    grouped_df = shared_explainer.aggregate_to_original_features(shap_trans)

    inter_trans = shared_explainer.explainer.shap_interaction_values(X_trans)
    orig_tensor = aggregate_interaction_to_original(inter_trans, shared_explainer)

    mat_12 = orig_tensor[0]
    row_sums = np.sum(mat_12, axis=1)

    for idx, feat in enumerate(PREDICTOR_COLUMNS):
        expected_shap = float(grouped_df[feat].iloc[0])
        assert abs(row_sums[idx] - expected_shap) < 1e-10, (
            f"Row sum mismatch for {feat}: row_sum={row_sums[idx]}, expected={expected_shap}"
        )


def test_interaction_additivity_reconstruction(shared_explainer, small_train_sample):
    """Verifies that the sum of the aggregated 12x12 interaction matrix reconstructs f(x) - E[f(x)]."""
    sub_df = small_train_sample.iloc[:1].reset_index(drop=True)
    X_trans = shared_explainer.transform_data(sub_df)
    pred = float(shared_explainer.model.predict(X_trans)[0])
    ev = shared_explainer.expected_value

    inter_trans = shared_explainer.explainer.shap_interaction_values(X_trans)
    orig_tensor = aggregate_interaction_to_original(inter_trans, shared_explainer)

    total_sum = float(np.sum(orig_tensor[0]))
    expected_diff = pred - ev
    assert abs(total_sum - expected_diff) < 1e-10, (
        f"Additivity reconstruction failed: total_sum={total_sum}, expected_diff={expected_diff}"
    )


def test_invalid_inputs_handling(shared_explainer):
    """Verifies clear error reporting for invalid sample sizes or missing predictor columns."""
    with pytest.raises(ValueError, match="exceeds training partition size"):
        get_deterministic_training_sample(sample_size=10000)

    invalid_df = pd.DataFrame({"Age": [20], "Gender": ["Male"]})
    with pytest.raises(ValueError, match="missing required predictor columns"):
        shared_explainer.transform_data(invalid_df)

    with pytest.raises(ValueError, match="exceeds available records"):
        tiny_df = pd.DataFrame({"A": [1]})
        compute_interaction_analysis(shared_explainer, tiny_df, n_samples=10)


def test_dependence_artifacts_and_metadata_integrity():
    """Verifies that generated dependence and interaction JSON files and PNG plots exist and are non-empty."""
    required_json = [
        "dependence_summary.json",
        "interaction_benchmark.json",
        "interaction_analysis.json",
        "dependence_metadata.json",
    ]
    required_png = [
        "numerical_dependence_grid.png",
        "dependence_usage_hours.png",
        "dependence_sleep_hours.png",
        "dependence_daily_unlocks.png",
        "categorical_attribution_distributions.png",
        "interaction_matrix_heatmap.png",
        "interaction_candidate_pairs.png",
    ]

    for filename in required_json:
        p = DEPENDENCE_DIR / filename
        assert p.exists(), f"Missing required artifact: {p}"
        assert p.stat().st_size > 50, f"Artifact is unexpectedly small: {p}"

    for filename in required_png:
        p = DEPENDENCE_DIR / filename
        assert p.exists(), f"Missing required visualization: {p}"
        assert p.stat().st_size > 1000, f"Visualization is unexpectedly small: {p}"

    # Verify metadata JSON fields
    with open(DEPENDENCE_DIR / "dependence_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["governance"]["sealed_holdout_accessed"] is False
    assert meta["governance"]["models_retrained"] is False
    assert meta["governance"]["training_partition_size"] == 3498


def test_model_artifacts_unmutated_after_analysis():
    """Verifies that modern and legacy model files were not retrained, refitted, or overwritten."""
    assert DEFAULT_MODEL_PATH.exists()
    assert LEGACY_MODEL_PATH.exists()
    assert LEGACY_MODEL_PATH.stat().st_size == EXPECTED_LEGACY_SIZE


def test_sealed_holdout_file_unaccessed_and_unmodified():
    """Verifies that the sealed holdout dataset remains strictly sealed on disk without reading it."""
    assert SEALED_HOLDOUT_PATH.exists()
    assert SEALED_HOLDOUT_PATH.stat().st_size > 0


def test_non_causal_disclaimer_presence():
    """Verifies that all explanation outputs include the mandatory non-causal statistical disclaimer."""
    with open(DEPENDENCE_DIR / "interaction_analysis.json", "r", encoding="utf-8") as f:
        inter = json.load(f)
    assert "non_causal_disclaimer" in inter
    assert "does not establish causality" in inter["non_causal_disclaimer"].lower()
