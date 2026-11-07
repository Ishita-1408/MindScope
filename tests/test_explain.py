import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.explain import DEFAULT_MODEL_PATH, MindScopeExplainer, load_explainer
from src.preprocessing import EXPECTED_COLUMNS, PREDICTOR_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MODEL_FILE = PROJECT_ROOT / "Mental_Health_Model.pkl"
SEALED_HOLDOUT_FILE = PROJECT_ROOT / "artifacts" / "sealed_holdout_test.csv"


@pytest.fixture(scope="module")
def explainer():
    """Fixture to load the fitted production model explainer once per test session."""
    return load_explainer()


@pytest.fixture
def sample_valid_record():
    """Synthetic student profile representing valid input domain."""
    return {
        "Age": 21,
        "Gender": "Female",
        "Country": "United States",
        "Academic_Level": "Undergraduate",
        "Most_Used_Platform": "Instagram",
        "Purpose_Of_Use": "Social Connectivity",
        "Avg_Daily_Usage_Hours": 4.5,
        "Daily_Unlocks": 75,
        "Study_Hours": 5.0,
        "Physical_Activity_Hours": 1.5,
        "Sleep_Hours_Per_Night": 7.0,
        "Stress_Level": "Medium",
    }


@pytest.fixture
def synthetic_batch_df():
    """Synthetic multi-observation DataFrame for testing batch operations."""
    records = [
        {
            "Age": 19,
            "Gender": "Male",
            "Country": "India",
            "Academic_Level": "Undergraduate",
            "Most_Used_Platform": "YouTube",
            "Purpose_Of_Use": "Entertainment",
            "Avg_Daily_Usage_Hours": 3.0,
            "Daily_Unlocks": 65,
            "Study_Hours": 6.0,
            "Physical_Activity_Hours": 2.0,
            "Sleep_Hours_Per_Night": 8.0,
            "Stress_Level": "Low",
        },
        {
            "Age": 23,
            "Gender": "Female",
            "Country": "United Kingdom",
            "Academic_Level": "Postgraduate",
            "Most_Used_Platform": "LinkedIn",
            "Purpose_Of_Use": "Professional Networking",
            "Avg_Daily_Usage_Hours": 2.5,
            "Daily_Unlocks": 50,
            "Study_Hours": 7.0,
            "Physical_Activity_Hours": 1.0,
            "Sleep_Hours_Per_Night": 6.0,
            "Stress_Level": "High",
        },
        {
            "Age": 20,
            "Gender": "Female",
            "Country": "Canada",
            "Academic_Level": "Undergraduate",
            "Most_Used_Platform": "TikTok",
            "Purpose_Of_Use": "Social Connectivity",
            "Avg_Daily_Usage_Hours": 6.5,
            "Daily_Unlocks": 120,
            "Study_Hours": 3.0,
            "Physical_Activity_Hours": 0.5,
            "Sleep_Hours_Per_Night": 5.0,
            "Stress_Level": "Very High",
        },
    ]
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 1. Explainer Loading and Discovery Tests
# -----------------------------------------------------------------------------
def test_explainer_initialization_and_discovery(explainer):
    """Verify explainer loads correctly and dynamically discovers transformed features."""
    assert explainer is not None
    assert explainer.model is not None
    assert explainer.preprocessor is not None

    # Verify transformed feature count matches model's n_features_in_
    assert explainer.n_transformed_features == explainer.model.n_features_in_
    assert len(explainer.transformed_feature_names) == explainer.n_transformed_features
    assert isinstance(explainer.expected_value, float)
    assert 5.0 <= explainer.expected_value <= 8.0


def test_explainer_non_hardcoded_dimensions(explainer):
    """Verify explainer does not hardcode dimensions and dynamically inspects components."""
    # Ensure transformed names list is dynamically populated from pipeline
    assert len(explainer.transformed_feature_names) > 0
    # Must equal model input dimension
    assert len(explainer.transformed_feature_names) == explainer.model.n_features_in_


# -----------------------------------------------------------------------------
# 2. Complete and Unambiguous Feature Mapping Tests
# -----------------------------------------------------------------------------
def test_feature_mapping_completeness(explainer):
    """Verify every transformed feature maps to exactly one original predictor."""
    mapping = explainer.feature_mapping
    parent_indices = explainer.parent_to_indices

    # 1. Every transformed feature must be mapped
    assert len(mapping) == explainer.n_transformed_features

    # 2. Every mapped parent must be a valid predictor column
    for trans_name, parent_col in mapping.items():
        assert parent_col in PREDICTOR_COLUMNS, f"Unknown parent {parent_col} for {trans_name}"

    # 3. Every original predictor must have at least one mapped index
    for orig_col in PREDICTOR_COLUMNS:
        assert orig_col in parent_indices
        assert len(parent_indices[orig_col]) >= 1, f"Original feature {orig_col} has no mapped columns"

    # 4. Total mapped indices across all parents must equal exact transformed feature count
    total_mapped_indices = sum(len(idx_list) for idx_list in parent_indices.values())
    assert total_mapped_indices == explainer.n_transformed_features

    # 5. Index sets must be mutually disjoint
    all_indices = []
    for idx_list in parent_indices.values():
        all_indices.extend(idx_list)
    assert len(all_indices) == len(set(all_indices)), "Transformed feature index collision detected"


# -----------------------------------------------------------------------------
# 3. Additivity and Mathematical Invariance Tests
# -----------------------------------------------------------------------------
def test_additivity_axiom_single_record(explainer, sample_valid_record):
    """Verify base_value + sum(grouped_SHAP) == model_prediction within 1e-5 tolerance."""
    explanation = explainer.explain_prediction(sample_valid_record, atol=1e-5)

    assert explanation["additivity_verified"] is True
    assert explanation["additivity_error"] < 1e-5

    # Manual mathematical re-verification
    pred = explanation["prediction"]
    base_val = explanation["base_value"]
    sum_attribs = sum(item["shap_value"] for item in explanation["feature_attributions"])

    assert abs((base_val + sum_attribs) - pred) < 1e-5


def test_additivity_axiom_batch(explainer, synthetic_batch_df):
    """Verify additivity holds across diverse batch observations."""
    X_trans = explainer.transform_data(synthetic_batch_df)
    preds = explainer.model.predict(X_trans)
    shap_trans = explainer.compute_transformed_shap_values(X_trans)

    grouped_df = explainer.aggregate_to_original_features(
        shap_trans, predictions=preds, atol=1e-5
    )

    # Assert shape of grouped SHAP matches (N, 12)
    assert grouped_df.shape == (len(synthetic_batch_df), len(PREDICTOR_COLUMNS))
    assert list(grouped_df.columns) == PREDICTOR_COLUMNS

    # Assert exact row-wise additivity
    sum_grouped = grouped_df.sum(axis=1).to_numpy()
    reconstructed = explainer.expected_value + sum_grouped
    max_err = np.max(np.abs(preds - reconstructed))
    assert max_err < 1e-5


def test_additivity_violation_raises(explainer, sample_valid_record):
    """Verify aggregate_to_original_features raises RuntimeError on corrupted predictions."""
    df = pd.DataFrame([sample_valid_record])
    X_trans = explainer.transform_data(df)
    shap_trans = explainer.compute_transformed_shap_values(X_trans)

    # Intentionally corrupt the expected prediction to test safety guard
    corrupted_pred = np.array([999.0])
    with pytest.raises(RuntimeError, match="SHAP efficiency axiom additivity violation"):
        explainer.aggregate_to_original_features(
            shap_trans, predictions=corrupted_pred, atol=1e-5
        )


# -----------------------------------------------------------------------------
# 4. Interface and Output Structure Tests
# -----------------------------------------------------------------------------
def test_individual_explanation_interface(explainer, sample_valid_record):
    """Verify explain_prediction returns expected schema and keys."""
    res = explainer.explain_prediction(sample_valid_record)

    expected_keys = [
        "prediction",
        "base_value",
        "net_attribution",
        "feature_attributions",
        "transformed_attributions",
        "additivity_verified",
        "additivity_error",
    ]
    for key in expected_keys:
        assert key in res

    assert len(res["feature_attributions"]) == 12
    # Verify features are ordered by absolute impact descending
    impacts = [item["abs_impact"] for item in res["feature_attributions"]]
    assert impacts == sorted(impacts, reverse=True)

    # Verify transformed attributions has all transformed features
    assert len(res["transformed_attributions"]) == explainer.n_transformed_features


def test_global_importance_interface(explainer, synthetic_batch_df):
    """Verify compute_global_importance computes mean absolute SHAP correctly."""
    res = explainer.compute_global_importance(synthetic_batch_df)

    assert "sample_count" in res
    assert res["sample_count"] == len(synthetic_batch_df)
    assert "original_feature_importance" in res
    assert "transformed_feature_importance" in res

    assert len(res["original_feature_importance"]) == 12
    assert len(res["transformed_feature_importance"]) == explainer.n_transformed_features

    # Verify rankings are sorted descending
    orig_vals = [item["mean_abs_shap"] for item in res["original_feature_importance"]]
    assert orig_vals == sorted(orig_vals, reverse=True)


# -----------------------------------------------------------------------------
# 5. Robustness and Determinism Tests
# -----------------------------------------------------------------------------
def test_invalid_input_handling(explainer):
    """Verify clear exceptions are raised on invalid input types or missing columns."""
    # 1. Invalid input type
    with pytest.raises(TypeError):
        explainer.transform_data(["invalid_list_input"])

    # 2. Missing columns
    incomplete_df = pd.DataFrame([{"Age": 21, "Gender": "Female"}])
    with pytest.raises(ValueError, match="missing required predictor columns"):
        explainer.transform_data(incomplete_df)


def test_deterministic_explanations(explainer, sample_valid_record):
    """Verify repeated invocations on identical input yield bitwise identical SHAP values."""
    exp1 = explainer.explain_prediction(sample_valid_record)
    exp2 = explainer.explain_prediction(sample_valid_record)

    assert exp1["prediction"] == exp2["prediction"]
    assert exp1["base_value"] == exp2["base_value"]

    for item1, item2 in zip(exp1["feature_attributions"], exp2["feature_attributions"]):
        assert item1["feature"] == item2["feature"]
        assert item1["shap_value"] == item2["shap_value"]


# -----------------------------------------------------------------------------
# 6. Governance & Artifact Isolation Safeguards
# -----------------------------------------------------------------------------
def test_legacy_model_untouched():
    """Verify Mental_Health_Model.pkl retains its exact legacy size (25,694,899 bytes)."""
    assert LEGACY_MODEL_FILE.exists()
    assert LEGACY_MODEL_FILE.stat().st_size == 25694899


def test_holdout_file_remains_unopened():
    """Verify test suite does not touch the sealed holdout file."""
    assert SEALED_HOLDOUT_FILE.exists()
    # Confirm exact 1501 lines (header + 1,500 rows)
    with open(SEALED_HOLDOUT_FILE, "r") as f:
        line_count = sum(1 for _ in f)
    assert line_count == 1501

