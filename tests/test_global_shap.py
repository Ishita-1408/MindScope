import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from sklearn.model_selection import train_test_split

from src.global_shap import (
    DATA_FILE,
    DISTRIBUTION_PNG,
    EXPLANATIONS_DIR,
    GLOBAL_BEESWARM_PNG,
    GLOBAL_IMPORTANCE_JSON,
    METADATA_JSON,
    SUMMARY_BAR_PNG,
    TRANSFORMED_BEESWARM_PNG,
    TRANSFORMED_IMPORTANCE_JSON,
    get_deterministic_training_sample,
)
from src.preprocessing import PREDICTOR_COLUMNS, load_and_validate_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MODEL_FILE = PROJECT_ROOT / "Mental_Health_Model.pkl"
PNG_MAGIC_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# =============================================================================
# A. Deterministic Sampling Tests
# =============================================================================
def test_deterministic_training_sampling():
    """
    Verifies that sampling uses training records only, returns exactly 500 records,
    is deterministic with random_state=42, and has zero intersection with holdout.
    Does not open or inspect 'artifacts/sealed_holdout_test.csv'.
    """
    # 1. Execute deterministic sample helper
    sample_df, total_train = get_deterministic_training_sample(sample_size=500, random_state=42)

    # 2. Assert partition constraints
    assert total_train == 3498
    assert len(sample_df) == 500
    assert list(sample_df.columns) == PREDICTOR_COLUMNS

    # 3. Repeat to assert strict determinism
    sample_df_repeat, _ = get_deterministic_training_sample(sample_size=500, random_state=42)
    pd.testing.assert_frame_equal(sample_df, sample_df_repeat)

    # 4. Verify membership in independent training split
    X, y = load_and_validate_data(DATA_FILE)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    assert len(X_train) == 3498
    assert len(X_test) == 1500

    # All sampled index positions must originate from X_train and be disjoint from X_test
    train_sampled = X_train.sample(n=500, random_state=42)
    pd.testing.assert_frame_equal(sample_df, train_sampled.reset_index(drop=True))
    assert set(train_sampled.index).isdisjoint(set(X_test.index))


# =============================================================================
# B. Artifact Schema and Integrity Tests
# =============================================================================
def test_global_importance_artifact_schema_and_values():
    """
    Validates global_importance.json:
    - Exactly 12 features corresponding to PREDICTOR_COLUMNS.
    - Expected schema fields and finite numerical values.
    - Descending rank ordering by mean_abs_shap.
    - Relative importance sums to 1.0 (100%) within floating tolerance.
    """
    assert GLOBAL_IMPORTANCE_JSON.exists(), f"Missing artifact: {GLOBAL_IMPORTANCE_JSON}"
    with open(GLOBAL_IMPORTANCE_JSON, "r") as f:
        records = json.load(f)

    assert isinstance(records, list)
    assert len(records) == 12

    expected_keys = {
        "feature",
        "mean_abs_shap",
        "mean_signed_shap",
        "median_abs_shap",
        "std_shap",
        "min_shap",
        "max_shap",
        "rank",
    }

    observed_features = []
    mean_abs_values = []

    for idx, item in enumerate(records, 1):
        assert set(item.keys()) == expected_keys
        feat = item["feature"]
        assert feat in PREDICTOR_COLUMNS
        observed_features.append(feat)

        assert item["rank"] == idx

        # Numerical finiteness assertions
        for num_key in ["mean_abs_shap", "mean_signed_shap", "median_abs_shap", "std_shap", "min_shap", "max_shap"]:
            val = item[num_key]
            assert isinstance(val, (int, float))
            assert np.isfinite(val), f"Non-finite value found for {feat}.{num_key}: {val}"

        assert item["mean_abs_shap"] >= 0.0
        assert item["median_abs_shap"] >= 0.0
        assert item["min_shap"] <= item["max_shap"]
        mean_abs_values.append(item["mean_abs_shap"])

    # Must contain all 12 unique predictor columns
    assert set(observed_features) == set(PREDICTOR_COLUMNS)

    # Strictly descending order by mean_abs_shap
    assert mean_abs_values == sorted(mean_abs_values, reverse=True)

    # Relative importance calculation normalization check
    total_importance = sum(mean_abs_values)
    assert total_importance > 0.0
    relative_shares = [val / total_importance for val in mean_abs_values]
    assert np.isclose(sum(relative_shares), 1.0, atol=1e-12)


def test_transformed_importance_artifact_schema():
    """
    Validates transformed_importance.json:
    - Exactly 38 features matching the pipeline's transformed dimensions.
    - Parent features belong to PREDICTOR_COLUMNS.
    - Ranks strictly descending 1 to 38.
    - Finite numbers.
    """
    assert TRANSFORMED_IMPORTANCE_JSON.exists(), f"Missing artifact: {TRANSFORMED_IMPORTANCE_JSON}"
    with open(TRANSFORMED_IMPORTANCE_JSON, "r") as f:
        records = json.load(f)

    assert isinstance(records, list)
    assert len(records) == 38

    expected_keys = {
        "transformed_feature",
        "parent_feature",
        "mean_abs_shap",
        "mean_signed_shap",
        "std_shap",
        "rank",
    }

    mean_abs_values = []
    for idx, item in enumerate(records, 1):
        assert set(item.keys()) == expected_keys
        assert item["rank"] == idx
        assert item["parent_feature"] in PREDICTOR_COLUMNS

        for num_key in ["mean_abs_shap", "mean_signed_shap", "std_shap"]:
            val = item[num_key]
            assert np.isfinite(val)
        assert item["mean_abs_shap"] >= 0.0
        mean_abs_values.append(item["mean_abs_shap"])

    assert mean_abs_values == sorted(mean_abs_values, reverse=True)


def test_global_analysis_metadata_schema():
    """
    Validates global_analysis_metadata.json governance and mathematical fields.
    """
    assert METADATA_JSON.exists(), f"Missing artifact: {METADATA_JSON}"
    with open(METADATA_JSON, "r") as f:
        meta = json.load(f)

    required_keys = {
        "analysis_identifier",
        "model_identifier",
        "model_artifact_path",
        "dataset_source",
        "training_partition_total_samples",
        "evaluation_sample_size",
        "sample_selection_method",
        "sealed_holdout_access",
        "base_value_expected_score",
        "feature_count_original",
        "feature_count_transformed",
        "shap_version",
        "python_version",
        "platform",
        "computation_time_seconds",
        "execution_timestamp_utc",
        "additivity_verification",
        "feature_mapping_summary",
    }
    assert required_keys.issubset(set(meta.keys()))

    assert meta["sealed_holdout_access"] is False
    assert meta["training_partition_total_samples"] == 3498
    assert meta["evaluation_sample_size"] == 500
    assert meta["feature_count_original"] == 12
    assert meta["feature_count_transformed"] == 38

    # Additivity proof in metadata
    additivity = meta["additivity_verification"]
    assert additivity["verified"] is True
    assert additivity["observed_max_discrepancy"] < 1e-5
    assert np.isfinite(additivity["observed_max_discrepancy"])

    # Feature mapping sums to 38
    mapping_summary = meta["feature_mapping_summary"]
    assert len(mapping_summary) == 12
    assert sum(mapping_summary.values()) == 38


# =============================================================================
# C. Visualization Integrity Tests
# =============================================================================
@pytest.mark.parametrize(
    "image_path",
    [
        SUMMARY_BAR_PNG,
        GLOBAL_BEESWARM_PNG,
        TRANSFORMED_BEESWARM_PNG,
        DISTRIBUTION_PNG,
    ],
)
def test_visualization_files_exist_and_valid(image_path: Path):
    """
    Verifies that the visualization files exist, are non-empty,
    contain a valid PNG 8-byte signature, and can be read by PIL.
    Does not regenerate or rewrite any plots.
    """
    assert image_path.exists(), f"Missing visualization: {image_path}"
    file_size = image_path.stat().st_size
    assert file_size > 50_000, f"Visualization {image_path.name} unexpectedly small ({file_size} bytes)"

    # Verify standard PNG magic signature
    with open(image_path, "rb") as f:
        magic = f.read(8)
    assert magic == PNG_MAGIC_SIGNATURE, f"File {image_path.name} does not match PNG magic signature"

    # Verify PIL can open and read header metadata
    with Image.open(image_path) as img:
        assert img.format == "PNG"
        width, height = img.size
        assert width >= 1000
        assert height >= 1000


# =============================================================================
# D. Legacy Artifact Safety Safeguard
# =============================================================================
def test_legacy_model_untouched():
    """Verify legacy Mental_Health_Model.pkl retains its exact original size (25,694,899 bytes)."""
    assert LEGACY_MODEL_FILE.exists()
    assert LEGACY_MODEL_FILE.stat().st_size == 25694899

