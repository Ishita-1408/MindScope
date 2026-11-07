import json
from pathlib import Path
from typing import Any, Dict
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

import main
from src.explain import MindScopeExplainer, load_explainer
from src.preprocessing import PREDICTOR_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MODEL_FILE = PROJECT_ROOT / "Mental_Health_Model.pkl"
SEALED_HOLDOUT_FILE = PROJECT_ROOT / "artifacts" / "sealed_holdout_test.csv"


@pytest.fixture(scope="module")
def explainer():
    """Load production MindScopeExplainer once for the test module."""
    return load_explainer()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(main.app)


@pytest.fixture
def synthetic_record() -> Dict[str, Any]:
    """Clearly identified synthetic student profile for demonstration and testing."""
    return {
        "Age": 21,
        "Gender": "Female",
        "Country": "United States",
        "Academic_Level": "Undergraduate",
        "Most_Used_Platform": "Instagram",
        "Purpose_Of_Use": "Entertainment",
        "Avg_Daily_Usage_Hours": 5.5,
        "Daily_Unlocks": 85,
        "Study_Hours": 4.0,
        "Physical_Activity_Hours": 1.0,
        "Sleep_Hours_Per_Night": 6.5,
        "Stress_Level": "Medium",
    }


@pytest.fixture
def api_payload() -> Dict[str, Any]:
    """Valid API payload format for FastAPI endpoints."""
    return {
        "age": 21,
        "gender": "Female",
        "country": "United States",
        "academic_level": "Undergraduate",
        "most_used_platform": "Instagram",
        "purpose_of_use": "Entertainment",
        "avg_daily_usage_hours": 5.5,
        "daily_unlocks": 85,
        "study_hours": 4.0,
        "physical_activity_hours": 1.0,
        "sleep_hours_per_night": 6.5,
        "stress_level": "Medium",
    }


# =============================================================================
# 1. Output Schema and Key Validation
# =============================================================================
def test_local_explanation_schema_and_keys(explainer, synthetic_record):
    """Verify explain_prediction returns all required local XAI components."""
    res = explainer.explain_prediction(synthetic_record)

    expected_top_keys = {
        "prediction",
        "base_value",
        "net_attribution",
        "feature_attributions",
        "waterfall_steps",
        "transformed_attributions",
        "additivity_verified",
        "additivity_error",
        "disclaimer",
    }
    assert expected_top_keys.issubset(set(res.keys()))

    # Verify exactly 12 original-feature attribution groups
    assert len(res["feature_attributions"]) == 12

    expected_item_keys = {
        "feature",
        "display_name",
        "value",
        "unit",
        "formatted_value",
        "shap_value",
        "direction",
        "abs_impact",
        "rank",
        "step_start",
        "cumulative_score",
    }
    for item in res["feature_attributions"]:
        assert expected_item_keys.issubset(set(item.keys()))


# =============================================================================
# 2. Prediction Consistency & Additive Reconstruction
# =============================================================================
def test_prediction_consistency_with_pipeline(explainer, synthetic_record):
    """Verify local explanation prediction matches pipeline.predict to machine precision."""
    df = pd.DataFrame([synthetic_record])
    direct_pred = float(explainer.pipeline.predict(df[PREDICTOR_COLUMNS])[0])

    res = explainer.explain_prediction(synthetic_record)
    assert np.isclose(res["prediction"], direct_pred, atol=1e-12)


def test_baseline_plus_contributions_reconstructs_prediction(explainer, synthetic_record):
    """Verify efficiency axiom: base_value + sum(contributions) == prediction within 1e-5."""
    res = explainer.explain_prediction(synthetic_record, atol=1e-5)

    assert res["additivity_verified"] is True
    assert res["additivity_error"] < 1e-5

    # Manual mathematical re-verification
    base_val = res["base_value"]
    sum_contributions = sum(item["shap_value"] for item in res["feature_attributions"])
    reconstructed = base_val + sum_contributions

    assert abs(reconstructed - res["prediction"]) < 1e-5

    # Net attribution check
    assert np.isclose(res["net_attribution"], res["prediction"] - base_val, atol=1e-12)


# =============================================================================
# 3. Original Feature Groups & Waterfall Structure
# =============================================================================
def test_exactly_12_original_feature_attribution_groups(explainer, synthetic_record):
    """Verify all 12 original features are present, ranked 1..12 descending by abs_impact."""
    res = explainer.explain_prediction(synthetic_record)
    features = [item["feature"] for item in res["feature_attributions"]]

    assert set(features) == set(PREDICTOR_COLUMNS)
    assert len(features) == 12

    ranks = [item["rank"] for item in res["feature_attributions"]]
    assert ranks == list(range(1, 13))

    impacts = [item["abs_impact"] for item in res["feature_attributions"]]
    assert impacts == sorted(impacts, reverse=True)


def test_waterfall_steps_sequence(explainer, synthetic_record):
    """
    Verify waterfall sequence starts at baseline E[y], step-wise accumulates
    each feature's delta, and terminates at final model prediction.
    """
    res = explainer.explain_prediction(synthetic_record)
    steps = res["waterfall_steps"]

    # Exactly 14 steps: 1 baseline + 12 attributions + 1 prediction
    assert len(steps) == 14

    # Step 0: Baseline
    assert steps[0]["step_type"] == "baseline"
    assert np.isclose(steps[0]["cumulative_score"], res["base_value"], atol=1e-4)

    # Intermediate Steps 1..12: Attributions
    running = res["base_value"]
    for i, step in enumerate(steps[1:13]):
        assert step["step_type"] == "attribution"
        running += step["delta"]
        assert np.isclose(step["cumulative_score"], running, atol=1e-4)

    # Step 13: Prediction
    assert steps[-1]["step_type"] == "prediction"
    assert np.isclose(steps[-1]["cumulative_score"], res["prediction"], atol=1e-4)


# =============================================================================
# 4. Finiteness, Directionality & Determinism
# =============================================================================
def test_finite_contribution_values(explainer, synthetic_record):
    """Verify all numerical values are finite."""
    res = explainer.explain_prediction(synthetic_record)

    assert np.isfinite(res["prediction"])
    assert np.isfinite(res["base_value"])
    assert np.isfinite(res["net_attribution"])

    for item in res["feature_attributions"]:
        assert np.isfinite(item["shap_value"])
        assert np.isfinite(item["abs_impact"])
        assert np.isfinite(item["step_start"])
        assert np.isfinite(item["cumulative_score"])


def test_positive_and_negative_contribution_handling(explainer, synthetic_record):
    """Verify direction label cleanly distinguishes positive and negative contributions."""
    res = explainer.explain_prediction(synthetic_record)

    for item in res["feature_attributions"]:
        shap = item["shap_value"]
        direction = item["direction"]
        if shap >= 0.0:
            assert direction == "positive"
        else:
            assert direction == "negative"


def test_deterministic_output_for_identical_inputs(explainer, synthetic_record):
    """Verify repeat calls produce bitwise identical output."""
    res1 = explainer.explain_prediction(synthetic_record)
    res2 = explainer.explain_prediction(synthetic_record)

    assert res1["prediction"] == res2["prediction"]
    assert res1["base_value"] == res2["base_value"]
    assert res1["net_attribution"] == res2["net_attribution"]

    for item1, item2 in zip(res1["feature_attributions"], res2["feature_attributions"]):
        assert item1["feature"] == item2["feature"]
        assert item1["shap_value"] == item2["shap_value"]
        assert item1["cumulative_score"] == item2["cumulative_score"]


def test_invalid_input_handling_in_explainer(explainer):
    """Verify defensive exceptions on corrupted inputs."""
    # 1. Multi-row DataFrame
    multi_row = pd.DataFrame([{"Age": 21}, {"Age": 22}])
    with pytest.raises(ValueError, match="expects a single record"):
        explainer.explain_prediction(multi_row)

    # 2. Invalid data type
    with pytest.raises(TypeError, match="record must be a dict or 1-row DataFrame"):
        explainer.explain_prediction(["invalid_list"])

    # 3. Missing required columns
    with pytest.raises(ValueError, match="missing required predictor columns"):
        explainer.explain_prediction({"Age": 21, "Gender": "Female"})


# =============================================================================
# 5. FastAPI /explain Endpoint Tests
# =============================================================================
def test_api_explain_endpoint_valid_payload(client, api_payload):
    """Verify POST /explain returns 200 and valid LocalExplanationResponse."""
    response = client.post("/explain", json=api_payload)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_mental_health_score" in data
    assert "expected_baseline_score" in data
    assert "net_attribution" in data
    assert "feature_attributions" in data
    assert "waterfall_steps" in data
    assert "additivity_verified" in data
    assert "disclaimer" in data

    assert data["additivity_verified"] is True
    assert len(data["feature_attributions"]) == 12
    assert len(data["waterfall_steps"]) == 14
    assert "shifts this model's prediction relative to its expected baseline" in data["disclaimer"]
    assert "does not indicate better or worse mental health" in data["disclaimer"]


def test_api_explain_endpoint_rejections(client, api_payload):
    """Verify POST /explain enforces the exact observational/engineering bounds."""
    # Age below 18 rejected
    bad_age = dict(api_payload)
    bad_age["age"] = 17
    assert client.post("/explain", json=bad_age).status_code == 422

    # Negative physical activity rejected
    bad_activity = dict(api_payload)
    bad_activity["physical_activity_hours"] = -0.5
    assert client.post("/explain", json=bad_activity).status_code == 422


def test_api_explain_service_unavailable(client, api_payload, monkeypatch):
    """Verify POST /explain returns HTTP 503 when explainer is unavailable."""
    monkeypatch.setattr(main, "explainer", None)
    response = client.post("/explain", json=api_payload)
    assert response.status_code == 503
    assert "Explainability engine is currently unavailable" in response.json()["detail"]


# =============================================================================
# 6. Legacy Model & Holdout Safety Invariants
# =============================================================================
def test_legacy_model_and_holdout_untouched():
    """Verify Mental_Health_Model.pkl retains exact size and holdout was not modified."""
    assert LEGACY_MODEL_FILE.exists()
    assert LEGACY_MODEL_FILE.stat().st_size == 25694899

