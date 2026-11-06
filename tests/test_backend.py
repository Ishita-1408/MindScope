import os
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def valid_payload():
    return {
        "age": 21,
        "gender": "Male",
        "country": "India",
        "academic_level": "Undergraduate",
        "most_used_platform": "Instagram",
        "purpose_of_use": "Entertainment",
        "avg_daily_usage_hours": 4.5,
        "daily_unlocks": 80,
        "study_hours": 5.0,
        "physical_activity_hours": 1.5,
        "sleep_hours_per_night": 7.0,
        "stress_level": "Medium"
    }


def test_root_endpoint(client):
    """Verify GET / returns structured platform metadata and operational status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "platform" in data
    assert "version" in data
    assert data["status"] == "operational"
    assert data["documentation"] == "/docs"
    assert data["health"] == "/health"


def test_health_endpoint_healthy(client):
    """Verify GET /health returns operational health and model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "Mental_Health_Model.pkl" in data["model_path"]


def test_predict_valid_input(client, valid_payload):
    """Verify POST /predict returns rounded numerical score for valid inputs."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_mental_health_score" in data
    score = data["predicted_mental_health_score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 10.0


def test_predict_age_below_18_rejected(client, valid_payload):
    """Verify that age < 18 is rejected with HTTP 422."""
    valid_payload["age"] = 17
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("age" in str(e.get("loc", [])) for e in errors)


def test_predict_age_above_24_rejected(client, valid_payload):
    """Verify that age > 24 is rejected with HTTP 422."""
    valid_payload["age"] = 25
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("age" in str(e.get("loc", [])) for e in errors)


def test_predict_invalid_categorical_gender(client, valid_payload):
    """Verify invalid gender category is rejected with HTTP 422."""
    valid_payload["gender"] = "Non-binary"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_invalid_categorical_stress_level(client, valid_payload):
    """Verify invalid stress level category is rejected with HTTP 422."""
    valid_payload["stress_level"] = "Extreme"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_negative_physical_activity_rejected(client, valid_payload):
    """Verify negative physical activity hours are rejected rather than clipped."""
    valid_payload["physical_activity_hours"] = -1.0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("physical_activity_hours" in str(e.get("loc", [])) for e in errors)


def test_predict_excessive_daily_unlocks_rejected(client, valid_payload):
    """Verify daily unlocks > 500 (engineering bound) is rejected with HTTP 422."""
    valid_payload["daily_unlocks"] = 600
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_insufficient_sleep_rejected(client, valid_payload):
    """Verify sleep < 2.0 hours is rejected with HTTP 422."""
    valid_payload["sleep_hours_per_night"] = 1.0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_missing_required_fields(client):
    """Verify missing required fields triggers HTTP 422."""
    response = client.post("/predict", json={"age": 20})
    assert response.status_code == 422


def test_model_loading_failure_handling(client, valid_payload, monkeypatch):
    """Verify that when model is None, /health reports degraded and /predict returns 503."""
    monkeypatch.setattr(main, "model", None)
    monkeypatch.setattr(main, "model_load_error", "Simulated file corruption error")

    # Health endpoint check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "degraded"
    assert health_data["model_loaded"] is False
    assert "Simulated file corruption error" in health_data["details"]

    # Prediction endpoint check
    predict_resp = client.post("/predict", json=valid_payload)
    assert predict_resp.status_code == 503
    assert "Predictive model is currently unavailable" in predict_resp.json()["detail"]


def test_different_working_directory_path_safety():
    """Verify that main.py path resolution works independently of process cwd."""
    original_cwd = os.getcwd()
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    try:
        os.chdir(docs_dir)
        # Verify BASE_DIR is still correctly anchored to repo root
        assert main.MODEL_PATH.exists()
        assert main.MODEL_PATH.name == "Mental_Health_Model.pkl"
    finally:
        os.chdir(original_cwd)

