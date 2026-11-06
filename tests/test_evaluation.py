import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    calculate_metrics,
    perform_diagnostic_residual_analysis,
)
from src.preprocessing import EXPECTED_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
FINAL_MODEL_FILE = MODELS_DIR / "mindscope_rf_v1.joblib"
FINAL_MODEL_METADATA_FILE = MODELS_DIR / "mindscope_rf_v1_metadata.json"
EVALUATION_REPORT_JSON = EXPERIMENTS_DIR / "final_holdout_evaluation.json"
RESIDUAL_ANALYSIS_JSON = EXPERIMENTS_DIR / "residual_analysis.json"
HOLDOUT_PREDICTIONS_CSV = EXPERIMENTS_DIR / "holdout_predictions.csv"
LEGACY_MODEL_FILE = PROJECT_ROOT / "Mental_Health_Model.pkl"


# -----------------------------------------------------------------------------
# 1. Metric Calculation and Mathematical Safeguards Tests
# -----------------------------------------------------------------------------
def test_metric_calculations_on_synthetic_data():
    """Verify calculate_metrics accurately computes unrounded R², MAE, RMSE and tolerances."""
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


def test_undefined_decile_r2_handling():
    """Verify decile R² preserves None / handles zero variance without crashing."""
    # When all y_true values are identical (SS_tot == 0), decile R² is mathematically undefined
    synthetic_df = pd.DataFrame({
        TARGET_COLUMN: [5.0, 5.0, 5.0, 5.0],
        "predicted_score": [5.1, 4.9, 5.2, 4.8],
        "residual": [-0.1, 0.1, -0.2, 0.2],
        "abs_error": [0.1, 0.1, 0.2, 0.2],
    })

    diagnostics = perform_diagnostic_residual_analysis(synthetic_df)
    decile_entry = diagnostics["decile_breakdown"][0]

    assert decile_entry["local_decile_r2"] is None, "Undefined decile R² must be preserved as None"
    assert "decile_r2_note" in decile_entry


# -----------------------------------------------------------------------------
# 2. Diagnostic Generation Tests (Using Synthetic Predictions)
# -----------------------------------------------------------------------------
def test_diagnostic_generation_on_synthetic_predictions():
    """Verify perform_diagnostic_residual_analysis generates all required diagnostics."""
    np.random.seed(42)
    n = 200
    y_true = np.random.uniform(3.0, 9.0, n)
    residuals = np.random.normal(0, 0.5, n)
    y_pred = y_true - residuals

    synthetic_df = pd.DataFrame({
        TARGET_COLUMN: y_true,
        "predicted_score": y_pred,
        "residual": residuals,
        "abs_error": np.abs(residuals),
    })

    diag = perform_diagnostic_residual_analysis(synthetic_df)

    assert "summary_statistics" in diag
    assert "normality_diagnostics" in diag
    assert "heteroscedasticity" in diag
    assert "residual_histogram" in diag
    assert "qq_plot_samples" in diag
    assert "predicted_score_bin_spread" in diag
    assert "decile_breakdown" in diag
    assert "digit_heaping_check" in diag

    assert len(diag["residual_histogram"]) == 10
    assert len(diag["qq_plot_samples"]) >= 10
    assert len(diag["predicted_score_bin_spread"]) == 4


# -----------------------------------------------------------------------------
# 3. Model Loading, Prediction, and Metadata Verification
# -----------------------------------------------------------------------------
def test_saved_model_loading_and_inference():
    """Verify saved modern pipeline loads cleanly and predicts on synthetic inputs."""
    assert FINAL_MODEL_FILE.exists(), f"Model artifact must exist at {FINAL_MODEL_FILE}"
    model = joblib.load(FINAL_MODEL_FILE)

    synthetic_input = pd.DataFrame([{
        "Age": 22,
        "Gender": "Male",
        "Country": "United Kingdom",
        "Academic_Level": "Postgraduate",
        "Most_Used_Platform": "LinkedIn",
        "Purpose_Of_Use": "Professional Networking",
        "Avg_Daily_Usage_Hours": 3.5,
        "Daily_Unlocks": 50,
        "Study_Hours": 6.0,
        "Physical_Activity_Hours": 2.0,
        "Sleep_Hours_Per_Night": 6.5,
        "Stress_Level": "High",
    }])

    pred = model.predict(synthetic_input)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert 1.0 <= pred[0] <= 10.0


def test_saved_model_metadata_completeness():
    """Verify metadata JSON contains all required provenance and environment fields."""
    assert FINAL_MODEL_METADATA_FILE.exists(), "Metadata file must exist"
    with open(FINAL_MODEL_METADATA_FILE, "r") as f:
        meta = json.load(f)

    required_keys = [
        "model_identifier",
        "configuration_id",
        "algorithm",
        "feature_schema",
        "physical_activity_treatment",
        "selected_hyperparameters",
        "candidate_hyperparameter_grid",
        "cv_splitter",
        "cv_scoring_criterion",
        "best_cv_r2_score",
        "training_sample_count",
        "random_seed",
        "environment_info",
        "training_timestamp_utc",
        "model_artifact_path",
        "model_artifact_size_bytes",
    ]
    for key in required_keys:
        assert key in meta, f"Metadata missing required key: {key}"

    assert meta["training_sample_count"] == 3498
    assert meta["random_seed"] == 42
    assert meta["configuration_id"] == "M4a_RF_Full_Clip"


# -----------------------------------------------------------------------------
# 4. Artifact Integrity and Safeguard Tests
# -----------------------------------------------------------------------------
def test_holdout_evaluation_and_predictions_schema():
    """Verify holdout evaluation and prediction artifacts without modifying them."""
    assert EVALUATION_REPORT_JSON.exists()
    assert HOLDOUT_PREDICTIONS_CSV.exists()
    assert RESIDUAL_ANALYSIS_JSON.exists()

    with open(EVALUATION_REPORT_JSON, "r") as f:
        eval_data = json.load(f)
    assert eval_data["model_metadata"]["training_sample_size"] == 3498
    assert eval_data["model_metadata"]["holdout_sample_size"] == 1500

    preds_df = pd.read_csv(HOLDOUT_PREDICTIONS_CSV)
    assert len(preds_df) == 1500
    assert "predicted_score" in preds_df.columns
    assert "residual" in preds_df.columns
    assert "abs_error" in preds_df.columns

    # Verify mathematical consistency
    calc_res = preds_df[TARGET_COLUMN] - preds_df["predicted_score"]
    np.testing.assert_allclose(preds_df["residual"].to_numpy(), calc_res.to_numpy(), atol=1e-5)


def test_repeated_evaluation_safeguard():
    """Verify that execute_phase_2_step_3 refuses to re-evaluate without force flag."""
    from src.evaluate import execute_phase_2_step_3

    # When evaluation report exists and force_reevaluation=False, must raise RuntimeError
    with pytest.raises(RuntimeError, match="ONE-TIME EVALUATION SAFEGUARD"):
        execute_phase_2_step_3(force_reevaluation=False)


def test_legacy_model_artifact_preservation():
    """Verify Mental_Health_Model.pkl retains its exact legacy size (25,694,899 bytes)."""
    assert LEGACY_MODEL_FILE.exists()
    assert LEGACY_MODEL_FILE.stat().st_size == 25694899, (
        f"Legacy model file size changed! Expected 25,694,899 bytes, found {LEGACY_MODEL_FILE.stat().st_size}"
    )
