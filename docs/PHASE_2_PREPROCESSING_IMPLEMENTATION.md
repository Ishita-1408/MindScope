# MindScope — Phase 2: Preprocessing Implementation Report (Step 2.1)
**Document Version:** 1.0.0  
**Date:** 2026-11-06  
**Status:** Implemented, Tested & Verified (All 39 Tests Passing)  
**Project:** MindScope — Student Mental Health Analytics & Predictive Intelligence Platform  
**Target Environment:** Python 3.13.3, Scikit-Learn 1.8.0, Pandas 3.0.2, NumPy 2.4.4 inside `.venv`  

---

## 1. Executive Summary

In accordance with Phase 2, Step 2.1 of the MindScope engineering specification, the foundational **Leakage-Free Data Preprocessing Architecture** has been designed, implemented, and thoroughly tested. This implementation completely resolves the architectural defects and information leakage identified in the original notebook workflow (`ML_Project.ipynb`).

### Scope Strictly Preserved
- **Source Code Scope:** Only `src/__init__.py`, `src/preprocessing.py`, and `tests/test_preprocessing.py` were created.
- **Model Training Prohibited:** No models were fitted, tuned, or re-serialized during this step.
- **Data Preservation:** The original dataset `Student Social Media And Mental Health Impact.csv` was preserved without mutation.
- **Artifact Safety:** `Mental_Health_Model.pkl` was not modified or overwritten.
- **No Premature Changes:** `main.py`, `index.html`, and `script.js` remain completely untouched.

A dedicated test suite of 17 unit and integration tests was implemented in `tests/test_preprocessing.py`. When run alongside the existing test suite, all **39 tests passed with a 100% success rate**.

---

## 2. Implementation Architecture & Technical Decisions

The preprocessing system is centralized in [`src/preprocessing.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/src/preprocessing.py) and exports four core components:

### 2.1 Data Ingestion & Deduplication (`load_and_validate_data`)
- **Immutability:** The raw CSV file on disk is read-only.
- **Schema Validation:** Verifies the presence of all 13 columns (12 predictors + continuous target `Mental_Health_Score`). Missing columns immediately raise a structured `ValueError`.
- **Deduplication:** Automatically drops exact duplicate records (`drop_duplicates()`), reducing the raw 5,000 records to 4,998 unique student rows ($N = 4,998$).
- **Predictor / Target Separation:** Strictly decouples $X$ ($4,998 \times 12$) from $y$ ($4,998$ float values).
- **Sensitivity Exclude Switch:** Exposes `exclude_negative_activity: bool = False`. When set to `True`, drops the 10 rows with negative physical activity ($N = 4,988$) for offline sensitivity comparisons.

### 2.2 Scikit-Learn Compliant `CountryTopKEncoder`
- **Leakage Elimination:** In the original notebook, country frequencies were computed globally prior to splitting. In `CountryTopKEncoder`, frequencies are learned **strictly during `fit()`** on the training slice.
- **Preservation of Raw 'Other':** In the raw dataset, `'Other'` is already the largest category (1,880 rows). `CountryTopKEncoder` handles this explicitly:
  - When `include_other_in_top_k=True` (baseline mode), `'Other'` is treated as one of the $k$ budget categories alongside the top $k-1$ specific countries.
  - When `include_other_in_top_k=False`, the top $k$ named countries are selected, and `'Other'` is guaranteed as the fallback label.
  - Any infrequent or unseen country at inference time is mapped cleanly to `'Other'` without generating duplicate or conflicting labels.
- **Estimator API Compliance:** Implements standard `fit()`, `transform()`, and `get_feature_names_out()` methods. It is 100% compatible with `sklearn.base.clone`, `get_params()`, and `set_params()`.

### 2.3 Physical Activity Sensitivity Handler (`PhysicalActivityHandler`)
The 10 records with negative values ($-0.1$ to $-0.4$) are supported through three distinct, uncoupled configurations:
1. `strategy='clip'`: Caps negative values at 0.0 ($\max(x, 0.0)$), reflecting the true physical boundary.
2. `strategy='nan'`: Replaces negative values with `np.nan`, allowing downstream `SimpleImputer(strategy='median')` to impute using fold-level training medians.
3. `strategy='passthrough'`: Leaves values untouched (used when observations are filtered via `load_and_validate_data(exclude_negative_activity=True)`).

### 2.4 Universal Estimator Compatibility & Dense Matrix Guarantees
- `HistGradientBoostingRegressor` requires dense 2D float arrays and throws a runtime `TypeError` when passed `scipy.sparse` matrices.
- To guarantee cross-estimator compatibility without complex branching, `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` was configured across all categorical branches.
- **Memory Footprint Evaluation:** With 3,498 training records and 39 encoded features, the dense in-memory matrix is $3,498 \times 39 \times 8 \text{ bytes} \approx 1.09 \text{ MB}$. This eliminates sparse-matrix execution errors across `HistGradientBoostingRegressor`, `RandomForestRegressor`, `Ridge`, and `LinearRegression` with negligible memory consumption.

### 2.5 Feature Schema Configurations
- **Full Configuration (12 Predictors):** Ingests all 12 raw features, yielding a 39-column dense feature representation.
- **Reduced Configuration (11 Predictors):** Omits `Purpose_Of_Use` via `build_preprocessor(drop_purpose=True)`, yielding a 35-column dense feature representation (exactly 4 dummy columns removed).

---

## 3. Automated Test Suite & Empirical Verification

A comprehensive automated test suite was constructed in [`tests/test_preprocessing.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/tests/test_preprocessing.py) and executed within `.venv`.

### 3.1 Preprocessing Test Results Breakdown (`tests/test_preprocessing.py`)

| Test Name | Verified Functionality | Status |
| :--- | :--- | :--- |
| `test_data_loading_and_deduplication` | Deduplicates raw 5,000 rows to 4,998 unique rows; separates $X$ and $y$. | **PASSED** |
| `test_data_loading_exclude_negative_activity` | Drops 10 negative rows, producing exactly 4,988 non-negative rows. | **PASSED** |
| `test_data_loading_missing_file` | Raises `FileNotFoundError` for non-existent paths. | **PASSED** |
| `test_data_loading_schema_validation` | Raises `ValueError` if required columns are missing. | **PASSED** |
| `test_country_top_k_encoder_clone_and_params` | Verifies `sklearn.base.clone`, `get_params()`, and `set_params()`. | **PASSED** |
| `test_country_top_k_encoder_fit_and_transform` | Verifies fit-only learning and mapping unseen categories to `'Other'`. | **PASSED** |
| `test_country_top_k_encoder_unfitted_raises` | Verifies `RuntimeError` on calling `transform()` prior to `fit()`. | **PASSED** |
| `test_physical_activity_handler_clip` | Verifies clamping of negative values to 0.0. | **PASSED** |
| `test_physical_activity_handler_nan` | Verifies conversion of negative values to `np.nan`. | **PASSED** |
| `test_physical_activity_handler_invalid_strategy`| Verifies `ValueError` when an invalid strategy string is passed. | **PASSED** |
| `test_pipeline_missing_value_imputation` | Injects NaNs across numeric and categorical columns; verifies zero-NaN output. | **PASSED** |
| `test_full_vs_reduced_feature_schemas` | Verifies dimension difference between full (39 cols) and reduced (35 cols). | **PASSED** |
| `test_no_preprocessing_fit_on_test_data` | Verifies learned encoder categories remain unchanged after test transform. | **PASSED** |
| `test_candidate_estimator_pipeline_compatibility[LinearRegression]` | Verifies pipeline execution and prediction with `LinearRegression`. | **PASSED** |
| `test_candidate_estimator_pipeline_compatibility[Ridge]` | Verifies pipeline execution and prediction with `Ridge`. | **PASSED** |
| `test_candidate_estimator_pipeline_compatibility[RandomForestRegressor]` | Verifies pipeline execution and prediction with `RandomForestRegressor`. | **PASSED** |
| `test_candidate_estimator_pipeline_compatibility[HistGradientBoostingRegressor]` | Verifies dense matrix execution with `HistGradientBoostingRegressor`. | **PASSED** |

### 3.2 Full Project Regression Test Run
Executing `pytest tests -v` confirmed that all 39 tests in the repository pass without failure:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.7.0 -- .venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Ishit\OneDrive\Desktop\MindScope
plugins: anyio-4.16.0
collected 39 items

tests/test_backend.py::test_root_endpoint PASSED                         [  2%]
tests/test_backend.py::test_health_endpoint_healthy PASSED               [  5%]
tests/test_backend.py::test_predict_valid_input PASSED                   [  7%]
tests/test_backend.py::test_predict_age_below_18_rejected PASSED         [ 10%]
tests/test_backend.py::test_predict_age_above_24_rejected PASSED         [ 12%]
tests/test_backend.py::test_predict_invalid_categorical_gender PASSED    [ 15%]
tests/test_backend.py::test_predict_invalid_categorical_stress_level PASSED [ 17%]
tests/test_backend.py::test_predict_negative_physical_activity_rejected PASSED [ 20%]
tests/test_backend.py::test_predict_excessive_daily_unlocks_rejected PASSED [ 23%]
tests/test_backend.py::test_predict_insufficient_sleep_rejected PASSED   [ 25%]
tests/test_backend.py::test_predict_missing_required_fields PASSED       [ 28%]
tests/test_backend.py::test_model_loading_failure_handling PASSED        [ 30%]
tests/test_backend.py::test_different_working_directory_path_safety PASSED [ 33%]
tests/test_frontend_contracts.py::test_index_html_has_no_stray_characters PASSED [ 35%]
tests/test_frontend_contracts.py::test_index_html_contains_error_label_element PASSED [ 38%]
tests/test_frontend_contracts.py::test_index_html_age_field_bounds PASSED [ 41%]
tests/test_frontend_contracts.py::test_script_js_dynamic_api_base PASSED [ 43%]
tests/test_frontend_contracts.py::test_script_js_defensive_error_rendering PASSED [ 46%]
tests/test_frontend_contracts.py::test_script_js_duplicate_submission_prevention PASSED [ 48%]
tests/test_frontend_contracts.py::test_script_js_non_json_error_handling PASSED [ 51%]
tests/test_frontend_contracts.py::test_script_js_numerical_bounds_alignment PASSED [ 53%]
tests/test_frontend_contracts.py::test_field_id_alignment_between_html_and_js PASSED [ 56%]
tests/test_preprocessing.py::test_data_loading_and_deduplication PASSED  [ 58%]
tests/test_preprocessing.py::test_data_loading_exclude_negative_activity PASSED [ 61%]
tests/test_preprocessing.py::test_data_loading_missing_file PASSED       [ 64%]
tests/test_preprocessing.py::test_data_loading_schema_validation PASSED  [ 66%]
tests/test_preprocessing.py::test_country_top_k_encoder_clone_and_params PASSED [ 69%]
tests/test_preprocessing.py::test_country_top_k_encoder_fit_and_transform PASSED [ 71%]
tests/test_preprocessing.py::test_country_top_k_encoder_unfitted_raises PASSED [ 74%]
tests/test_preprocessing.py::test_physical_activity_handler_clip PASSED  [ 76%]
tests/test_preprocessing.py::test_physical_activity_handler_nan PASSED   [ 79%]
tests/test_preprocessing.py::test_physical_activity_handler_invalid_strategy PASSED [ 82%]
tests/test_preprocessing.py::test_pipeline_missing_value_imputation PASSED [ 84%]
tests/test_preprocessing.py::test_full_vs_reduced_feature_schemas PASSED [ 87%]
tests/test_preprocessing.py::test_no_preprocessing_fit_on_test_data PASSED [ 89%]
tests/test_preprocessing.py::test_candidate_estimator_pipeline_compatibility[LinearRegression-kwargs0] PASSED [ 92%]
tests/test_preprocessing.py::test_candidate_estimator_pipeline_compatibility[Ridge-kwargs1] PASSED [ 94%]
tests/test_preprocessing.py::test_candidate_estimator_pipeline_compatibility[RandomForestRegressor-kwargs2] PASSED [ 97%]
tests/test_preprocessing.py::test_candidate_estimator_pipeline_compatibility[HistGradientBoostingRegressor-kwargs3] PASSED [100%]

======================= 39 passed, 9 warnings in 4.41s ========================
```
*(Note: The 9 captured warnings are the legacy `InconsistentVersionWarning` notices emitted by the legacy `Mental_Health_Model.pkl` in `test_backend.py`, which will be eliminated when the new pipeline is trained and serialized in Step 2.2).*

---

## 4. Current Limitations & Transition to Step 2.2

1. **Model Retraining Deferred:** The preprocessing pipeline is fully functional and unit tested, but has not yet been fitted on the complete training set to persist a new model artifact.
2. **Next Milestone Action (Step 2.2):** Execute the 5-fold nested cross-validation experiment across the four candidate estimators (`LinearRegression`, `Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`), evaluate the sensitivity strategies and feature redundancy, select the winning configuration using training data only, refit on all 3,498 training records, and persist `Mental_Health_Model.pkl` under native `scikit-learn==1.8.0`.

---
*End of Phase 2 Step 2.1 Implementation Report. Awaiting user review and authorization before proceeding to Step 2.2 (Model Training & Nested Cross-Validation).*
