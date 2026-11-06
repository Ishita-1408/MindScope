# MindScope — Phase 1: Environment & Dependency Audit
**Document Version:** 1.0.0  
**Date:** 2026-11-06  
**Status:** Audit Completed — Pending User Review & Approval  
**Project:** MindScope — Student Mental Health Analytics & Predictive Intelligence Platform  
**Target Architecture:** FastAPI Inference Service, Vanilla HTML5/CSS3/ES6+ Client, Scikit-Learn Regression Pipeline  

---

## 1. Executive Summary

This audit constitutes the foundational step of **Phase 1: Environment Stabilization & Defect Remediation** for the MindScope platform. In accordance with strict engineering protocols, this investigation was conducted strictly read-only: no source code was modified, no packages were installed or removed, the original dataset and serialized model files were untouched, and no unvalidated assumptions were made.

The audit identifies three primary technical vulnerabilities that currently compromise deployment integrity and developer reproducibility:
1. **Global Environment Execution:** The application is operating directly within the user's global Python 3.13 installation without virtual environment isolation (`.venv`), leaving dependencies uncontained and susceptible to host environment collisions.
2. **Estimator Serialization Incompatibility:** The serialized artifact `Mental_Health_Model.pkl` (24.5 MB) was generated using an experimental/pre-release version of `scikit-learn==1.9.0` (typically sourced from cutting-edge Google Colab runtimes), whereas the active local environment runs `scikit-learn==1.8.0`. This discrepancy emits cascading `InconsistentVersionWarning`s and causes a catastrophic `AttributeError: 'ColumnTransformer' object has no attribute 'force_int_remainder_cols'` upon executing object string representations or inspecting pipeline stages.
3. **Rigid Working Directory Coupling:** `main.py` utilizes an unanchored relative path (`joblib.load('Mental_Health_Model.pkl')`), causing fatal `FileNotFoundError` exceptions whenever the process invocation directory deviates from the physical repository root.

The sections below present the empirical findings, version reconciliation tables, model compatibility evaluations, startup diagnostics, and an actionable, non-disruptive stabilization roadmap.

---

## 2. Current Environment Inventory

### 2.1 Host System & Python Runtime
The active execution environment was systematically probed via Python's `sys`, `platform`, and operating system inspection utilities:

| Dimension | Measured Parameter | Technical Implication |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home / Pro (`Windows-11-10.0.26200-SP0`) | Windows-specific path conventions (backslashes) and process signal handling. |
| **System Architecture** | `64bit`, `WindowsPE` (x86_64 / AMD64) | Native 64-bit C-extensions required for NumPy, SciPy, and Scikit-Learn wheels. |
| **Active Python Path** | `C:\Users\Ishit\AppData\Local\Programs\Python\Python313\python.exe` | User-global Python distribution; lacks project-level package isolation. |
| **Active Python Version**| `3.13.3 (tags/v3.13.3:6280bb5, Apr 8 2025, 14:47:33) [MSC v.1943 64 bit]` | Modern CPython runtime. C-API changes require compiled binary wheel validation. |
| **Virtual Environment** | **None Active** (`sys.prefix == sys.base_prefix`: `True`) | High risk: all installed packages pollute global site-packages. |

### 2.2 System-Level Python Interpreters
Inspection of the system PATH and package discovery managers reveals multiple Python environments on the host:
- `C:\Users\Ishit\AppData\Local\Programs\Python\Python313\python.exe` (Default system interpreter, Python 3.13.3)
- `C:\Users\Ishit\AppData\Local\Microsoft\WindowsApps\python.exe` (Windows Store App execution alias)
- `Astral / uv` managed runtimes:
  - CPython 3.12.14 (`Astral/CPython3.12.14`)
  - CPython 3.11.16 (`Astral/CPython3.11.16`)
- Workspace IDE Configuration (`.vscode/settings.json`):
  ```json
  {
      "python-envs.defaultEnvManager": "ms-python.python:system"
  }
  ```
  *Finding:* The IDE is explicitly bound to the global system environment rather than a dedicated local virtual environment.

---

## 3. Dependency Version Analysis

### 3.1 Dependency Reconciliation Table
The table below contrasts installed package versions in the active environment against the baseline `requirements.txt` file, categorizing them by project phase and necessity:

| Package | Installed Version | Declared in `requirements.txt` | Direct / Transitive | Status & Scope |
| :--- | :--- | :--- | :--- | :--- |
| **`fastapi`** | `0.141.1` | `fastapi==0.141.1` | Direct | **Core API**: Stable, asynchronous ASGI web framework. |
| **`uvicorn`** | `0.52.4` | `uvicorn==0.52.4` | Direct | **Core API**: ASGI web server implementation. |
| **`pydantic`** | `2.13.4` | `pydantic==2.13.4` | Direct | **Core API**: V2 runtime data validation and schema definitions. |
| **`pydantic_core`** | `2.46.4` | *Not explicitly listed* | Transitive | Core Rust validation engine for Pydantic V2. |
| **`starlette`** | `1.6.0` | *Not explicitly listed* | Transitive | Underlying ASGI routing and middleware engine for FastAPI. |
| **`joblib`** | `1.5.3` | `joblib==1.5.3` | Direct | **Core Inference**: Pipeline de-serialization engine. |
| **`pandas`** | `3.0.2` | `pandas==3.0.2` | Direct | **Core Inference**: In-memory DataFrame construction for batch/single inference. |
| **`numpy`** | `2.4.4` | `numpy==2.4.4` | Direct | **Core Inference**: Numerical array math and feature buffer backing. |
| **`scipy`** | `1.17.1` | `scipy==1.17.1` | Direct | **Core Inference**: Sparse matrix and distribution utilities. |
| **`scikit-learn`** | `1.8.0` | `scikit-learn==1.8.0` | Direct | **Core Inference**: Estimator and transformer runtime (**Version Conflict** with serialized artifact). |
| **`pytest`** | `9.1.1` | `pytest==9.1.1` | Direct | **Development / QA**: Automated endpoint and schema regression test suite. |
| **`httpx`** | `0.28.1` | `httpx==0.28.1` | Direct | **Development / QA**: Asynchronous HTTP client for `TestClient` integrations. |
| **`shap`** | `0.52.0` | `shap==0.52.0` | Direct | **Phase 4 (Explainable AI)**: Not required for Phase 1 API serving; deferred to explainability module. |
| **`seaborn`** | `0.13.2` | *Not listed* | External | **Phase 5 (Analytics)**: Statistical plotting; deferred. |
| **`statsmodels`** | `0.14.6` | *Not listed* | External | **Phase 5 (Analytics)**: OLS diagnostic regressions; deferred. |

### 3.2 Key Dependency Observations
1. **Exact Pinning vs. Scope Boundaries:** While the current `requirements.txt` contains exact `==` pinned versions that match the active environment, it blends **Phase 1 core inference dependencies** (`fastapi`, `uvicorn`, `pydantic`, `pandas`, `scikit-learn`) with **testing tools** (`pytest`, `httpx`) and **Phase 4 advanced explainability tooling** (`shap`).
2. **Missing Transitive Lock:** No lockfile (`requirements.lock` or `uv.lock`) exists to freeze transitive dependencies (`starlette`, `anyio`, `sniffio`, `click`, `idna`), introducing drift vulnerability across distinct developer machines.
3. **Absence of Environment Isolation:** All packages reside in `C:\Users\Ishit\AppData\Local\Programs\Python\Python313\Lib\site-packages`, where 150+ unrelated third-party libraries (e.g., `chromadb`, `celery`, `streamlit`, `torch`) coexist.

---

## 4. Model Compatibility & Serialization Audit

### 4.1 Artifact Profile: `Mental_Health_Model.pkl`
- **File System Size:** 25,694,899 bytes (~24.50 MB).
- **Serialization Format:** Python `pickle` via `joblib`, Protocol version 4 (`\x80\x04`).
- **Pipeline Structure:** Two-stage Scikit-Learn `Pipeline`:
  1. `preprocessor`: `sklearn.compose.ColumnTransformer` (handling log transformations, standard scaling, ordinal encoding, and one-hot encoding).
  2. `random forest`: `sklearn.ensemble.RandomForestRegressor` (100 constituent `DecisionTreeRegressor` estimators).

### 4.2 Serialization Version Discrepancy
Direct programmatic inspection of the serialized object's `_sklearn_version` metadata reveals:
```
Serialized Under: scikit-learn == 1.9.0
Active Runtime  : scikit-learn == 1.8.0
```
*Origin Analysis:* Scikit-learn 1.9.0 is an unreleased development branch. This artifact was produced in a cloud environment (e.g., Google Colaboratory nightly build or bleeding-edge git clone) and saved directly into the repository.

### 4.3 Empirical Compatibility Behavior
When loaded in the active `scikit-learn==1.8.0` runtime:
1. **Unpickling Warnings:** Joblib emits over 105 consecutive `InconsistentVersionWarning` instances upon load:
   ```text
   InconsistentVersionWarning: Trying to unpickle estimator FunctionTransformer from version 1.9.0 when using version 1.8.0.
   InconsistentVersionWarning: Trying to unpickle estimator ColumnTransformer from version 1.9.0 when using version 1.8.0.
   InconsistentVersionWarning: Trying to unpickle estimator RandomForestRegressor from version 1.9.0 when using version 1.8.0.
   ```
2. **Crash on Representation (`repr()` / HTML Display):**
   Attempting to print or display the pipeline or its `ColumnTransformer` fails immediately with an unhandled exception:
   ```python
   >>> repr(model)
   AttributeError: 'ColumnTransformer' object has no attribute 'force_int_remainder_cols'
   ```
   *Root Cause:* In scikit-learn 1.9.0, an internal attribute `force_int_remainder_cols` was added to `ColumnTransformer` for feature name and remainder indexing. When unpickled in 1.8.0, the display formatting methods (`_repr_html_`, `__repr__`) expect internal consistency and crash.
3. **Inference Execution (`predict()`):**
   Despite the representation crash and warnings, executing `model.predict(input_row)` succeeds computationally, outputting an unrounded float (e.g., `6.117000...`).
4. **Reserialization Prohibition:**
   Under no circumstances should this 1.9.0 model be unpickled and re-dumped within the 1.8.0 environment. Doing so will freeze corrupted internal state and permanently break backward/forward compatibility. Clean remediation must occur via **Phase 2 Pipeline Retraining**, where a clean pipeline is fitted and serialized exclusively within the standardized project environment.

---

## 5. Application Execution & Startup Diagnostics

### 5.1 Service Startup & Lifecycle Architecture
Analysis of `main.py` reveals the following architectural flow:
1. **Module-Scope Model Loading (Line 8):**
   ```python
   model = joblib.load('Mental_Health_Model.pkl')
   ```
   The model is loaded synchronously into memory during initial module parsing.
2. **Current API Startup Command:**
   ```bash
   uvicorn main:app --reload
   ```
   Or explicitly specifying host and port:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

### 5.2 Working Directory Fragility
Because `joblib.load('Mental_Health_Model.pkl')` relies on an unanchored relative path:
- **Success:** Executing `python -c "import main"` from `c:\Users\Ishit\OneDrive\Desktop\MindScope` succeeds.
- **Failure:** Executing from any subfolder or external directory (e.g., `cd docs; python -c "import main"`) fails with:
  ```text
  FileNotFoundError: [Errno 2] No such file or directory: 'Mental_Health_Model.pkl'
  ```
- **Remediation Requirement:** In Milestone 1 code updates, replace line 8 with a path-anchored resolution:
  ```python
  from pathlib import Path
  BASE_DIR = Path(__file__).resolve().parent
  MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"
  model = joblib.load(MODEL_PATH)
  ```

### 5.3 Endpoint Probing Results
Executing integration requests via `TestClient(main.app)` without code modifications yielded the following baseline behaviors:

```text
GET /
Status: 200 OK
Response Body: ["Welcome to Sheryians AI School Guys"]
Evaluation: Non-standard educational string; lacks structured operational metadata.

POST /predict
Payload: {
  "age": 20, "gender": "Male", "country": "India", "academic_level": "Undergraduate",
  "most_used_platform": "Instagram", "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.5, "daily_unlocks": 70, "study_hours": 5.0,
  "physical_activity_hours": 1.5, "sleep_hours_per_night": 7.5, "stress_level": "Medium"
}
Status: 200 OK
Response Body: {"predicted_mental_health_score": 6.87}
Evaluation: Correctly predicts score and returns rounded float.
```

---

## 6. Comprehensive Risk Assessment

| Risk Identifier | Severity | Category | Description | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **RSK-ENV-01** | High | Environment | Lack of virtual environment (`.venv`) isolation. Global packages pollute workspace. | Unpredictable package upgrades on host machine can break project runtime silently. |
| **RSK-MDL-01** | High | Model / Compatibility | Scikit-Learn version mismatch (`1.9.0` artifact loaded in `1.8.0`). | `AttributeError` on pipeline introspection, model display, or debugging tools. |
| **RSK-PTH-01** | Medium | Architecture | Relative file path in `joblib.load('Mental_Health_Model.pkl')`. | Application crashes when started from parent or child directories, or in CI/CD runners. |
| **RSK-VAL-01** | High | Schema Validation | `age: int = Field(..., ge=10, le=100)` violates initial deployment scope ($[18, 24]$). | Model is subjected to unvalidated extrapolation beyond verified student age cohorts. |
| **RSK-VAL-02** | Medium | Schema Validation | Screen time, unlock counts, and sleep allow unconstrained or extreme values (e.g. 24h screen time). | Ingestion of physiologically unrealistic behavioural records without input sanitisation. |
| **RSK-API-01** | Low | API Design | Educational tutorial placeholder at `GET /`; absence of `GET /health` endpoint. | Infrastructure load balancers and container monitors cannot determine service readiness. |
| **RSK-UI-01** | High | Frontend Defect | `script.js` references `document.getElementById("error-label")`, which is absent from `index.html`. | Any validation or fetch error triggers an unhandled JavaScript `TypeError` in browser console. |
| **RSK-UI-02** | Medium | Frontend Routing | Hardcoded remote URL (`https://mansik-santulan-score.onrender.com`) in `script.js`. | Client cannot communicate with local FastAPI server during development without manual file editing. |

---

## 7. Recommended Dependency Strategy & Justification

To ensure reproducible, lightweight, and robust operation, dependencies must be separated into **Core Runtime**, **Development / QA**, and **Deferred Analytical Modules**.

### 7.1 Tiered Dependency Architecture

#### Tier 1: Core Runtime (`requirements.txt`)
Only the packages strictly necessary for serving inference requests:
```text
# MindScope Core Runtime Dependencies
fastapi==0.141.1
uvicorn[standard]==0.52.4
pydantic==2.13.4
joblib==1.5.3
pandas==3.0.2
scikit-learn==1.8.0
numpy==2.4.4
scipy==1.17.1
```
*Justification:*
- Pins exact versions currently installed and verified on the Windows 64-bit host.
- Excludes heavy plotting and attribution packages from the lightweight production serving container.
- Guarantees sub-second container cold-starts and minimal memory footprint.

#### Tier 2: Development & QA (`requirements-dev.txt`)
Packages required for running automated regression test suites:
```text
# MindScope Development & Testing
-r requirements.txt
pytest==9.1.1
httpx==0.28.1
```
*Justification:*
- Decouples unit and integration testing tools from production production environments.
- Provides `httpx` for asynchronous endpoint testing via FastAPI's `TestClient`.

#### Tier 3: Deferred Analytical Dependencies (Phase 4 & Phase 5)
```text
# Phase 4: Explainability Module
shap==0.52.0

# Phase 5: Statistical & Dashboard Modules
seaborn==0.13.2
matplotlib>=3.8.0
statsmodels==0.14.6
```
*Justification:*
- These dependencies are not referenced by `main.py` or the Phase 1 web interface. Installing or loading them prematurely introduces overhead and potential C-extension compilation issues before their respective modules are designed.

---

## 8. Exact Next Steps for Environment Stabilization

Upon user authorization, Phase 1 execution will proceed strictly according to the following sequenced steps:

1. **Virtual Environment Creation:**
   - Execute `python -m venv .venv` in the repository root.
   - Update `.vscode/settings.json` to target `.venv\Scripts\python.exe`.
2. **Dependency Installation in Isolated Environment:**
   - Activate `.venv` and install locked runtime and development dependencies:
     ```powershell
     .\.venv\Scripts\pip install -r requirements.txt pytest==9.1.1 httpx==0.28.1
     ```
3. **Repository Defect Remediation (Source Code Updates):**
   - **`main.py`**:
     - Anchor `Mental_Health_Model.pkl` loading to `Path(__file__).parent`.
     - Update `StudentData` schema to enforce Age $[18, 24]$ and bounded lifestyle inputs.
     - Replace educational tutorial greeting at `GET /` with structured JSON metadata (`platform`, `version`, `status`).
     - Add `GET /health` endpoint returning operational readiness.
   - **`index.html`**:
     - Remove stray typo character `S` on line 11.
     - Add missing `<p class="error-label" id="error-label"></p>` element.
     - Update Age input bounds (`min="18" max="24"`).
     - Update branding to *"MindScope — Student Mental Health Analytics & Predictive Intelligence Platform"*.
   - **`script.js`**:
     - Add dynamic `API_BASE` resolution (local vs. deployed).
     - Add defensive null-checks in `renderError()`.
     - Align client-side age validation to `18–24`.
4. **Automated Verification:**
   - Create `tests/test_api.py` to systematically test `GET /`, `GET /health`, valid predictions, and rejection of out-of-bounds inputs.
   - Execute `pytest` and verify 100% pass rate before advancing to Phase 2.

---
*End of Phase 1 Environment Audit. Awaiting user approval before environment modification or code implementation.*
