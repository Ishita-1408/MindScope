# MindScope — Phase 1: Environment Setup & Verification Report
**Document Version:** 1.0.0  
**Date:** 2026-11-06  
**Status:** Environment Stabilized & Verified  
**Project:** MindScope — Student Mental Health Analytics & Predictive Intelligence Platform  
**Interpreter Target:** Python 3.13.3 (64-bit AMD64) inside `.venv`  

---

## 1. Overview & Setup Objectives

In accordance with the approved Phase 1.2 implementation plan, a dedicated, isolated Python virtual environment (`.venv`) has been created and verified for MindScope. This setup eliminates dependency bleed from the host machine's global Python environment, enforces exact package pinning, segregates runtime from development tools, and establishes strict boundaries around legacy model artifacts.

### Scope and Safety Guardrails Preserved
- **Zero Modifications to Source Code:** `main.py`, `index.html`, `style.css`, `script.js`, and `ML_Project.ipynb` were not modified.
- **Dataset Preservation:** `Student Social Media And Mental Health Impact.csv` was untouched.
- **Model Safety:** `Mental_Health_Model.pkl` was **not modified, overwritten, or reserialized**.
- **No Premature Feature Code:** React development and advanced analytics modules were not introduced.

---

## 2. Environment Architecture & Isolation Verification

### 2.1 Virtual Environment Creation
The virtual environment was provisioned from the system's CPython 3.13.3 binary:
```powershell
# Environment Provisioning Command
python -m venv .venv
```

### 2.2 Interpreter & Path Verification
Direct programmatic interrogation via `.venv\Scripts\python.exe` confirms complete isolation:

| Property | Evaluated Value | Verification Status |
| :--- | :--- | :--- |
| **`sys.executable`** | `C:\Users\Ishit\OneDrive\Desktop\MindScope\.venv\Scripts\python.exe` | **PASSED** (Points to isolated `.venv`) |
| **`sys.prefix`** | `C:\Users\Ishit\OneDrive\Desktop\MindScope\.venv` | **PASSED** |
| **`sys.base_prefix`** | `C:\Users\Ishit\AppData\Local\Programs\Python\Python313` | **PASSED** (Host Python runtime identified) |
| **Virtual Environment Active (`sys.prefix != sys.base_prefix`)** | `True` | **PASSED** (Strict isolation established) |
| **IDE Interpreter Configuration (`.vscode/settings.json`)** | `{"python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"}` | **PASSED** (IDE bound to `.venv`) |

---

## 3. Pinned Dependency Architecture

Dependencies have been categorized into two formal project files, separating production runtime requirements from development and QA tooling.

### 3.1 Core Runtime Dependencies (`requirements.txt`)
Contains only packages required for running the FastAPI inference server and performing Scikit-Learn predictions:
```text
# MindScope — Core Runtime Dependencies
# Python 3.13.3 Compatible
fastapi==0.141.1
uvicorn==0.52.4
pydantic==2.13.4
joblib==1.5.3
pandas==3.0.2
scikit-learn==1.8.0
numpy==2.4.4
scipy==1.17.1
```

### 3.2 Development & Testing Dependencies (`requirements-dev.txt`)
Contains tools required for regression testing and HTTP client simulation:
```text
# MindScope — Development and Testing Dependencies
-r requirements.txt
pytest==9.1.1
httpx==0.28.1
```

### 3.3 Deferred Future Dependencies (Phases 4 & 5)
The following packages are documented for subsequent project phases but are explicitly excluded from the Phase 1 core runtime:
- **`shap==0.52.0`**: Deferred to Phase 4 (Explainable AI module; TreeExplainer / KernelExplainer).
- **`statsmodels==0.14.6`**: Deferred to Phase 5 (Statistical multivariable analysis & OLS diagnostics).
- **`seaborn==0.13.2` / `matplotlib>=3.8.0`**: Deferred to Phase 5 (Static visual analytics & diagnostic distribution plots).

---

## 4. Verification Results & Package Inventory

### 4.1 Installed Package Version Audit
All installed packages inside `.venv` were verified against their specified versions:

| Package | Installed Version | Pinned Target | Status |
| :--- | :--- | :--- | :--- |
| **`fastapi`** | `0.141.1` | `0.141.1` | **MATCH** |
| **`uvicorn`** | `0.52.4` | `0.52.4` | **MATCH** |
| **`pydantic`** | `2.13.4` | `2.13.4` | **MATCH** |
| **`pydantic-core`** | `2.46.4` | (Transitive) | **VERIFIED** |
| **`joblib`** | `1.5.3` | `1.5.3` | **MATCH** |
| **`pandas`** | `3.0.2` | `3.0.2` | **MATCH** |
| **`scikit-learn`** | `1.8.0` | `1.8.0` | **MATCH** |
| **`numpy`** | `2.4.4` | `2.4.4` | **MATCH** |
| **`scipy`** | `1.17.1` | `1.17.1` | **MATCH** |
| **`pytest`** | `9.1.1` | `9.1.1` | **MATCH** |
| **`httpx`** | `0.28.1` | `0.28.1` | **MATCH** |
| **`starlette`** | `1.10.0` | (Transitive) | **VERIFIED** |

### 4.2 Module Import Test
Executing an integrated runtime import test across all dependencies inside `.venv`:
```python
import fastapi, uvicorn, pydantic, joblib, pandas, sklearn, numpy, scipy, pytest, httpx
```
**Result:** Executed with return code `0`. All modules resolved successfully without runtime link errors.

---

## 5. Model Compatibility & Known Limitations

### 5.1 Legacy Serialization Drift
The serialized model artifact `Mental_Health_Model.pkl` (24.5 MB) was investigated in `.venv`:
- **Discrepancy:** Artifact generated under `scikit-learn==1.9.0`; active environment runs `scikit-learn==1.8.0`.
- **Warning Count:** Exactly 122 `InconsistentVersionWarning` notices emitted upon `joblib.load()`.
- **Inspection Failure:** Calling `repr(model)` or `repr(model.named_steps['preprocessor'])` raises:
  ```python
  AttributeError: 'ColumnTransformer' object has no attribute 'force_int_remainder_cols'
  ```
- **Inference Verification:** Executing `model.predict(sample_df)` succeeds mathematically, generating predictions (e.g., `[8.083]`).

> [!WARNING]
> **Prohibition Against Re-saving Legacy Pickle:**
> Under no circumstances should `Mental_Health_Model.pkl` be re-serialized (`joblib.dump()`) within `scikit-learn 1.8.0`. Doing so will freeze corrupted internal state and permanently compromise pipeline behavior. In Phase 2, the pipeline will be retrained from scratch using fold-encapsulated estimators matching the locked environment.

---

## 6. Application Startup & Endpoint Verification

Executing `TestClient(main.app)` within `.venv` confirmed initial baseline execution:

```text
GET /
Status: 200 OK
Response: ["Welcome to Sheryians AI School Guys"]

POST /predict
Payload: {
  "age": 21, "gender": "Male", "country": "India", "academic_level": "Undergraduate",
  "most_used_platform": "YouTube", "purpose_of_use": "Education",
  "avg_daily_usage_hours": 3.5, "daily_unlocks": 50, "study_hours": 6.0,
  "physical_activity_hours": 1.0, "sleep_hours_per_night": 8.0, "stress_level": "Low"
}
Status: 200 OK
Response: {"predicted_mental_health_score": 8.08}
```

---

## 7. Version Control Configuration (`.gitignore`)

A comprehensive `.gitignore` file was initialized in the project root to guarantee that virtual environments, compiled bytecode, and test caches are not tracked:
- Excludes `.venv/`, `env/`, `venv/`.
- Excludes `__pycache__/`, `*.pyc`, `*.pyo`.
- Excludes `.pytest_cache/`, `.coverage`.
- Whitelists `.vscode/settings.json` while ignoring user-specific IDE caches.

---

## 8. Summary of Completed Deliverables

1. **`.venv` Provisioned:** Fully isolated CPython 3.13.3 virtual environment.
2. **Interpreter Configured:** `.vscode/settings.json` points to `${workspaceFolder}/.venv/Scripts/python.exe`.
3. **Core Dependencies Pinned:** `requirements.txt` restricted strictly to runtime inference packages.
4. **Dev Dependencies Pinned:** `requirements-dev.txt` created with `pytest` and `httpx`.
5. **`.gitignore` Configured:** Excludes all environment and cache artifacts.
6. **Integrity Verified:** All packages and baseline API endpoints tested inside `.venv`.
7. **Model Compatibility Documented:** Legacy artifact limitations logged; retraining slated for Phase 2.

*Phase 1.2 is complete. Awaiting user review and authorization before proceeding to Phase 1.3 (Source-Code Defect Remediation).*
