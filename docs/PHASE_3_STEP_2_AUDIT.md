# MindScope — Phase 3, Step 3.2: Methodological and Implementation Audit

**Audit Date**: 2026-11-07  
**Auditor**: Antigravity Automated Verification Subsystem  
**Scope**: Step 3.2 Global SHAP Analysis & Visualization Cache  
**Status**: READ-ONLY AUDIT COMPLETED  

---

## 1. Executive Summary

This document presents a comprehensive, read-only methodological audit of MindScope Phase 3, Step 3.2: Global SHAP Analysis and Visualization Cache. The audit examined source implementations ([`src/explain.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/src/explain.py), [`src/global_shap.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/src/global_shap.py)), test suites ([`tests/test_explain.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/tests/test_explain.py)), generated serialized artifacts ([`artifacts/explanations/`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations)), report documentation ([`docs/PHASE_3_STEP_2_GLOBAL_ANALYSIS.md`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/docs/PHASE_3_STEP_2_GLOBAL_ANALYSIS.md)), and environment specifications ([`requirements.txt`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/requirements.txt), [`requirements-dev.txt`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/requirements-dev.txt)).

The audit confirmed that data governance protocols were strictly respected: the sealed holdout dataset was never accessed, the legacy model artifact remains intact, and the TreeSHAP additive efficiency axiom was empirically validated to machine precision ($4.26 \times 10^{-14}$). However, the audit identified three discrepancies requiring correction: (1) a documentation inconsistency in the displayed "Relative Share" percentages in Table 3.1 which sum to 103.8% due to an incorrect divisor, (2) omitted pinned dependencies (`shap==0.52.0` and `matplotlib==3.11.2`) in the requirements manifests, and (3) a lack of dedicated automated unit tests for `src/global_shap.py` and the serialized explanation artifacts.

---

## 2. Training Sample & Data Governance

### 2.1 Audit Findings
- **Data Source**: [`Student Social Media And Mental Health Impact.csv`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/Student%20Social%20Media%20And%20Mental%20Health%20Impact.csv) (5,000 raw rows; 4,998 unique records after deduplication).
- **Split Implementation**: `train_test_split(X, y, test_size=0.30, random_state=42)` executed in [`src/global_shap.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/src/global_shap.py#L64-L66).
  - Empirical verification confirmed that `test_size=0.30` on the 4,998 deduplicated rows produces exactly 3,498 training records and 1,500 test records, with index arrays bitwise identical to the `test_size=1500` split used in Phase 2 model training.
- **Holdout Isolation**: `X_test` and `y_test` are never assigned to the explainer, saved, or evaluated. The file [`artifacts/sealed_holdout_test.csv`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/sealed_holdout_test.csv) was neither opened nor referenced.
- **Subsampling Integrity**: Exactly $N=500$ records were drawn from $X_{\text{train}}$ using `X_train.sample(n=500, random_state=42).reset_index(drop=True)`. The sampling is strictly deterministic and repeatable.
- **Legacy Artifact Preservation**: [`Mental_Health_Model.pkl`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/Mental_Health_Model.pkl) file size remains unchanged at exactly 25,694,899 bytes.

### 2.2 Verdict: PASS

---

## 3. SHAP Calculation Audit

### 3.1 Metric Formulations
In [`src/global_shap.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/src/global_shap.py#L88-L109), metrics across the $N=500$ evaluation sample instances are defined as follows:

1. **Mean Absolute SHAP ($I_f$)**:
   $$I_f = \frac{1}{N} \sum_{i=1}^{N} |\Phi_{i, f}|$$
   Calculated via `grouped_df.abs().mean(axis=0)`. Correctly quantifies overall magnitude of attribution.
2. **Mean Signed SHAP**:
   $$\overline{\Phi}_f = \frac{1}{N} \sum_{i=1}^{N} \Phi_{i, f}$$
   Calculated via `grouped_df.mean(axis=0)`. Quantifies the sample-wide net directional tendency of the model prediction relative to $\mathbb{E}[\hat{y}]$.
3. **Median Absolute SHAP**:
   $$\text{Median}(|\Phi_{1, f}|, \dots, |\Phi_{N, f}|)$$
   Calculated via `grouped_df.abs().median(axis=0)`. Resilient to outlier samples.
4. **Standard Deviation**:
   $$s_f = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (\Phi_{i, f} - \overline{\Phi}_f)^2}$$
   Calculated via `grouped_df.std(axis=0)`. Quantifies variance of individual attributions across diverse student profiles.
5. **Feature Ranking**:
   Features are sorted strictly descending by `mean_abs_shap`. Mean signed SHAP is **not** used to determine importance rank.

### 3.2 Non-Causal Attribution Safeguards
Mean signed SHAP is explicitly prevented from being interpreted as a causal effect or clinical diagnosis:
- The script and documentation repeatedly reinforce that SHAP values represent statistical attribution of the conditional expectation function $\mathbb{E}[\hat{y} \mid X]$.
- All generated plots include explicit footnotes: *"Note: SHAP values quantify statistical model attribution within the fitted response surface, not real-world causal effect sizes."*

### 3.3 Verdict: PASS

---

## 4. Relative Importance Calculation Audit

### 4.1 Investigation of Discrepancy
The user prompt noted that the displayed "Relative Share" percentages in [`docs/PHASE_3_STEP_2_GLOBAL_ANALYSIS.md`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/docs/PHASE_3_STEP_2_GLOBAL_ANALYSIS.md#L41-L55) did not equal each feature's mean absolute SHAP divided by the sum of all 12 mean absolute SHAP values.

An exhaustive audit of [`artifacts/explanations/global_importance.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_importance.json) and the documentation reveals:
1. **Serialized Artifact State**:
   - The JSON artifact [`global_importance.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_importance.json) **does not** contain a `relative_mass` or `relative_share` field. It stores purely unnormalized raw summary metrics (`mean_abs_shap`, `mean_signed_shap`, `median_abs_shap`, `std_shap`, `min_shap`, `max_shap`, `rank`).
2. **Actual Mathematical Sum**:
   $$\sum_{f=1}^{12} I_f = 0.754084 + 0.283828 + 0.056709 + 0.051374 + 0.046218 + 0.045367 + 0.030015 + 0.029700 + 0.028948 + 0.011504 + 0.010807 + 0.009681 = 1.358235$$
3. **Discrepancy Cause**:
   - In Table 3.1 of `docs/PHASE_3_STEP_2_GLOBAL_ANALYSIS.md`, the displayed percentages were calculated against an interim denominator of $\approx 1.3069$ during drafting (yielding $0.7541 / 1.3069 = 57.7\%$, $0.2838 / 1.3069 = 21.7\%$, etc.).
   - As a result, the reported percentages sum to **103.8%** rather than 100.0%.

### 4.2 Exact Mathematical Reconciliation Table

| Feature | Mean $\| \text{SHAP} \|$ | Documented Share (Erroneous) | True Relative Share ($\frac{I_f}{1.358235}$) | Discrepancy |
|:---|:---:|:---:|:---:|:---:|
| `Avg_Daily_Usage_Hours` | 0.7541 | 57.7% | **55.52%** | -2.18% |
| `Sleep_Hours_Per_Night` | 0.2838 | 21.7% | **20.90%** | -0.80% |
| `Daily_Unlocks` | 0.0567 | 4.3% | **4.18%** | -0.12% |
| `Purpose_Of_Use` | 0.0514 | 3.9% | **3.78%** | -0.12% |
| `Country` | 0.0462 | 3.5% | **3.40%** | -0.10% |
| `Most_Used_Platform` | 0.0454 | 3.5% | **3.34%** | -0.16% |
| `Physical_Activity_Hours` | 0.0300 | 2.3% | **2.21%** | -0.09% |
| `Study_Hours` | 0.0297 | 2.3% | **2.19%** | -0.11% |
| `Age` | 0.0289 | 2.2% | **2.13%** | -0.07% |
| `Academic_Level` | 0.0115 | 0.9% | **0.85%** | -0.05% |
| `Stress_Level` | 0.0108 | 0.8% | **0.80%** | 0.00% |
| `Gender` | 0.0097 | 0.7% | **0.71%** | +0.01% |
| **Total** | **1.3582** | **103.8%** | **100.00%** | **+3.80%** |

### 4.3 Verdict: WARNING (Documentation Correction Required)

---

## 5. Original-Feature Aggregation Interpretation Audit

### 5.1 Dynamic Mapping & Aggregation Verification
- **Transformed Feature Mapping**: All 38 output features from the ColumnTransformer map unambiguously to exactly one of the 12 original schema predictors:
  - `standard_numeric` (Age, Avg_Daily_Usage_Hours, Daily_Unlocks, Sleep_Hours_Per_Night): 4 features $\rightarrow$ 4 parents (1 each).
  - `skewed` (Study_Hours): 1 feature $\rightarrow$ 1 parent.
  - `physical` (Physical_Activity_Hours): 1 feature $\rightarrow$ 1 parent.
  - `ordinal` (Stress_Level): 1 feature $\rightarrow$ 1 parent.
  - `nominal` (Gender, Academic_Level, Purpose_Of_Use, Most_Used_Platform): 21 one-hot features $\rightarrow$ 4 parents (2, 3, 4, 12 respectively).
  - `country` (Country): 10 one-hot features $\rightarrow$ 1 parent.
- **Completeness**: $4 + 1 + 1 + 1 + 21 + 10 = 38$ transformed features. Disjoint and non-overlapping.
- **Additive Invariance**: For every record $i$, $\sum_{f=1}^{12} \Phi_{i, f} = \sum_{j=1}^{38} \phi_{i, j}$.

### 5.2 Game-Theoretic Clarification Requirement
- **Terminology Observation**:
  The report accurately states that transformed SHAP values are summed into parent feature groups. However, game-theoretically, computing TreeSHAP on 38 transformed features and summing attributions into 12 parent buckets produces an **aggregated transformed attribution**, which preserves the additive decomposition of $\hat{y}_i$.
  This is mathematically distinct from running a hypothetical 12-player Shapley game where categorical features are perturbed as unitary atomic blocks.
- **Recommendation**:
  A clear methodological note must be added to Section 3 of `docs/PHASE_3_STEP_2_GLOBAL_ANALYSIS.md` explicitly distinguishing aggregated transformed attributions from a 12-player atomic Shapley game.

### 5.3 Verdict: WARNING (Clarification Required)

---

## 6. Visualization Audit

### 6.1 Figure Integrity & Attributes
All 4 generated visualizations in [`artifacts/explanations/`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations) were inspected via image metadata and script examination:

| Visualization | Dimensions | DPI | Color Space | Elements Verified |
|:---|:---:|:---:|:---:|:---|
| [`global_shap_summary.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_summary.png) | $3000 \times 1950$ | 300 | RGBA | All 12 original features; sorted descending from top; exact numeric bar annotations; non-truncated axis scale; clear unit label. |
| [`global_shap_beeswarm.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_beeswarm.png) | $3300 \times 2250$ | 300 | RGBA | 500 individual points per feature; vertical jitter; continuous features mapped to coolwarm; categorical features mapped to slate grey; zero-line reference. |
| [`transformed_shap_beeswarm.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/transformed_shap_beeswarm.png) | $3300 \times 2550$ | 300 | RGBA | Top 20 transformed features; stripped pipeline prefixes; binary color shifts on one-hot indicators; zero-line reference. |
| [`global_shap_distribution.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_distribution.png) | $3300 \times 2250$ | 300 | RGBA | Dual horizontal violins (KDE) + boxplots (IQR, gold median line, fliers); zero-line reference; 12 features. |

### 6.2 Verdict: PASS

---

## 7. Dependency & Reproducibility Audit

### 7.1 Installed vs. Pinned Dependencies
- **Installed Runtime Environment**:
  - `python == 3.13.3`
  - `shap == 0.52.0` (installed in `.venv`)
  - `matplotlib == 3.11.2` (installed in `.venv`)
- **Inspection of Requirements Files**:
  - [`requirements.txt`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/requirements.txt): Lists `fastapi`, `uvicorn`, `pydantic`, `joblib`, `pandas`, `scikit-learn`, `numpy`, `scipy`. Neither `shap` nor `matplotlib` is listed.
  - [`requirements-dev.txt`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/requirements-dev.txt): Lists `-r requirements.txt`, `pytest`, `httpx`. Neither `shap` nor `matplotlib` is listed.
- **Reproducibility Risk**:
  A fresh deployment or container build using `pip install -r requirements.txt` or `pip install -r requirements-dev.txt` will fail to run `src/explain.py` or `src/global_shap.py`.
- **Metadata Traceability**:
  [`artifacts/explanations/global_analysis_metadata.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_analysis_metadata.json) correctly captures Python version, SHAP version, OS platform, random seed, runtime (85.81s), and UTC timestamp.

### 7.2 Verdict: WARNING (Pinning Required)

---

## 8. Automated Test Coverage Audit

### 8.1 Existing Test Suite
`tests/test_explain.py` contains 12 unit tests verifying:
- Explainer initialization and non-hardcoded dynamic discovery.
- Feature mapping completeness and disjointness.
- Single-record and batch efficiency additivity ($\text{base} + \sum \text{SHAP} = \hat{y}$).
- Error handling on missing columns / wrong data types.
- Determinism on identical inputs.
- Legacy model size preservation (25,694,899 bytes).
- Holdout row count (1,501 rows).

### 8.2 Missing Tests for Step 3.2
The following coverage gaps exist:
1. **Sampling Determinism Test**: No automated test directly asserts that `get_deterministic_training_sample(500, 42)` returns exactly 500 rows from $X_{\text{train}}$ without accessing holdout rows.
2. **Global SHAP Artifact Schema Test**: No automated test verifies the existence, schema, descending rank order, and finite value constraints (`np.isfinite`) of:
   - `artifacts/explanations/global_importance.json`
   - `artifacts/explanations/transformed_importance.json`
   - `artifacts/explanations/global_analysis_metadata.json`
3. **Visualization Artifacts Test**: No automated test verifies that all 4 PNG files exist and have non-zero file sizes.

### 8.3 Verdict: WARNING (Missing Step 3.2 Tests)

---

## 9. Discrepancies and Required Corrections

The following three non-destructive corrections are documented and ready for implementation upon user authorization:

1. **Correction 1: Documentation Table 3.1 Reconciliation**
   - File: `docs/PHASE_3_STEP_2_GLOBAL_ANALYSIS.md`
   - Action: Update the "Relative Share" column with exact proportions normalized by the true sum $\sum I_f = 1.358235$:
     - Usage: 55.52%, Sleep: 20.90%, Unlocks: 4.18%, Purpose: 3.78%, Country: 3.40%, Platform: 3.34%, Activity: 2.21%, Study: 2.19%, Age: 2.13%, Academic: 0.85%, Stress: 0.80%, Gender: 0.71% (Total: 100.00%).
   - Action: Add methodological text clarifying the game-theoretic distinction between aggregated transformed attributions and an atomic 12-player Shapley game.

2. **Correction 2: Requirements Manifest Update**
   - Files: `requirements.txt` and/or `requirements-dev.txt`
   - Action: Add pinned entries for `shap==0.52.0` and `matplotlib==3.11.2` so that environments can be reproduced without missing modules.

3. **Correction 3: Automated Test Suite Expansion**
   - File: `tests/test_explain.py` (or a dedicated `tests/test_global_shap.py`)
   - Action: Add test cases validating `get_deterministic_training_sample()`, verifying the structure and ranking of `global_importance.json` and `transformed_importance.json`, asserting all attributions are finite, and checking that all 4 visualization PNG files exist and are non-empty.

---

## 10. Final Readiness Verdict

In accordance with strict audit instructions, no files were modified and no tests were added during this review.

```
================================================================================
AUDIT SUMMARY:
  - Training Sample & Data Governance: PASS
  - SHAP Calculation & Efficiency:     PASS
  - Relative Importance Calculation:   WARNING (Table 3.1 divisor error)
  - Game-Theoretic Interpretation:     WARNING (Clarification needed)
  - Visualization Generation:          PASS
  - Dependency & Reproducibility:      WARNING (shap & matplotlib unpinned)
  - Automated Test Coverage:           WARNING (Artifact regression tests missing)
================================================================================

FINAL VERDICT:
CORRECTIONS REQUIRED
```

Awaiting user approval to implement the three identified corrections.

