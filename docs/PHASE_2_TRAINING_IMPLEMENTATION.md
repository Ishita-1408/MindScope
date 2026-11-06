# MindScope — Phase 2: Model Training & Nested Cross-Validation Report (Step 2.2)
**Document Version:** 1.0.0  
**Date:** 2026-11-06  
**Status:** Completed & Empirically Validated (All 47 Tests Passing)  
**Project:** MindScope — Student Mental Health Analytics & Predictive Intelligence Platform  
**Target Environment:** Python 3.13.3, Scikit-Learn 1.8.0, Pandas 3.0.2, NumPy 2.4.4 inside `.venv`  

---

## 1. Executive Summary

In accordance with Phase 2, Step 2.2 of the MindScope engineering specification, the complete **Model Experimentation and Nested Cross-Validation Protocol** has been executed. This step rigorously evaluated candidate regression architectures, feature schema variants, and noise sensitivity treatments entirely within the training data ($N = 3,498$), without consulting the sealed holdout test set ($N = 1,500$) or modifying production deployment artifacts.

### Key Methodological Safeguards Maintained
- **Strict Holdout Sealing:** The 30% holdout split ($N=1,500$) was partitioned using `random_state=42` and persisted to [`artifacts/sealed_holdout_test.csv`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/sealed_holdout_test.csv). It was **not evaluated or consulted** during model selection or hyperparameter tuning.
- **Nested Cross-Validation:** Implemented a true 5-fold outer and 5-fold inner cross-validation loop. Inner CV selected the complete modeling configuration (algorithm family, hyperparameters, feature set, activity treatment); outer CV provided the unbiased out-of-sample estimate of that automated selection procedure.
- **Production Artifact Safety:** In strict adherence to instructions, `Mental_Health_Model.pkl` was **not overwritten or modified**. Production refitting and model packaging are deferred to Step 2.3.
- **Immutability of Codebase & Dataset:** No modifications were made to `ML_Project.ipynb`, the raw CSV dataset, `main.py`, or frontend files.

---

## 2. Models & Experimental Configurations Evaluated

A comprehensive matrix of 9 distinct configurations was evaluated across identical outer folds:

| Config ID | Estimator Family | Feature Schema | Activity Noise Strategy | Search Space & Hyperparameters |
| :--- | :--- | :--- | :--- | :--- |
| **`M1_OLS_Full_Clip`** | `LinearRegression` | Full (12 features) | Bounded Clip ($\max(x, 0)$) | Analytical OLS solution (zero tuning). |
| **`M2a_Ridge_Full_Clip`** | `Ridge` | Full (12 features) | Bounded Clip ($\max(x, 0)$) | $\alpha \in [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]$ |
| **`M2b_Ridge_Reduced_Clip`** | `Ridge` | Reduced (11 features) | Bounded Clip ($\max(x, 0)$) | $\alpha \in [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]$ |
| **`M3a_HistGBR_Full_Clip`** | `HistGradientBoostingRegressor` | Full (12 features) | Bounded Clip ($\max(x, 0)$) | $\text{lr} \in [0.03, 0.05, 0.1]$, $\text{iter} \in [100, 150]$, $\text{depth} \in [4, 6, 8]$, $\text{leaf} \in [15, 25]$, $\text{L2} \in [0.0, 1.0, 5.0]$ |
| **`M3b_HistGBR_Full_NaNImpute`** | `HistGradientBoostingRegressor` | Full (12 features) | Pipeline NaN-Median Impute | Identical HistGBR search space. |
| **`M3c_HistGBR_Reduced_Clip`** | `HistGradientBoostingRegressor` | Reduced (11 features) | Bounded Clip ($\max(x, 0)$) | Identical HistGBR search space. |
| **`M4a_RF_Full_Clip`** | `RandomForestRegressor` | Full (12 features) | Bounded Clip ($\max(x, 0)$) | $n \in [100, 150]$, $\text{depth} \in [10, 15, \text{None}]$, $\text{split} \in [2, 5]$, $\text{leaf} \in [2, 4]$ |
| **`M4b_RF_Reduced_Clip`** | `RandomForestRegressor` | Reduced (11 features) | Bounded Clip ($\max(x, 0)$) | Identical Random Forest search space. |
| **`M4c_RF_Full_Exclude`** | `RandomForestRegressor` | Full (12 features) | Row Exclusion ($N_{\text{train}}=3,491$) | Identical Random Forest search space. |

---

## 3. Nested Cross-Validation Empirical Results

### 3.1 Unbiased Generalization of the Complete Selection Procedure
Across the 5 outer validation folds, the automated inner-selection procedure chose:
- **Fold 1:** `M4c_RF_Full_Exclude` (Outer Val $R^2 = 0.8222$, $\text{MAE} = 0.3930$, $\text{RMSE} = 0.5270$)
- **Fold 2:** `M4a_RF_Full_Clip` (Outer Val $R^2 = 0.8539$, $\text{MAE} = 0.3559$, $\text{RMSE} = 0.4801$)
- **Fold 3:** `M4a_RF_Full_Clip` (Outer Val $R^2 = 0.8536$, $\text{MAE} = 0.3569$, $\text{RMSE} = 0.4803$)
- **Fold 4:** `M3a_HistGBR_Full_Clip` (Outer Val $R^2 = 0.8383$, $\text{MAE} = 0.3810$, $\text{RMSE} = 0.4925$)
- **Fold 5:** `M4c_RF_Full_Exclude` (Outer Val $R^2 = 0.8342$, $\text{MAE} = 0.3896$, $\text{RMSE} = 0.5284$)

#### Aggregate Outer-Fold Performance Metrics (Unbiased Estimate)

| Evaluation Metric | Mean Out-of-Sample Value | Outer Fold Std Dev ($\sigma$) | Scientific Interpretation |
| :--- | :---: | :---: | :--- |
| **Coefficient of Determination ($R^2$)** | **$0.8404$** | $\pm 0.0121$ | Explains $84.04\%$ of score variance out-of-sample under strict fold isolation. |
| **Mean Absolute Error (MAE)** | **$0.3753$** | $\pm 0.0159$ | Average prediction error is $0.375$ score points on a 10-point scale. |
| **Root Mean Squared Error (RMSE)** | **$0.5017$** | $\pm 0.0217$ | Bounded penalty for outlier predictions. |
| **Tolerance $\text{Tol}_{\pm 0.1}$** | **$20.47\%$** | $\pm 1.62\%$ | $20.47\%$ of predictions fall within $\pm 0.10$ points of recorded score. |
| **Tolerance $\text{Tol}_{\pm 0.2}$** | **$37.51\%$** | $\pm 2.14\%$ | $37.51\%$ of predictions fall within $\pm 0.20$ points of recorded score. |
| **Overfitting Gap ($R^2_{\text{train}} - R^2_{\text{val}}$)** | **$0.1141$** | $\pm 0.0185$ | Substantial improvement over legacy notebook ($0.1032$ vs unconstrained $0.9808$ training fit). |

### 3.2 Individual Configuration Benchmark Matrix
Every candidate configuration was systematically evaluated across all 5 outer folds to produce an uncompromised comparative matrix:

| Rank | Configuration ID | Algorithm | Features | Activity Strategy | CV $R^2$ ($\mu \pm \sigma$) | CV MAE | CV RMSE | Overfit Gap | Mean Fit Time |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | `M4a_RF_Full_Clip` | `RandomForestRegressor` | Full (12) | Bounded Clip | **$0.8392 \pm 0.0128$** | $0.3764$ | $0.5034$ | $0.1151$ | $11.37\text{s}$ |
| **2** | `M4c_RF_Full_Exclude` | `RandomForestRegressor` | Full (12) | Exclude Rows | **$0.8386 \pm 0.0123$** | $0.3764$ | $0.5042$ | $0.1157$ | $10.71\text{s}$ |
| **3** | `M4b_RF_Reduced_Clip` | `RandomForestRegressor` | Reduced (11) | Bounded Clip | **$0.8369 \pm 0.0125$** | $0.3796$ | $0.5070$ | $0.1161$ | $10.45\text{s}$ |
| **4** | `M3a_HistGBR_Full_Clip` | `HistGradientBoosting` | Full (12) | Bounded Clip | **$0.8302 \pm 0.0104$** | $0.3966$ | $0.5177$ | $0.0907$ | $4.60\text{s}$ |
| **5** | `M3b_HistGBR_Full_NaNImpute`| `HistGradientBoosting` | Full (12) | NaN-Impute | **$0.8289 \pm 0.0102$** | $0.3967$ | $0.5197$ | $0.0909$ | $5.28\text{s}$ |
| **6** | `M3c_HistGBR_Reduced_Clip` | `HistGradientBoosting` | Reduced (11) | Bounded Clip | **$0.8283 \pm 0.0112$** | $0.3992$ | $0.5205$ | $0.0901$ | $4.98\text{s}$ |
| **7** | `M2a_Ridge_Full_Clip` | `Ridge` | Full (12) | Bounded Clip | **$0.7177 \pm 0.0168$** | $0.5243$ | $0.6674$ | $0.0075$ | $0.61\text{s}$ |
| **8** | `M1_OLS_Full_Clip` | `LinearRegression` | Full (12) | Bounded Clip | **$0.7177 \pm 0.0167$** | $0.5240$ | $0.6674$ | $0.0076$ | $0.05\text{s}$ |
| **9** | `M2b_Ridge_Reduced_Clip` | `Ridge` | Reduced (11) | Bounded Clip | **$0.7177 \pm 0.0167$** | $0.5241$ | $0.6675$ | $0.0075$ | $0.42\text{s}$ |

---

## 4. Key Scientific & Engineering Insights

### 4.1 Negative Physical Activity Sensitivity Analysis
The empirical comparison across all three negative activity treatments demonstrates that their performance differences are statistically negligible:
- **Bounded Clipping (`M4a_RF_Full_Clip`):** $R^2 = 0.8392 \pm 0.0128$, $\text{MAE} = 0.3764$.
- **Observation Exclusion (`M4c_RF_Full_Exclude`):** $R^2 = 0.8386 \pm 0.0123$, $\text{MAE} = 0.3764$ ($\Delta R^2 = 0.0006$).
- **Pipeline NaN Imputation (`M3b` vs `M3a`):** $\Delta R^2 = 0.0013$.

**Conclusion:** The 10 negative observations ($0.2\%$ of the cohort) exert zero disruptive leverage on model parameters. **Bounded clipping (`activity_strategy='clip'`) is methodologically superior** because it preserves 100% of student observations, imposes zero imputation overhead, and directly honors the physiological bound ($\ge 0.0$).

### 4.2 Feature Redundancy: Full vs. Reduced Schema
Comparing the Full Schema (12 features) against the Reduced Schema (omitting `Purpose_Of_Use` due to 93.2% deterministic coupling with `Most_Used_Platform`):
- **Random Forest:** Full $R^2 = 0.8392$ vs. Reduced $R^2 = 0.8369$ ($\Delta R^2 = 0.0023$, $\Delta \text{MAE} = 0.0032$).
- **HistGradientBoosting:** Full $R^2 = 0.8302$ vs. Reduced $R^2 = 0.8283$ ($\Delta R^2 = 0.0019$).
- **Ridge / OLS:** Full $R^2 = 0.7177$ vs. Reduced $R^2 = 0.7177$ ($\Delta R^2 < 0.0001$).

**Conclusion:** Dropping `Purpose_Of_Use` results in a minuscule out-of-sample penalty of only $0.0023$ $R^2$, while removing exact one-hot collinearity and eliminating split competition in tree structures. Both configurations remain available in the codebase, with the Full configuration providing the maximum predictive performance ($R^2 = 0.8392$).

### 4.3 Estimator Trade-offs: Random Forest vs. HistGradientBoosting vs. Ridge
- **Non-Linear Ensembles vs. Linear Models:** Non-linear tree ensembles achieve a $\sim 0.12$ higher $R^2$ ($0.8392$ vs $0.7177$) and a $\sim 0.15$ lower MAE ($0.3764$ vs $0.5243$) than linear models, confirming strong non-linear interactions between sleep, screen time, and stress.
- **Random Forest vs. HistGradientBoosting:**
  - Random Forest achieves the highest out-of-sample accuracy ($R^2 = 0.8392$, $\text{MAE} = 0.3764$).
  - HistGradientBoosting achieves a noticeably lower overfitting gap ($0.0907$ vs $0.1151$) and trains in less than half the time ($4.60\text{s}$ vs $11.37\text{s}$).

---

## 5. Computational Cost & Experiment Artifacts

- **Total Execution Time:** $471.02$ seconds (~$7.85$ minutes) for the full 5-fold outer $\times$ 5-fold inner nested CV across 9 candidate configurations.
- **Failed or Skipped Experiments:** **Zero.** All 9 configurations executed cleanly without runtime exceptions or numerical instability.
- **Dense Matrix Memory Verification:** Maximum in-memory feature matrix size per fold was $2,798 \times 39 \times 8 \text{ bytes} \approx 0.87 \text{ MB}$, completely avoiding memory pressure.

### Structured Experiment Outputs Persisted:
1. [`artifacts/experiments/nested_cv_summary.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/experiments/nested_cv_summary.json): Complete outer-fold metrics and generalization statistics.
2. [`artifacts/experiments/all_configurations_benchmarks.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/experiments/all_configurations_benchmarks.json): Mean and standard deviation for all 9 configurations across all 5 outer folds.
3. [`artifacts/experiments/fold_level_metrics.csv`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/experiments/fold_level_metrics.csv): Granular fold-by-fold metrics (45 rows).
4. [`artifacts/sealed_holdout_test.csv`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/sealed_holdout_test.csv): Sealed 1,500-sample test set (untouched).

---

## 6. Automated Test Suite Verification

A dedicated test suite of 8 verification tests was implemented in [`tests/test_training.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/tests/test_training.py) to validate methodological safeguards:

| Test Name | Verified Safety Property | Status |
| :--- | :--- | :--- |
| `test_partition_sizes_and_preservation` | Verifies exact partition sizes (3,498 train, 1,500 sealed holdout) and zero index overlap. | **PASSED** |
| `test_deterministic_splitting` | Verifies reproducibility across identical `random_state=42` splits. | **PASSED** |
| `test_sealed_holdout_file_integrity` | Verifies sealed holdout file exists, has 1,500 records, and contains all 13 columns. | **PASSED** |
| `test_calculate_metrics_correctness` | Verifies unrounded float metrics and accurate percentage calculations for $\text{Tol}_{\pm 0.1}$ and $\text{Tol}_{\pm 0.2}$. | **PASSED** |
| `test_nested_cv_inner_outer_separation` | Verifies that inner CV indices are strict subsets of outer training splits and never leak into outer validation splits. | **PASSED** |
| `test_experiment_configuration_matrix_completeness`| Verifies all 4 algorithm families, full/reduced schemas, and activity strategies are represented. | **PASSED** |
| `test_structured_experiment_artifacts_exist` | Verifies that JSON and CSV experiment outputs exist and conform to schemas. | **PASSED** |
| `test_production_model_artifact_was_not_overwritten`| Verifies `Mental_Health_Model.pkl` maintains legacy size (~24.5 MB) and was not modified. | **PASSED** |

### Complete Repository Test Suite:
Executing `pytest tests -v` confirmed that all **47 automated tests** in the project pass with a **100% success rate**.

---

## 7. Methodological Limitations & Recommendations for Step 2.3

1. **Unexplained Variance ($1 - R^2 \approx 16\%$):** Even with optimal non-linear tree ensembles, out-of-sample $R^2$ plateaus at $0.8404 \pm 0.0121$, leaving $\sim 16\%$ unexplained variance. As established in the Target Investigation, this reflects stochastic or unmeasured behavioral variance in the dataset.
2. **Production Model Selection Recommendation:**
   - **Recommended Winner:** `RandomForestRegressor (Full Schema, Bounded Clip)` achieved the highest out-of-sample generalization ($R^2 = 0.8392$, $\text{MAE} = 0.3764$, $\text{RMSE} = 0.5034$).
   - **Alternative Contender:** `HistGradientBoostingRegressor (Full Schema, Bounded Clip)` offers competitive accuracy ($R^2 = 0.8302$, $\text{MAE} = 0.3966$), faster inference, and a substantially smaller disk footprint ($< 5\text{MB}$ vs. $25\text{MB}$).
3. **Transition to Step 2.3:**
   - In Step 2.3, the selected winning architecture will be refit once on all 3,498 training samples and serialized to `Mental_Health_Model.pkl` under native `scikit-learn==1.8.0`.
   - The sealed holdout test set (`artifacts/sealed_holdout_test.csv`, $N=1,500$) will be evaluated **exactly once** to establish definitive out-of-sample test benchmarks.

---
*End of Phase 2 Step 2.2 Model Training Implementation Report. Awaiting user review and authorization before proceeding to Step 2.3 (Final Model Selection, Holdout Evaluation & Packaging).*
