# MindScope — Phase 2: Machine Learning Pipeline Modernisation & Retraining Design
**Document Version:** 1.1.0  
**Date:** 2026-11-06  
**Status:** Architecture Specification — Refined with Methodological Controls  
**Project:** MindScope — Student Mental Health Analytics & Predictive Intelligence Platform  
**Target Environment:** Python 3.13.3, Scikit-Learn 1.8.0, Pandas 3.0.2, NumPy 2.4.4 inside `.venv`  

---

## 1. Executive Summary

This specification establishes the engineering and methodological blueprint for **Phase 2: Machine Learning Pipeline Modernisation & Retraining**. Phase 2 transitions MindScope from an unvalidated experimental script into a rigorous, leakage-free, and reproducible machine learning system.

The empirical audit of the original notebook (`ML_Project.ipynb`) identified substantial structural deficiencies:
1. **Preprocessing Data Leakage:** Categorical cardinality reduction for `Country` was computed across the entire 5,000-record dataset prior to train-test splitting.
2. **Broken Pipeline Encapsulation:** Country grouping was performed via an external pandas `.apply()` function rather than inside a Scikit-Learn transformer. Consequently, the saved pipeline cannot accept raw `Country` data, forcing external consumers (`main.py`) to re-implement preprocessing steps manually.
3. **Discrepancies Between Documentation and Code:** 
   - Markdown claimed zero duplicates, while code executed `df.drop_duplicates()` (removing 2 duplicate pairs / 4 rows).
   - Markdown claimed an 80/20 train-test split, while code executed 70/30 (`test_size=0.30`).
   - Markdown claimed `SimpleImputer` safeguards in every pipeline branch, but code omitted imputers entirely.
   - Code performed hyperparameter tuning on Random Forest, but accidentally serialized the **unconstrained, default Random Forest** (`n_estimators=100, max_depth=None`), locking in severe training overfitting ($R^2_{\text{train}} = 0.9808$).
4. **Environment Drift:** The model was serialized under development build `scikit-learn==1.9.0` (installed via Colab cell 68), triggering 122 runtime warnings and fatal `AttributeError` exceptions when inspected in the pinned production environment (`scikit-learn==1.8.0`).

This refined design incorporates strict methodological controls:
- A multi-strategy sensitivity experiment for handling invalid negative physical activity values.
- A fully encapsulated, Scikit-Learn compliant `CountryTopKEncoder` that avoids global data contamination.
- Clarification of observed deterministic platform–purpose couplings across the top 7 platforms (~93.2% of data) and a structured redundancy evaluation.
- An unbiased nested cross-validation architecture where inner CV selects the complete modeling configuration (algorithm, preprocessing, features, hyperparameters) and outer CV estimates the generalization of that complete selection process.
- Dense feature matrix guarantees for compatibility with `HistGradientBoostingRegressor`.

---

## 2. Existing Workflow Audit (`ML_Project.ipynb`)

A comprehensive cell-by-cell inspection of `ML_Project.ipynb` reveals the exact execution trace and operational boundaries:

### 2.1 Workflow Inventory Table

| Phase / Cell | Operation Performed | Implementation Mechanism | Methodological Finding / Flaw |
| :--- | :--- | :--- | :--- |
| **Cell 4** | Data Ingestion | `pd.read_csv('Student Social Media And Mental Health Impact.csv')` | Ingests 5,000 records, 13 raw columns. |
| **Cell 8 vs 36** | Duplicate Handling | Cell 8 markdown: *"Zero duplicate rows."*<br>Cell 36 code: `df = df.drop_duplicates()` | **Documentation Conflict:** Dataset actually contains 2 duplicate pairs (4 rows total). Code drops 2 rows, resulting in 4,998 unique records. |
| **Cell 36** | Outlier Treatment | `df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0)` | **Global Mutation:** Exactly 10 negative records ($-0.1$ to $-0.4$) are clipped globally across `df` before splitting. |
| **Cell 41–43** | Feature Engineering | `top_countries = df['Country'].value_counts().index[:10].tolist()`<br>`df['Grouped_country'] = df['Country'].apply(group_countries)` | **Critical Data Leakage:** High-cardinality grouping is fit on the full dataset before train-test partitioning. Pipeline does not accept raw `Country`. |
| **Cell 48 vs 49** | Data Partitioning | Cell 48 markdown: *"We'll use an 80/20 split here"*<br>Cell 49 code: `train_test_split(..., test_size=0.30, random_state=42)` | **Documentation Conflict:** Split ratio implemented was 70% train ($N=3,498$) and 30% holdout test ($N=1,500$), contradicting markdown text. |
| **Cell 50 vs 51** | Pipeline Imputation | Cell 50 markdown: *"We include a SimpleImputer in every branch..."*<br>Cell 51 code: `SimpleImputer` completely absent. | **Omission Defect:** Preprocessing pipeline contains zero imputation logic. Real-world payloads with null values fail immediately. |
| **Cell 51** | Preprocessing Setup | `ColumnTransformer` with `FunctionTransformer(np.log1p)`, `StandardScaler`, `OrdinalEncoder`, `OneHotEncoder(handle_unknown='ignore')`. | Encodes `Grouped_country` (11 categories) instead of `Country`. Fails to encapsulate country grouping. |
| **Cell 56** | Baseline Model | `LinearRegression()` fit inside pipeline on `X_train`. | Evaluated on `X_test`: $R^2 = 0.7398$, $\text{MAE} = 0.5362$, $\text{RMSE} = 0.6760$. |
| **Cell 58** | Default Ensemble | `RandomForestRegressor()` fit inside pipeline on `X_train`. | Severe overfitting: $R^2_{\text{train}} = 0.9808$ vs. $R^2_{\text{test}} = 0.8776$ ($\Delta R^2 = 0.1032$). |
| **Cell 60** | Hyperparameter Tuning | `RandomizedSearchCV(n_iter=15, cv=5, scoring='r2', random_state=42)` | Best config: `n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2`. Achieved $R^2_{\text{test}} = 0.8650$, $\text{MAE} = 0.3689$. |
| **Cell 66** | Artifact Serialization | `joblib.dump(rf_pipeline, 'Mental_Health_Model.pkl')` | **Critical Artifact Blunder:** Author serialized `rf_pipeline` (**default, untuned model**), NOT `rf_best_pipeline` (`random_search.best_estimator_`). |
| **Cell 68** | Environment Mutation | `!pip install --upgrade scikit-learn==1.9.0` | Upgraded to unreleased dev build in Colab, creating forward serialization drift against standard releases. |

### 2.2 Operations Performed Before vs. After Partitioning

```
ORIGINAL WORKFLOW (LEAKAGE & MANUAL STEPS):
Raw CSV (5000) 
  │
  ├─► df.drop_duplicates()                     [GLOBAL MUTATION - 4998 rows]
  ├─► df['Physical_Activity_Hours'].clip(0)    [GLOBAL MUTATION - 10 values]
  ├─► top_countries = df['Country'].counts()   [DATA LEAKAGE - Global frequency]
  ├─► df['Grouped_country'] = apply(...)       [MANUAL COLUMN ADDITION]
  │
  ▼
train_test_split(test_size=0.30)
  ├── X_train (3498) ──► ColumnTransformer.fit() ──► Fit Model
  └── X_test  (1500) ──► ColumnTransformer.transform() ──► Evaluate
```

---

## 3. Data Leakage Assessment & Methodological Deficiencies

### 3.1 Country Grouping Information Leakage
- **Mechanism:** In cell 41, `df['Country'].value_counts().index[:10]` calculates country frequencies across all 5,000 rows (including the holdout test set).
- **Theoretical Concern:** The holdout split informs which categories are deemed "frequent." If country distributions shift or test folds have localized spikes, the model benefits from test-set demographic intelligence.
- **Architectural Consequence:** Because country grouping occurred outside Scikit-Learn, the exported pipeline was incomplete. Serving the model in `main.py` required writing custom application-layer preprocessing logic to mirror the notebook's manual script.
- **Remediation:** Implement a custom Scikit-Learn transformer (`CountryTopKEncoder`) that learns the top 10 categories strictly during `fit(X_train)` and maps unseen or infrequent countries to `'Other'` during `transform()`.

### 3.2 Global Negative Value Handling
- **Mechanism:** `df['Physical_Activity_Hours'].clip(lower=0)` altered 10 records in the raw dataframe prior to splitting.
- **Theoretical Concern:** Modifying features prior to splitting breaches data pipeline encapsulation and imposes an unverified assumption without empirical comparison.
- **Remediation:** Keep the raw dataset completely immutable. Evaluate negative activity values via a structured sensitivity experiment within the training pipeline.

### 3.3 Optimism Bias in Model Selection
- **Mechanism:** In cell 60, `RandomizedSearchCV` was fit on `X_train` using 5-fold CV to select hyperparameters. However, model comparison and selection decisions were confirmed by evaluating performance directly on `X_test`.
- **Theoretical Concern:** When the test set is used repeatedly to compare models (Linear Regression vs. Default RF vs. Tuned RF), test-set variance informs model selection, producing optimistically biased performance estimates.
- **Remediation:** Implement **nested cross-validation** (outer 5-fold CV for performance estimation; inner 5-fold CV for hyperparameter tuning) entirely within the training split ($N=3,498$). The holdout test set ($N=1,500$) is locked until final evaluation.

---

## 4. Modern Preprocessing Pipeline Architecture

MindScope Phase 2 introduces an encapsulated, robust Scikit-Learn pipeline. The pipeline accepts raw input DataFrames containing the 12 original predictors and directly outputs model-ready numerical feature arrays.

```
MULTI-BRANCH ENCAPSULATED PREPROCESSING PIPELINE:
Raw Input DataFrame (12 Features: Age, Gender, Country, Academic_Level, ...)
  │
  ├── [Country] ─────────────────► CountryTopKEncoder(k=10) ──► SimpleImputer('most_frequent') ──► OneHotEncoder(sparse_output=False)
  │
  ├── [Study_Hours] ────────────► SimpleImputer('median') ────► Log1pTransform() ──────────────► StandardScaler()
  │
  ├── [Physical_Activity_Hours] ─► PhysicalActivityHandler ──► SimpleImputer('median') ────────► StandardScaler()
  │                               (Clip / NaN-Impute / Exclude)
  ├── [Age, Usage, Unlocks, Sleep]►SimpleImputer('median') ─────────────────────────────────────► StandardScaler()
  │
  ├── [Stress_Level] ────────────► SimpleImputer('most_frequent') ──────────────────────────────► OrdinalEncoder([Low..Very High])
  │
  └── [Gender, Acad., Plat., Purp.]SimpleImputer('most_frequent') ──────────────────────────────► OneHotEncoder(sparse_output=False)
```

### 4.1 Custom Transformer Specifications

#### 1. `CountryTopKEncoder`
To guarantee full compatibility with `sklearn.base.clone`, grid searches, and fold isolation, `CountryTopKEncoder` follows Scikit-Learn estimator conventions:
- Strict `__init__` signature with primitive default arguments (`top_k=10, other_label="Other"`).
- Learns category frequencies strictly from the training slice during `fit()`.
- Explicitly recognizes if `'Other'` already exists in the raw data, ensuring infrequent categories cleanly aggregate into `'Other'` without generating redundant multi-label representations.
- Consistent transformation of unseen categories during inference.
- Complete independence from global dataset statistics.

```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class CountryTopKEncoder(BaseEstimator, TransformerMixin):
    """
    Leakage-free category grouping transformer for high-cardinality country values.
    Learns the top-K most frequent countries strictly during fit() on training data,
    and maps any infrequent or unseen category to `other_label` during transform().
    """
    def __init__(self, top_k=10, other_label="Other"):
        self.top_k = top_k
        self.other_label = other_label
        # Attributes ending with trailing underscore are set during fit()
        self.top_countries_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0].dropna()
        elif isinstance(X, pd.Series):
            series = X.dropna()
        else:
            series = pd.Series(np.asarray(X).ravel()).dropna()

        # Learn top K strictly from the training slice
        counts = series.value_counts()
        self.top_countries_ = counts.index[:self.top_k].tolist()
        return self

    def transform(self, X):
        if self.top_countries_ is None:
            raise RuntimeError("Transformer has not been fitted yet.")
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            series = X
        else:
            series = pd.Series(np.asarray(X).ravel())

        top_set = set(self.top_countries_)
        # Preserve valid top countries; aggregate others and NaN into other_label
        transformed = series.apply(
            lambda val: val if pd.notna(val) and val in top_set else self.other_label
        )
        return transformed.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["Country_Grouped"], dtype=object)
```

#### 2. Negative Physical Activity: Sensitivity Experiment Design
Rather than making an arbitrary decision to automatically clip negative values, Phase 2 designs a formal sensitivity experiment comparing three methodologically sound treatments:

| Strategy | Implementation Details | Theoretical Rationale | Expected Trade-off |
| :--- | :--- | :--- | :--- |
| **Strategy 1: Bounded Clipping** | Transform values via $\max(x, 0.0)$ using a pipeline transformer. | Domain limits require non-negative activity; negative values ($-0.1$ to $-0.4$) represent small sensor/offset noise around zero. | Preserves all records; caps noise at the realistic boundary. |
| **Strategy 2: Pipeline Imputation (Set to NaN)** | Convert negative values to `NaN` inside the pipeline and impute via fold-level training median (`SimpleImputer(strategy='median')`). | If negative values represent recording errors rather than zero values, statistical imputation avoids assuming zero activity. | Replaces noise with the central tendency of the student cohort. |
| **Strategy 3: Observation Exclusion (Sensitivity Check)** | Exclude the 10 affected records ($N=10$ out of 4,998, $0.2\%$) from training splits. | Tests whether the presence of these 10 records exerts undue leverage on regression weights or tree splits. | Eliminates noisy records entirely; confirms model parameter stability. |

*Validation Comparison:* The cross-validation performance ($R^2$, MAE, RMSE) of all three strategies will be compared during the inner cross-validation phase. The original CSV remains untouched.

#### 3. Defensive Imputation Safeguards
Every pipeline branch includes a `SimpleImputer` stage to guarantee zero-crash execution during production inference:
- Continuous features: `SimpleImputer(strategy='median')`
- Categorical features: `SimpleImputer(strategy='most_frequent')`

#### 4. Categorical Encoders & Dense Matrix Compatibility
- `Stress_Level`: `OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Very High']], handle_unknown='use_encoded_value', unknown_value=-1)`
- Nominal Categories: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

> [!IMPORTANT]
> **Dense Matrix Compatibility for HistGradientBoostingRegressor:**  
> `HistGradientBoostingRegressor` requires dense 2D float arrays and throws a `TypeError` if fed a `scipy.sparse` matrix. By setting `sparse_output=False` across all `OneHotEncoder` instances, the pipeline outputs a unified, dense numerical matrix (38 columns for full schema, 34 for reduced schema). Given the dataset size ($3,498 \times 38 \times 8 \text{ bytes} \approx 1.06\text{ MB}$), this introduces zero risk of memory exhaustion while guaranteeing cross-model compatibility.

---

## 5. Model Experimentation Matrix

To determine the optimal predictive engine without predetermined algorithmic bias, Phase 2 evaluates four distinct model families across identical preprocessing splits:

| Model Identifier | Estimator Class | Rationale & Strengths | Hyperparameter Search Grid | Evaluation Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **M1: Baseline OLS** | `LinearRegression` | Parametric benchmark; zero hyperparameters; interpretable coefficients. | None (Analytical solution) | Establishes linear baseline and variance ceiling ($R^2 \approx 0.74$). |
| **M2: Regularized Ridge** | `Ridge` | L2-regularized linear model; addresses collinearity between platform and purpose. | `alpha`: `[0.01, 0.1, 1.0, 10.0, 50.0, 100.0]` | Assesses whether L2 shrinkage stabilizes collinear coefficients. |
| **M3: Random Forest** | `RandomForestRegressor` | Non-linear ensemble; captures interaction effects; robust to monotonic transforms. | `n_estimators`: `[100, 200, 300]`<br>`max_depth`: `[8, 12, 16, None]`<br>`min_samples_split`: `[2, 5, 10]`<br>`min_samples_leaf`: `[1, 2, 4]`<br>`max_features`: `['sqrt', 0.8, 1.0]` | Evaluates bagging non-linear performance and controls depth to halt overfitting. |
| **M4: Gradient Boosting** | `HistGradientBoostingRegressor` | Fast, histogram-based gradient boosting; handles continuous and one-hot features efficiently; compact serialized size. | `learning_rate`: `[0.02, 0.05, 0.1]`<br>`max_iter`: `[100, 150, 200]`<br>`max_depth`: `[4, 6, 8]`<br>`min_samples_leaf`: `[15, 20, 30]`<br>`l2_regularization`: `[0.0, 1.0, 10.0]` | Evaluates whether gradient boosting achieves lower residuals with smaller disk size ($< 5$ MB). |

### 5.1 Measured Evaluation Dimensions
Each candidate model will be evaluated across seven formal dimensions:
1. **$R^2$ Score (Unrounded):** Primary goodness-of-fit metric.
2. **MAE (Mean Absolute Error):** Average score divergence in native points.
3. **RMSE (Root Mean Squared Error):** Penalty metric for outlier mistakes.
4. **Generalization Gap ($\Delta R^2$):** $R^2_{\text{train}} - R^2_{\text{val}}$ to quantify overfitting.
5. **Training Wall-Clock Time:** Computational complexity during fitting.
6. **Inference Latency ($p95$ per sample):** Microseconds per prediction (NFR-01 target: $< 100\text{ms}$).
7. **Serialized Artifact Size:** Disk footprint (NFR-03 target: $< 50\text{MB}$).

---

## 6. Validation Architecture & Nested Cross-Validation

To guarantee completely unbiased performance estimation and eliminate data leakage, validation follows a strict nested protocol:

```
DATA SPLITTING AND VALIDATION HIERARCHY:
Raw CSV: 5,000 Records
  │
  └─► Deduplication (Drop 2 exact duplicate pairs) ──► N = 4,998 Records
        │
        ├─► Holdout Test Split (30%, random_state=42) ─► N = 1,500 Records [SEALED UNTIL FINAL EVALUATION]
        │
        └─► Training Set (70%, random_state=42) ───────► N = 3,498 Records
              │
              ▼
        5-FOLD NESTED CROSS-VALIDATION (Within 3,498 Training Records):
        ┌────────────────────────────────────────────────────────────────────────┐
        │ Outer Fold k (1..5): 80% Outer Train (2798) / 20% Outer Val (700)      │
        │                                                                        │
        │   Inner 5-Fold CV (Within Outer Train 2798):                           │
        │   • Compares Complete Configurations:                                  │
        │     - Candidate Estimators (OLS, Ridge, RF, HistGBR)                   │
        │     - Feature Configurations (Full vs. Reduced)                        │
        │     - Preprocessing Alternatives (Activity Clip vs. Impute)            │
        │     - Hyperparameter Search Spaces                                     │
        │   • Selects the Best Full Configuration for Outer Fold k               │
        │                                                                        │
        │   Outer Evaluation:                                                    │
        │   • Refits Selected Configuration on full Outer Train (2798)           │
        │   • Transforms Outer Val (700) using Outer Train fitted pipeline       │
        │   • Records Unbiased Generalisation Metrics                            │
        │     (Estimates performance of the selection procedure)                 │
        └────────────────────────────────────────────────────────────────────────┘
              │
              ▼
        Final Model Selection (Run on complete 3,498 Training Set):
        • Select winning configuration using training data only
        • Refit winning configuration on ALL 3,498 Training Records
              │
              ▼
        Final Holdout Evaluation:
        • Single, definitive evaluation on sealed 1,500 Holdout Records
```

### 6.1 Methodological Safeguards Against Model Selection Bias

> [!IMPORTANT]
> **Distinction Between Inner Selection and Outer Generalization:**  
> A critical flaw in naive validation workflows is using outer-fold validation scores to select a winning algorithm and then reporting those same outer-fold scores as an unbiased estimate of that winner. This re-introduces selection bias.
> 
> Under MindScope's nested protocol:
> 1. **Inner CV** selects the optimal modeling configuration (algorithm family, hyperparameters, feature set, preprocessing strategy) within each outer training fold.
> 2. **Outer CV** evaluates that complete automated selection procedure on unseen outer validation folds, providing an unbiased estimate of the pipeline's generalization capability.
> 3. **Final Refit:** To produce the deployable artifact, the complete selection procedure is executed on the entire 3,498 training set. The winning configuration is refit on all 3,498 records and evaluated **exactly once** on the sealed holdout test set ($N=1,500$).

---

## 7. Feature Redundancy Experiment Design

### 7.1 Empirical Coupling Context
As established in `docs/TARGET_VARIABLE_INVESTIGATION.md`, exact deterministic conditional coupling exists for the **top seven social media platforms** (accounting for 4,662 / 5,000 records, or **93.24% of the dataset**):
- Facebook $\equiv$ Networking ($100\%$)
- Instagram / TikTok / Snapchat $\equiv$ Entertainment ($100\%$)
- LinkedIn / YouTube $\equiv$ Education ($100\%$)
- Twitter $\equiv$ News ($100\%$)

In contrast, the remaining five regional and messaging platforms (WhatsApp, LINE, KakaoTalk, VKontakte, WeChat, totaling 338 records / $6.76\%$) reflect localized usage patterns.

### 7.2 Experimental Comparison Protocol
The modeling suite will evaluate two structural configurations under identical nested CV protocols:

| Dimension | Configuration A: Full Feature Schema | Configuration B: Reduced Feature Schema |
| :--- | :--- | :--- |
| **Predictor Count** | 12 features (All original predictors). | 11 features (Drop `Purpose_Of_Use`; retain `Most_Used_Platform`). |
| **Rationale** | Preserves original dataset baseline and reflects non-deterministic messaging platforms. | `Most_Used_Platform` contains 12 categories that uniquely partition the 4 purposes for 93.2% of the cohort. Dropping purpose removes redundancy without loss of predictive signal. |
| **Linear Collinearity** | Matrix condition number $\gg 10^3$; high VIF in linear baseline. | Condition number drops significantly; stabilizes regression coefficients. |
| **Ensemble Impact** | Evaluates whether tree splits benefit from redundant branches. | Evaluates tree depth efficiency and feature importance concentration. |
| **SHAP Impact (Phase 4)**| Split credit diluted between platform and purpose. | Unambiguous attribution assigned to specific platform choices. |

*Selection Rule:* If Configuration B matches Configuration A's out-of-sample $R^2$ within $\pm 0.003$ and reduces MAE or complexity, Configuration B will be selected for production deployment to maximize model parsimony and interpretability.

---

## 8. Evaluation & Error Analysis Protocol

### 8.1 Primary Continuous Metrics (Unrounded)
All primary metrics are computed on raw floating-point predictions ($\hat{y}_i$):
- **$R^2$:** Goodness of fit against baseline mean.
- **MAE:** $\frac{1}{n} \sum |y_i - \hat{y}_i|$
- **RMSE:** $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$

### 8.2 Supplementary Tolerance Metrics
To analyze prediction resolution around the discrete $0.1$ recorded score increments:
- **$\text{Tol}_{0.1}$:** Percentage of test samples where $|y_i - \hat{y}_i| \le 0.10$.
- **$\text{Tol}_{0.2}$:** Percentage of test samples where $|y_i - \hat{y}_i| \le 0.20$.

*Methodological Rule:* Tolerance metrics are strictly descriptive error concentrations and must never be represented as clinical validity thresholds.

### 8.3 Residual Diagnostics & Visual Analysis
Phase 2 will generate four comprehensive diagnostic plots:
1. **Residual Distribution Histogram & KDE:** Evaluates residual normality ($\epsilon = y - \hat{y}$), testing for zero mean ($\mu \approx 0$), skewness, and excess kurtosis.
2. **Actual vs. Predicted Scatter Plot:** Visualizes alignment along the identity line ($y = \hat{y}$) across the score continuum ($3.6$ to $9.4$).
3. **Residual vs. Predicted Scatter Plot:** Evaluates heteroscedasticity (checking whether error variance expands at extreme score boundaries).
4. **Binned Error Profile:** Evaluates MAE across score ranges ($[3.5, 5.0)$, $[5.0, 6.5)$, $[6.5, 8.0)$, $[8.0, 9.5]$) to identify non-uniform accuracy.

### 8.4 Descriptive Subgroup Error Analysis
To ensure fair and uniform model performance across demographic cohorts, MAE and RMSE will be reported across:
- `Academic_Level`: High School ($N \approx 500$) vs. Undergraduate ($N \approx 3,000$) vs. Graduate ($N \approx 1,500$).
- `Gender`: Male vs. Female.
- `Stress_Level`: Low, Medium, High, Very High.

*Safety Constraint:* Subgroup analyses are purely statistical error verifications. They will not assert causal or clinical differences.

---

## 9. Proposed Implementation Sequence

To execute Phase 2 without disruption, implementation will proceed in four discrete stages:

```
STAGE 1: Reusable Preprocessing Module (src/preprocessing.py)
├── Implement CountryTopKEncoder (with fit-only learning & 'Other' preservation)
├── Implement PhysicalActivityHandler (supporting Clipping, NaN-Imputation, Exclusion)
├── Build build_preprocessor(drop_purpose=False, activity_strategy='clip')
└── Unit tests verifying pipeline transform shapes and dense matrix outputs

STAGE 2: Model Training & Experiment Runner (src/train.py)
├── Load dataset and deduplicate (4,998 rows)
├── Execute 70/30 train_test_split (random_state=42)
├── Run 5-fold Nested CV across M1 (OLS), M2 (Ridge), M3 (Random Forest), M4 (HistGBR)
├── Evaluate preprocessing sensitivity alternatives (Clip vs. Impute vs. Exclude)
├── Evaluate feature redundancy configurations (Full vs. Reduced)
├── Tabulate unbiased outer CV generalization performance
├── Refit selected configuration on full X_train (3,498)
└── Persist native scikit-learn 1.8.0 pipeline to Mental_Health_Model.pkl

STAGE 3: Evaluation & Error Diagnostics (src/evaluate.py)
├── Evaluate serialized model once on sealed holdout X_test (1,500)
├── Generate unrounded R², MAE, RMSE and tolerance metrics (Tol_0.1, Tol_0.2)
├── Generate residual plots and subgroup error breakdowns
└── Save evaluation report to docs/MODEL_PERFORMANCE_REPORT.md

STAGE 4: Service Integration & Pipeline Unit Testing (tests/test_pipeline.py)
├── Verify Mental_Health_Model.pkl loads in sklearn 1.8.0 with ZERO warnings
├── Verify repr(model) executes without AttributeError
├── Verify FastAPI endpoints (/predict, /health) function seamlessly with new model
└── Run full test suite confirming 100% pass rate
```

---

## 10. Summary of Risks & Decision Gates

| Decision Point | Evaluated Options | Decision Criteria & Methodology |
| :--- | :--- | :--- |
| **Physical Activity Noise** | 1. Bounded Clip<br>2. NaN Imputation<br>3. Exclude Rows ($N=10$) | Selected via inner CV sensitivity comparison. If variance difference is negligible ($< 0.001$), bounded clipping is retained for simplicity. |
| **Model Family** | 1. OLS / Ridge<br>2. Random Forest<br>3. HistGradientBoosting | Evaluated via nested CV. If HistGradientBoosting matches RF accuracy ($R^2 \approx 0.87$), HistGBR is preferred for compact size ($< 5\text{MB}$) and fast inference. |
| **Feature Redundancy** | 1. Full 12 Features<br>2. Reduced 11 Features | Evaluated via nested CV. If $R^2$ degradation is $< 0.003$, reduced model is selected for cleaner interpretability. |
| **Matrix Representation** | Dense arrays via `sparse_output=False` | Enforced globally to guarantee full compatibility with `HistGradientBoostingRegressor`. |

---
*End of Refined Phase 2 Machine Learning Pipeline Design Specification. Awaiting user review and authorization before creating pipeline source code or training models.*
