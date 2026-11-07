# MindScope — Phase 3, Step 3.2: Global SHAP Analysis & Visualization Cache

## 1. Executive Summary

This report documents the execution, findings, and artifact cache for **Step 3.2: Global SHAP Analysis & Visualization Cache** of the MindScope Explainable AI (XAI) subsystem. Using the trained production model pipeline ([`mindscope_rf_v1.joblib`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/models/mindscope_rf_v1.joblib)) and the core explainer engine ([`MindScopeExplainer`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/src/explain.py)), we evaluated tree-path SHAP (*SHapley Additive exPlanations*) attributions across a deterministic evaluation sample drawn strictly from the training partition.

All explanations strictly respect data leakage boundaries: the sealed holdout dataset ([`artifacts/sealed_holdout_test.csv`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/sealed_holdout_test.csv)) was **not** opened, accessed, or scored. Additive efficiency was empirically verified to machine precision ($\max |\Delta| = 4.26 \times 10^{-14}$), confirming exact tree-path additivity across all $N=500$ evaluated records.

---

## 2. Methodology & Data Governance

### 2.1 Partitioning & Deterministic Sampling
To ensure reproducibility and eliminate data snooping:
1. **Raw Dataset Source**: [`Student Social Media And Mental Health Impact.csv`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/Student%20Social%20Media%20And%20Mental%20Health%20Impact.csv) (5,000 raw rows; 4,998 unique records after deduplication).
2. **Train/Holdout Split**: Reconstructed identically to Phase 2 via [`src/preprocessing.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/src/preprocessing.py) and [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) (`test_size=1500`, `random_state=42`), isolating the 3,498-record training partition $X_{\text{train}}$.
3. **Sealed Holdout Isolation**: The 1,500-record test set was completely untouched.
4. **Evaluation Subsampling**: A deterministic sample of $N=500$ instances was drawn from $X_{\text{train}}$ using `random_state=42`. As established in Step 3.1 validation, $N=500$ provides tight standard errors while maintaining high computational tractability (~85 seconds execution time for full feature-space decomposition).

### 2.2 Model Architecture & Feature Dimensions
- **Fitted Pipeline**: [`artifacts/models/mindscope_rf_v1.joblib`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/models/mindscope_rf_v1.joblib)
- **Base Estimator**: `RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=2, min_samples_split=2, random_state=42)`
- **Expected Value (Base Value $\mathbb{E}[\hat{y}]$)**: **6.2159**
- **Original Feature Count**: 12 tabular predictors
- **Transformed Feature Count**: 38 transformed features output by the ColumnTransformer (incorporating `CountryTopKEncoder`, log1p skew transform, bounded physical activity clipping, standard scalers, and one-hot encoders).

---

## 3. Global Feature Importance Results (Original 12 Predictors)

### 3.1 Methodological Formulation & Game-Theoretic Framework

The model's tree-based explainability follows a two-tier aggregation framework:
1. **Transformed Space Execution**: The fitted Random Forest estimator operates directly on **38 transformed numerical features** emitted by the preprocessing ColumnTransformer. TreeSHAP evaluates tree splits across these 38 dimensions.
2. **Feature Grouping & Dynamic Mapping**: The 38 transformed features are mapped dynamically into the **12 original input-feature groups** using the pipeline's metadata.
3. **Additive Prediction Invariance**: For every evaluated record $i$, the signed attributions of all transformed components belonging to parent feature $f$ are summed:
   $$\Phi_{i, f} = \sum_{j \in \text{components}(f)} \phi_{i, j}$$
   This summation strictly preserves the additive decomposition (efficiency axiom) of the model prediction:
   $$\hat{y}_i = \mathbb{E}[\hat{y}] + \sum_{j=1}^{38} \phi_{i, j} = \mathbb{E}[\hat{y}] + \sum_{f=1}^{12} \Phi_{i, f}$$
4. **Aggregated Transformed Attributions vs. Atomic 12-Player Games**:
   These attributions represent **aggregated transformed-feature attributions**. They are not necessarily equivalent to Shapley values calculated from a hypothetical separate 12-player game where each original feature is treated as a single atomic player (which would perturb entire categorical blocks simultaneously during tree traversal). Instead, they represent the additive aggregation of the 38-player game played across the actual decision nodes of the trained Random Forest.
5. **Global Importance Metric**:
   Global mean absolute feature importance is calculated from the absolute values of these aggregated per-record attributions:
   $$I_f = \frac{1}{N} \sum_{i=1}^{N} |\Phi_{i, f}|$$
   The relative importance share is defined as:
   $$\text{Relative Share}_f = \frac{I_f}{\sum_{k=1}^{12} I_k} \times 100\%$$
   where the total feature attribution mass across all 12 original features is:
   $$\sum_{k=1}^{12} I_k = 1.358235249146866$$
   Unrounded relative shares sum to exactly 100.0%.
6. **Non-Causal Interpretation**:
   These metrics strictly describe model behavior within its learned response surface. They must not be conflated with real-world causal effects, psychological mechanisms, or clinical diagnoses.

### 3.2 Global Importance Ranking Table

| Rank | Predictor Feature | Mean $\| \text{SHAP} \|$ | Mean Signed SHAP | Median $\| \text{SHAP} \|$ | Std Dev | Min SHAP | Max SHAP | Relative Share |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `Avg_Daily_Usage_Hours` | **0.7541** | -0.0583 | 0.8296 | 0.8549 | -1.3597 | +1.8884 | **55.52%** |
| **2** | `Sleep_Hours_Per_Night` | **0.2838** | +0.0087 | 0.2830 | 0.3120 | -0.6470 | +0.6450 | **20.90%** |
| **3** | `Daily_Unlocks` | **0.0567** | -0.0158 | 0.0396 | 0.0800 | -0.5135 | +0.3125 | **4.18%** |
| **4** | `Purpose_Of_Use` | **0.0514** | +0.0043 | 0.0367 | 0.0691 | -0.2004 | +0.3489 | **3.78%** |
| **5** | `Country` | **0.0462** | +0.0031 | 0.0231 | 0.0941 | -0.8542 | +0.4853 | **3.40%** |
| **6** | `Most_Used_Platform` | **0.0454** | -0.0001 | 0.0348 | 0.0636 | -0.2154 | +0.2874 | **3.34%** |
| **7** | `Physical_Activity_Hours` | **0.0300** | +0.0021 | 0.0229 | 0.0403 | -0.2052 | +0.1523 | **2.21%** |
| **8** | `Study_Hours` | **0.0297** | +0.0029 | 0.0213 | 0.0409 | -0.1841 | +0.1648 | **2.19%** |
| **9** | `Age` | **0.0289** | +0.0014 | 0.0194 | 0.0414 | -0.1244 | +0.2544 | **2.13%** |
| **10** | `Academic_Level` | **0.0115** | -0.0001 | 0.0062 | 0.0191 | -0.1050 | +0.0967 | **0.85%** |
| **11** | `Stress_Level` | **0.0108** | +0.0057 | 0.0075 | 0.0159 | -0.0519 | +0.1097 | **0.80%** |
| **12** | `Gender` | **0.0097** | +0.0006 | 0.0063 | 0.0140 | -0.0668 | +0.0727 | **0.71%** |

> [!NOTE]
> **Rounding Note**: The unrounded relative percentages sum to exactly **100.0%**. The two-decimal rounded values sum to **100.01%** ($55.52 + 20.90 + 4.18 + 3.78 + 3.40 + 3.34 + 2.21 + 2.19 + 2.13 + 0.85 + 0.80 + 0.71 = 100.01\%$) due strictly to independent rounding of each individual feature's quotient; values are reported as true individual roundings rather than artificially forced.

### 3.3 Key Scientific Observations

1. **Hierarchy of Behavioral Predictors**:
   - `Avg_Daily_Usage_Hours` (55.52%) and `Sleep_Hours_Per_Night` (20.90%) collectively account for **76.42%** of the model's total feature attribution mass.
   - High screen usage acts as the primary downward driver of predicted mental health scores (pushing scores down by up to $-1.36$ points), while low usage elevates predicted scores by up to $+1.89$ points.
   - Sleep hours demonstrate a strong monotonic positive relationship: adequate sleep pushes predicted scores upward, whereas sleep deficit reduces them.

2. **Secondary Behavioral Signals**:
   - `Daily_Unlocks` (0.0567), `Purpose_Of_Use` (0.0514), `Country` (0.0462), and `Most_Used_Platform` (0.0454) form a distinct second tier of modest contributors.
   - Frequent smartphone unlocking provides an additional negative attribution on score, reflecting habitual task interruptions.

3. **Demographic & Contextual Baselines**:
   - Demographic variables (`Gender`: 0.0097, `Academic_Level`: 0.0115, `Age`: 0.0289) and self-reported `Stress_Level` (0.0108) have minimal marginal predictive weight once behavioral habits (usage, sleep, unlocks, purpose) are accounted for.

---

## 4. Transformed Feature Level Decomposition (Top 15 Features)

Inspecting individual transformed features reveals specific categorical level dynamics:

| Rank | Transformed Feature | Parent Feature | Mean $\| \text{SHAP} \|$ | Mean Signed SHAP | Std Dev |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | `standard_numeric__Avg_Daily_Usage_Hours` | `Avg_Daily_Usage_Hours` | 0.7541 | -0.0583 | 0.8540 |
| 2 | `standard_numeric__Sleep_Hours_Per_Night` | `Sleep_Hours_Per_Night` | 0.2838 | +0.0087 | 0.3116 |
| 3 | `standard_numeric__Daily_Unlocks` | `Daily_Unlocks` | 0.0567 | -0.0158 | 0.0799 |
| 4 | `nominal__Purpose_Of_Use_Entertainment` | `Purpose_Of_Use` | 0.0458 | +0.0027 | 0.0635 |
| 5 | `physical__Physical_Activity_Hours` | `Physical_Activity_Hours` | 0.0300 | +0.0021 | 0.0403 |
| 6 | `skewed__Study_Hours` | `Study_Hours` | 0.0297 | +0.0029 | 0.0408 |
| 7 | `standard_numeric__Age` | `Age` | 0.0289 | +0.0014 | 0.0414 |
| 8 | `nominal__Most_Used_Platform_Instagram` | `Most_Used_Platform` | 0.0236 | +0.0006 | 0.0326 |
| 9 | `country__Country_Grouped_Other` | `Country` | 0.0130 | -0.0016 | 0.0206 |
| 10 | `nominal__Most_Used_Platform_WhatsApp` | `Most_Used_Platform` | 0.0125 | -0.0031 | 0.0335 |
| 11 | `country__Country_Grouped_Turkey` | `Country` | 0.0121 | +0.0011 | 0.0521 |
| 12 | `ordinal__Stress_Level` | `Stress_Level` | 0.0108 | +0.0057 | 0.0159 |
| 13 | `nominal__Academic_Level_Undergraduate` | `Academic_Level` | 0.0101 | +0.0004 | 0.0177 |
| 14 | `nominal__Purpose_Of_Use_Socializing` | `Purpose_Of_Use` | 0.0097 | -0.0014 | 0.0194 |
| 15 | `country__Country_Grouped_Japan` | `Country` | 0.0094 | +0.0003 | 0.0371 |

### Categorical Insights:
- **`Purpose_Of_Use_Entertainment`**: By far the most influential purpose category (Mean $|\text{SHAP}| = 0.0458$), indicating that consuming content strictly for entertainment exerts a discernible predictive influence compared to professional, work, or academic usage.
- **Platform Influences**: `Instagram` (0.0236) and `WhatsApp` (0.0125) exhibit the highest attributions among platform indicators, whereas platforms like LinkedIn, Reddit, and Discord have near-zero marginal influence on the model's output.

---

## 5. Visual Artifact Cache

The following publication-quality visualizations have been generated and cached in [`artifacts/explanations/`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations):

1. **[`global_shap_summary.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_summary.png)**:
   - *Description*: Clean horizontal bar chart depicting mean absolute SHAP values across all 12 original schema features, annotated with exact numeric attributions. Highlights the steep dominance of screen usage and sleep duration.
2. **[`global_shap_beeswarm.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_beeswarm.png)**:
   - *Description*: Beeswarm plot for the 12 original features. Features are ranked vertically by importance, with individual points showing the spread, density, and directionality of attributions across the 500 sampled students.
3. **[`transformed_shap_beeswarm.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/transformed_shap_beeswarm.png)**:
   - *Description*: High-resolution beeswarm plot displaying the top 20 transformed feature channels, highlighting the binary shift in attributions for one-hot encoded categories (e.g., Entertainment, Instagram, WhatsApp, Turkey, Undergraduate).
4. **[`global_shap_distribution.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_distribution.png)**:
   - *Description*: Dual-layer violin and boxplot visualization capturing the distribution shape, median, interquartile range (IQR), and positive/negative skewness of signed SHAP values across all 12 original features.

---

## 6. Scientific Limitations & Statistical Non-Causality

> [!IMPORTANT]
> **Attribution vs. Causation**:
> SHAP attributions reflect the **statistical contribution** of each feature toward the conditional expectation $\mathbb{E}[\hat{y} \mid X]$ of the trained `RandomForestRegressor`. They describe **how the fitted model weights features** in its ensemble trees, **not** causal mechanisms in human psychology or clinical pathology.
>
> - A negative SHAP value for high `Avg_Daily_Usage_Hours` does not establish that reducing screen time will causally improve a student's mental health score by that amount (potential confounding, reverse causality, or unobserved life stressors).
> - Attributions must not be interpreted as psychiatric diagnostics or clinical treatment recommendations.

---

## 7. Verification & Audit Trail

| Audit Dimension | Value / Status | Verification Check |
|:---|:---|:---:|
| **Sample Size ($N$)** | 500 instances | Confirmed deterministic random sample (`random_state=42`) |
| **Training Partition Size** | 3,498 instances | Confirmed identical to Phase 2 training split |
| **Holdout Dataset Integrity** | Untouched | `artifacts/sealed_holdout_test.csv` was never accessed |
| **Max Additivity Discrepancy** | $4.26 \times 10^{-14}$ | Pass ($\ll 10^{-5}$ tolerance) |
| **All Attributions Finite** | 100% finite | No `NaN`, `Inf`, or missing attributions |
| **Test Suite Status** | 67 / 67 Passed | Complete test suite passes cleanly |

### Generated Artifact Manifest
- [`artifacts/explanations/global_importance.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_importance.json) (3.7 KB)
- [`artifacts/explanations/transformed_importance.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/transformed_importance.json) (10.0 KB)
- [`artifacts/explanations/global_analysis_metadata.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_analysis_metadata.json) (1.4 KB)
- [`artifacts/explanations/global_shap_summary.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_summary.png) (228 KB)
- [`artifacts/explanations/global_shap_beeswarm.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_beeswarm.png) (551 KB)
- [`artifacts/explanations/transformed_shap_beeswarm.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/transformed_shap_beeswarm.png) (625 KB)
- [`artifacts/explanations/global_shap_distribution.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_distribution.png) (389 KB)

