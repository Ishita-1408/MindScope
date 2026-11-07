# MindScope — Phase 3: Explainable AI (XAI) Comprehensive Technical Report

**Document Status:** Official Consolidated Technical Report  
**Milestone:** Phase 3, Step 3.5 (Final Technical Report)  
**Execution Date:** 2026-11-07  
**Runtime Environment:** Python 3.13.3 (AMD64, Windows 11), scikit-learn 1.8.0, shap 0.52.0, matplotlib 3.11.2  
**Target Estimator:** MindScope Production Random Forest Pipeline ([`artifacts/models/mindscope_rf_v1.joblib`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/models/mindscope_rf_v1.joblib))  
**Expected Model Baseline:** $\mathbb{E}[\hat{y}] = 6.21592853$ points  
**Sealed Holdout Isolation:** Fully Verified ([`artifacts/sealed_holdout_test.csv`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/sealed_holdout_test.csv) unaccessed during Phase 3)  

---

## 1. Executive Summary

### 1.1 Purpose and Scientific Mandate
Phase 3 establishes an Explainable AI (XAI) subsystem for MindScope, rendering the production Random Forest regression pipeline transparent, interpretable, and verifiable. In behavioral and health analytics, black-box predictions risk generating unsupported causal assumptions, masking algorithmic biases, or misleading academic advisors and students. 

The XAI module implements cooperative game-theoretic feature attribution using **TreeSHAP** (SHapley Additive exPlanations). Operating strictly as an external, read-only analytical layer around the frozen model pipeline, it decomposes the model's conditional response surface at both population scale (global mechanics) and individual case scale (local prediction waterfalls).

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              MINDSCOPE XAI ARCHITECTURE                               │
│                                                                                       │
│   Input Record (12 Predictors)                                                        │
│   │                                                                                   │
│   ▼                                                                                   │
│   Fitted ColumnTransformer (`preprocessor.transform`)                                 │
│   │                                                                                   │
│   ▼                                                                                   │
│   38-Dimensional Transformed Space (One-Hot, Standard Scaled, Log-Transformed)        │
│   │                                                                                   │
│   ▼                                                                                   │
│   TreeSHAP Explainer (`shap.TreeExplainer(model, feature_perturbation="tree_path")`)  │
│   │                                                                                   │
│   ├─────────────────────────────────────────┬─────────────────────────────────────┤   │
│   ▼                                         ▼                                     │   │
│   38 Transformed Attributions               12 Grouped Predictor Attributions     │   │
│   (Indicator & Level Attributions)          (Additive Axiomatic Invariance)       │   │
│   │                                         │                                     │   │
│   ├── Local Waterfall Steps                 ├── Global Mean |SHAP| Rankings       │   │
│   ├── Non-Linear Dependence Grids           ├── Normalized Importance Shares      │   │
│   └── 38x38 TreeSHAP Interaction Matrix     └── 12x12 Aggregated Interaction Grid │   │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Model and Explainer Configuration
- **Underlying Estimator:** `RandomForestRegressor(n_estimators=150, max_depth=None, min_samples_split=2, min_samples_leaf=2, random_state=42)` encapsulated inside a scikit-learn `Pipeline`.
- **Explainer Implementation:** [`src/explain.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/src/explain.py) (`MindScopeExplainer`).
- **Explanation Method:** `feature_perturbation="tree_path_dependent"`, evaluating observational conditional expectations $E[Y \mid X_S = x_S]$ using node weights across the 150 trees.
- **Expected Baseline:** Exactly $6.2159$ points on the continuous score scale.

### 1.3 What Phase 3 Establishes vs. What It Does Not Establish
- **What Phase 3 Establishes:**
  1. Complete additive decomposition of predictions into 12 original features with verified mathematical additivity. Maximum observed discrepancy is $8.881784 \times 10^{-16}$ points (approximately $8.88 \times 10^{-16}$) for the local single-profile demonstration and $4.2633 \times 10^{-14}$ points (approximately $4.26 \times 10^{-14}$) across the $N=500$ global reference training cohort, both demonstrating floating-point machine precision well within the enforced test tolerance ($\text{atol} = 10^{-5}$).
  2. The relative predictive importance of input variables within this specific Random Forest model ($76.42\%$ concentrated in screen time and sleep duration).
  3. Non-linear response curves and zero-crossing thresholds in the model's decision manifold.
  4. Mathematical validation of pairwise interaction aggregation ($38 \times 38 \to 12 \times 12$) satisfying symmetry, row-sum consistency, and total additivity.
  5. API and frontend integration providing student-facing attribution waterfalls with neutral directional phrasing.
- **What Phase 3 Does Not Establish:**
  1. **No Causal Inference:** Attributions reflect observational associations learned by decision trees from cross-sectional data. A positive or negative attribution does not imply that intervening on a habit will causally alter student mental health.
  2. **No Clinical Validity:** `Mental_Health_Score` is an undocumented index from an unverified online dataset, not a clinically validated psychiatric scale (such as PHQ-9 or GAD-7). Attributions carry no diagnostic or therapeutic authority.
  3. **No Population Generalizability:** Findings describe the properties of this dataset and model; they cannot be generalized to broader adolescent or university populations.

---

## 2. Technical Architecture

### 2.1 Dynamic Feature Mapping ($38 \to 12$)
The preprocessor maps 12 raw student attributes into 38 numerical inputs:
- `country`: 10 binary indicators (Top 9 countries + `Other`).
- `skewed`: 1 continuous column (`Study_Hours` via $\log(1+x)$ + Standard Scaling).
- `physical`: 1 continuous column (`Physical_Activity_Hours` via lower clipping at $0.0$ + Standard Scaling).
- `standard_numeric`: 4 continuous columns (`Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Sleep_Hours_Per_Night` via Standard Scaling).
- `ordinal`: 1 ordinal integer column (`Stress_Level`, encoded $0–3$).
- `nominal`: 21 binary indicators (`Gender`: 2, `Academic_Level`: 3, `Most_Used_Platform`: 12, `Purpose_Of_Use`: 4).

Rather than hardcoding feature indices, `MindScopeExplainer` inspects `preprocessor.get_feature_names_out()` dynamically at initialization, verifying that dimension $M=38$ matches `model.n_features_in_`. Each transformed column is mapped to its parent predictor using deterministic prefix and substring resolution (`explainer.parent_to_indices`).

### 2.2 Tree-Path-Dependent Explanation Method
TreeSHAP is executed with `feature_perturbation="tree_path_dependent"`. This method computes Shapley values by traversing the internal decision trees using node sample counts recorded during training. 

**Advantages:**
- Requires no external background dataset, eliminating background sampling variance.
- Evaluates realistic feature combinations that respect the observational correlation structure of the data.
- Executes single-sample local explanations in under $15\text{ ms}$.

### 2.3 Mathematical Additivity Validation
The Shapley efficiency axiom guarantees that the sum of transformed attributions equals the model prediction minus the expected value:
$$\sum_{j=1}^{38} \phi_{i, j} = \hat{y}_i - \mathbb{E}[\hat{y}]$$

By aggregating across component subsets $J_k \subset \{1, \dots, 38\}$ for each original feature $k \in \{1, \dots, 12\}$:
$$\Phi_{i, k} = \sum_{j \in J_k} \phi_{i, j}$$

Summing across all 12 original features reconstructs the prediction identically:
$$\mathbb{E}[\hat{y}] + \sum_{k=1}^{12} \Phi_{i, k} = \hat{y}_i$$

Empirical validation across both individual explanations and population reference cohorts confirms that additivity holds to floating-point precision:
- **Local Synthetic Profile:** Discrepancy is $|\hat{y} - (\mathbb{E}[\hat{y}] + \sum \Phi)| = 8.881784 \times 10^{-16}\text{ points}$ (approximately $8.88 \times 10^{-16}$).
- **Global Reference Cohort ($N=500$ Training Records):** Maximum absolute discrepancy across all 500 observations is $\max |\Delta| = 4.2633 \times 10^{-14}\text{ points}$ (approximately $4.26 \times 10^{-14}$).
Both results satisfy the automated test guard tolerance ($\text{atol} = 1.0 \times 10^{-5}$) and confirm that total feature attribution is conserved under linear partition aggregation, with only negligible floating-point numerical discrepancy.

### 2.4 Backend and Frontend Integration
- **Backend API (`main.py`):** Exposes `POST /explain`, accepting a 12-field student record and returning the model prediction, baseline ($6.2159$), individual feature attributions ($\Phi_k$), direction labels (`positive` / `negative`), feature ranks, a 14-step cumulative waterfall sequence, and the mandatory non-causal disclaimer.
- **Frontend Presentation (`index.html`, `style.css`, `script.js`):** Displays a feature attribution card with neutral slate/blue styling:
  - Positive attribution (`.pill-positive`): Sky-blue badge displaying `↑ +{value}`, with the tooltip: *"Moves the model prediction above its expected baseline."*
  - Negative attribution (`.pill-negative`): Slate-stone badge displaying `↓ −{value}`, with the tooltip: *"Moves the model prediction below its expected baseline."*
  - Evaluative green/red coloring and moralizing terms (*healthy, unhealthy, good habit, bad habit, risk reduction*) are strictly excluded.

---

## 3. Global Feature Importance

### 3.1 Ranked Predictor Importance ($N=500$ Deterministic Training Records)
Global importance is quantified as the mean absolute Shapley value across $N=500$ training observations:
$$I_k = \frac{1}{N} \sum_{i=1}^{N} |\Phi_{i, k}|$$

Normalized importance shares represent each predictor's fraction of the total mean absolute SHAP attribution sum ($\sum_{k=1}^{12} I_k = 1.358235249146866$):

| Rank | Original Predictor Feature | Mean \|SHAP\| ($I_k$) | Exact Normalized Share | Displayed Share | Mean Signed SHAP | Std Dev |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | `Avg_Daily_Usage_Hours` | $0.754084$ | $55.519413\%$ | **$55.52\%$** | $-0.0583$ | $0.8549$ |
| **2** | `Sleep_Hours_Per_Night` | $0.283828$ | $20.896848\%$ | **$20.90\%$** | $+0.0087$ | $0.3120$ |
| **3** | `Daily_Unlocks` | $0.056709$ | $4.175184\%$ | **$4.18\%$** | $-0.0158$ | $0.0800$ |
| **4** | `Purpose_Of_Use` | $0.051374$ | $3.782421\%$ | **$3.78\%$** | $+0.0043$ | $0.0691$ |
| **5** | `Country` | $0.046218$ | $3.402806\%$ | **$3.40\%$** | $+0.0031$ | $0.0941$ |
| **6** | `Most_Used_Platform` | $0.045367$ | $3.340113\%$ | **$3.34\%$** | $-0.0001$ | $0.0636$ |
| **7** | `Physical_Activity_Hours` | $0.030015$ | $2.209864\%$ | **$2.21\%$** | $+0.0021$ | $0.0403$ |
| **8** | `Study_Hours` | $0.029700$ | $2.186661\%$ | **$2.19\%$** | $+0.0029$ | $0.0409$ |
| **9** | `Age` | $0.028948$ | $2.131315\%$ | **$2.13\%$** | $+0.0014$ | $0.0414$ |
| **10** | `Academic_Level` | $0.011504$ | $0.846948\%$ | **$0.85\%$** | $-0.0001$ | $0.0191$ |
| **11** | `Stress_Level` | $0.010807$ | $0.795669\%$ | **$0.80\%$** | $+0.0057$ | $0.0159$ |
| **12** | `Gender` | $0.009681$ | $0.712759\%$ | **$0.71\%$** | $+0.0006$ | $0.0140$ |
| **Total** | — | **$1.358235$** | **$100.000000\%$** | **$100.01\%$** | — | — |

*(Note: Rounded display shares sum to $100.01\%$ due strictly to independent half-up rounding of individual quotients; exact unrounded values sum to $100.000000\%$).*

### 3.2 Signed vs. Absolute SHAP Values
- **Mean Absolute SHAP ($I_k$):** Quantifies total impact or explanatory power. A feature that pushes some students up by $+1.0$ and others down by $-1.0$ has a large mean absolute attribution ($1.0$).
- **Mean Signed SHAP:** Quantifies net directional bias across the reference sample. For instance, `Avg_Daily_Usage_Hours` has a mean absolute attribution of $0.7541$ but a mean signed attribution of $-0.0583$. The signed mean near zero indicates that positive and negative attributions roughly counterbalance each other across the centered training cohort.

### 3.3 Reference to Cached Global Visualizations
All artifacts are cached in [`artifacts/explanations/`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations):
- [`global_shap_summary.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_summary.png): Horizontal bar chart ranking all 12 predictors by mean absolute SHAP value.
- [`global_shap_beeswarm.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_beeswarm.png): Population beeswarm depicting attribution spread and feature value coloring for the 12 original features.
- [`transformed_shap_beeswarm.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/transformed_shap_beeswarm.png): High-resolution beeswarm across all 38 indicator and continuous transformed features.
- [`global_shap_distribution.png`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/global_shap_distribution.png): Violin distributions illustrating attribution skewness across predictors.

---

## 4. Local Prediction Explanations

### 4.1 Prediction Reconstruction Mechanics
For any single profile, the local explanation decomposes the gap between the expected baseline and the final prediction:
$$\hat{y}_i = \mathbb{E}[\hat{y}] + \sum_{k=1}^{12} \Phi_{i, k}$$

Each feature contribution is tagged with natural units (e.g. `hours/day`, `unlocks/day`) and sorted in descending order of absolute magnitude $|\Phi_k|$ to form a 14-step waterfall sequence (Step 0: baseline, Steps 1–12: feature contributions, Step 13: prediction).

### 4.2 Illustrative Synthetic Case Study
To demonstrate waterfall mechanics without disclosing genuine student records, a synthetic profile was evaluated:
- **Input Attributes:** Age 21, Female, USA, Undergraduate, Instagram, Entertainment, Screen Time $5.5\text{ h/d}$, Unlocks $85\text{ /d}$, Study $4.0\text{ h/d}$, Physical Activity $1.0\text{ h/d}$, Sleep $6.5\text{ h/night}$, Stress Medium.
- **Expected Baseline:** $6.2159$ points.
- **Model Prediction:** $5.3610$ points (Net Shift: $-0.8549$ points).
- **Additivity Verification Error:** $8.88 \times 10^{-16}$ (Verified bitwise exact).

| Step | Type | Feature / Step Label | Feature Value | Attribution ($\Phi$) | Direction Label | Cumulative Score |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | Baseline | Expected Model Baseline $\mathbb{E}[\hat{y}]$ | — | — | Baseline | **$6.2159$** |
| 1 | Feature | Daily Social Media Usage | $5.5\text{ hrs/day}$ | **$-0.5387$** | Negative | $5.6773$ |
| 2 | Feature | Daily Device Unlocks | $85\text{ unlocks/day}$ | **$-0.1255$** | Negative | $5.5518$ |
| 3 | Feature | Primary Purpose of Use | Entertainment | **$-0.0969$** | Negative | $5.4549$ |
| 4 | Feature | Most Used Social Platform | Instagram | **$-0.0837$** | Negative | $5.3712$ |
| 5 | Feature | Daily Physical Activity | $1.0\text{ hrs/day}$ | **$-0.0675$** | Negative | $5.3037$ |
| 6 | Feature | Country of Residence | United States | **$+0.0528$** | Positive | $5.3565$ |
| 7 | Feature | Sleep Duration | $6.5\text{ hrs/night}$ | **$-0.0381$** | Negative | $5.3184$ |
| 8 | Feature | Daily Study Duration | $4.0\text{ hrs/day}$ | **$+0.0254$** | Positive | $5.3438$ |
| 9 | Feature | Student Age | $21\text{ yrs}$ | **$+0.0123$** | Positive | $5.3561$ |
| 10 | Feature | Academic Level | Undergraduate | **$+0.0079$** | Positive | $5.3640$ |
| 11 | Feature | Self-Reported Stress Level | Medium | **$-0.0052$** | Negative | $5.3588$ |
| 12 | Feature | Gender | Female | **$+0.0022$** | Positive | $5.3610$ |
| 13 | Prediction | Final Model Prediction $\hat{y}$ | — | **$-0.8549$** | Net Shift | **$5.3610$** |

### 4.3 Neutral Directional Disambiguation
A negative attribution (such as Screen Time $\Phi = -0.5387$) indicates strictly that $5.5\text{ hours/day}$ pushed the prediction below the model baseline for this student profile. It does **not** signify that the student has poor mental health, clinical impairment, or an unhealthy lifestyle.

---

## 5. Feature Dependence Analysis

Feature dependence analysis investigates how model predictions shift across individual feature domains using $N=500$ deterministic training records.

### 5.1 Numerical Predictors Summary

| Feature | Units | Domain Range | Mean | Pearson $r$ | Spearman $\rho$ | Mean \|SHAP\| | Response Characteristics |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `Avg_Daily_Usage_Hours` | hours/day | $[1.4, 8.7]$ | $5.16$ | $-0.937$ | $-0.952$ | $0.7541$ | Non-linear; zero-crossing threshold at $4.8–5.2\text{ h/d}$. |
| `Sleep_Hours_Per_Night` | hours/night | $[4.2, 9.6]$ | $6.97$ | $+0.788$ | $+0.768$ | $0.2838$ | Steep drop below $6\text{ h/night}$; plateau above $8.5\text{ h/night}$. |
| `Daily_Unlocks` | unlocks/day | $[71, 273]$ | $172.4$ | $-0.090$ | $-0.169$ | $0.0567$ | Weak monotonic trend; negative spread above $200\text{ unlocks/d}$. |
| `Physical_Activity_Hours`| hours/day | $[-0.1, 3.8]$ | $1.86$ | $+0.235$ | $+0.367$ | $0.0300$ | Mild positive monotonic association. |
| `Study_Hours` | hours/day | $[0.6, 7.2]$ | $3.93$ | $+0.124$ | $+0.182$ | $0.0297$ | Mild positive association with wide spread. |
| `Age` | years | $[18, 24]$ | $21.05$ | $+0.375$ | $+0.363$ | $0.0289$ | Modest positive trend across collegiate age range. |

### 5.2 Key Response Patterns
1. **Screen Time (`Avg_Daily_Usage_Hours`) — Observed Zero-Crossing Threshold:**
   - In the range $[1.4, 3.9]\text{ h/d}$ ($Q_1$), attributions are uniformly positive (mean $+1.150\text{ pts}$).
   - Between $3.9$ and $5.15\text{ h/d}$ ($Q_2$), attributions moderate to $+0.201\text{ pts}$.
   - The observed zero-crossing threshold occurs between **$4.8\text{ and }5.2\text{ hours/day}$**, where attributions cross zero relative to baseline.
   - For usage $> 5.15\text{ h/d}$ ($Q_3$ and $Q_4$), attributions become increasingly negative (mean $-0.633\text{ pts}$), dropping to a minimum of $-1.360\text{ pts}$ at $8.7\text{ h/d}$.
2. **Sleep Duration (`Sleep_Hours_Per_Night`) — Positive Shift Plateau:**
   - Sleep $< 6.0\text{ hours/night}$ shifts predictions sharply downward (minimum $-0.647\text{ pts}$).
   - Between $6.0$ and $8.0\text{ hours/night}$, attributions climb through zero.
   - Above $8.5\text{ hours/night}$, attributions plateau near $+0.645\text{ pts}$, indicating diminishing predictive sensitivity in the decision trees to excess sleep.
3. **Daily Unlocks (`Daily_Unlocks`):**
   - While overall correlation is weak ($\rho = -0.169$), high device unlock counts ($> 200\text{ unlocks/day}$) exhibit sporadic negative shifts down to $-0.513\text{ pts}$.

### 5.3 Categorical Attribution Distributions
- `Purpose_Of_Use`: `Networking` (mean $+0.0628$) and `Education` ($+0.0540$) shift predictions above baseline. `Entertainment` (50% of cohort) shifts predictions below baseline (mean $-0.0458$).
- `Most_Used_Platform`: `WhatsApp` (mean $+0.1732$) and `LINE` ($+0.0570$) exhibit positive shifts, whereas `Instagram` (mean $-0.0621$) and `KakaoTalk` ($-0.0652$) exhibit negative shifts. High-volume platforms like `TikTok` ($-0.0034$) and `Facebook` ($-0.0046$) center near zero.
- `Academic_Level`, `Stress_Level`, `Gender`: Exhibit narrow attribution bands with mean absolute impacts $< 0.012\text{ points}$.

---

## 6. Feature Interaction Analysis

### 6.1 Aggregation Mathematics ($38 \times 38 \to 12 \times 12$)
TreeSHAP interaction matrices quantify pure second-order non-additive effects:
$$\Phi_{i, j, k} = \text{Pure interaction between transformed features } j \text{ and } k$$

To aggregate into the 12 original features without loss of additivity:
$$\Phi_{i, A, B}^{\text{orig}} = \sum_{j \in I(A)} \sum_{k \in I(B)} \Phi_{i, j, k}^{\text{trans}}$$

**Mathematical Verification:**
1. **Symmetry:** $\max |M - M^T| = 1.08 \times 10^{-16}$.
2. **Row-Sum Consistency:** $\max |\sum_B \Phi_{A, B} - \Phi_A| < 2 \times 10^{-14}$.
3. **Prediction Reconstruction:** $\max |\sum_{A, B} \Phi_{A, B} - (f(x) - \mathbb{E}[f(x)])| < 2 \times 10^{-14}$.
4. **Metric Convention:** Reported values represent the single off-diagonal cell $(A, B)$. The pair's total contribution to the prediction shift is $2 \times \bar{\Phi}_{A, B}^{\text{inter}}$.

### 6.2 Exploratory $N=10$ Benchmark Findings
> [!NOTE]
> The interaction rankings below are exploratory findings from a deterministic benchmark sample of $N=10$ training observations. They demonstrate the operational validity of the aggregation pipeline. Because $N=10$ is small and lacks population-level confidence intervals, these rankings must not be interpreted as definitive population-wide properties or real-world effect sizes.

Top pairwise interactions ranked by mean absolute interaction strength in [`artifacts/explanations/dependence/interaction_analysis.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/dependence/interaction_analysis.json):

| Rank | Feature Pair | Mean \|Interaction\| | Interpretation Note |
|:---:|:---|:---:|:---|
| **1** | `Avg_Daily_Usage_Hours` $\times$ `Sleep_Hours_Per_Night` | **$0.1775\text{ pts}$** | Largest observed interaction in the $N=10$ sample; higher sleep duration was observed alongside a positive interaction attribution that moderates negative screen time shifts. |
| **2** | `Country` $\times$ `Avg_Daily_Usage_Hours` | $0.0436\text{ pts}$ | Categorical $\times$ Continuous joint tree conditioning. |
| **3** | `Most_Used_Platform` $\times$ `Avg_Daily_Usage_Hours` | $0.0419\text{ pts}$ | Categorical $\times$ Continuous joint tree conditioning. |
| **4** | `Avg_Daily_Usage_Hours` $\times$ `Daily_Unlocks` | **$0.0382\text{ pts}$** | Non-additive joint shift when screen time and unlocks are both high. |
| **5** | `Avg_Daily_Usage_Hours` $\times$ `Study_Hours` | $0.0244\text{ pts}$ | Continuous $\times$ Continuous interaction. |
| **6** | `Purpose_Of_Use` $\times$ `Avg_Daily_Usage_Hours` | $0.0231\text{ pts}$ | Categorical $\times$ Continuous interaction. |
| **7** | `Daily_Unlocks` $\times$ `Sleep_Hours_Per_Night` | $0.0205\text{ pts}$ | Continuous $\times$ Continuous interaction. |
| **8** | `Country` $\times$ `Sleep_Hours_Per_Night` | $0.0193\text{ pts}$ | Categorical $\times$ Continuous interaction. |
| **9** | `Purpose_Of_Use` $\times$ `Sleep_Hours_Per_Night` | $0.0190\text{ pts}$ | Categorical $\times$ Continuous interaction. |
| **10** | `Avg_Daily_Usage_Hours` $\times$ `Physical_Activity_Hours` | **$0.0171\text{ pts}$** | Reconciled rank; minor positive attribution buffering in high usage records. |
| **14** | `Study_Hours` $\times$ `Sleep_Hours_Per_Night` | **$0.0146\text{ pts}$** | Reconciled value; modest interaction effect in the fitted ensemble. |
| **28** | `Sleep_Hours_Per_Night` $\times$ `Stress_Level` | **$0.0049\text{ pts}$** | Reconciled rank/value; near purely additive in the fitted forest. |

### 6.3 Runtime Benchmarks and Scalability Projections
Execution throughput on Windows AMD64, Python 3.13.3 (150 trees, 38 features):
- $N=1$: $7.96\text{ s}$ ($7.96\text{ s/obs}$)
- $N=2$: $17.10\text{ s}$ ($8.55\text{ s/obs}$)
- $N=5$: $40.24\text{ s}$ ($8.05\text{ s/obs}$)
- $N=10$ (Standalone): $139.15\text{ s}$ ($13.92\text{ s/obs}$)

Throughput varied from $\approx 8.2\text{ s/obs}$ during short benchmark loops to $\approx 13.9\text{ s/obs}$ during sustained standalone execution. Cumulative CPU utilization, thread scheduling, and potential thermal throttling are possible contributing factors for this variation.

**Approximate Linear Extrapolations (Not Guaranteed Execution Times):**
- $N=500$ Reference Sample: Projected $\approx \mathbf{1.14\text{ to }1.93\text{ hours}}$.
- $N=3,498$ Full Training Partition: Projected $\approx \mathbf{7.97\text{ to }13.52\text{ hours}}$.

Full population interaction calculation is computationally prohibitive for continuous integration and automated builds. The $N=10$ sample serves as an approved exploratory feasibility demonstration.

---

## 7. Answers to Project Research Questions

### RQ1: Screen Time vs. Sleep Duration Association
- **Empirical Findings:**
  - `Avg_Daily_Usage_Hours` is the primary predictor in the model ($I = 0.7541\text{ pts}$, $55.52\%$ normalized share). It displays a strong negative monotonic association with predicted scores ($r = -0.937$, $\rho = -0.952$).
  - `Sleep_Hours_Per_Night` is the secondary predictor ($I = 0.2838\text{ pts}$, $20.90\%$ normalized share), exhibiting a positive monotonic association ($r = +0.788$, $\rho = +0.768$).
  - Together, they account for **$76.42\%$ of all feature attribution** in the model.
  - In the exploratory $N=10$ interaction benchmark, `Usage x Sleep` exhibited the largest off-diagonal interaction ($0.1775\text{ pts}$). Higher sleep duration was observed alongside a positive interaction attribution that softened the downward prediction shift of extreme screen time.

### RQ2: Platform and Motivation Specificity
- **Empirical Findings:**
  - Controlling for screen time and demographics, platform choice and usage motivation account for modest attribution shares: `Purpose_Of_Use` ($3.78\%$, rank 4) and `Most_Used_Platform` ($3.34\%$, rank 6).
  - Categorical attribution distributions ($N=500$) indicate that using platforms for `Networking` ($+0.0628$) and `Education` ($+0.0540$) shifts predictions above baseline, while `Entertainment` shifts predictions below baseline ($-0.0458$).
  - Among platforms, `WhatsApp` ($+0.1732$) shifts predictions above baseline, while `Instagram` ($-0.0621$) shifts predictions below baseline.
  - Collinearity analysis from Phase 2 established that platform deterministically dictates purpose for $93.2\%$ of records. Tree splits partition credit across both features, diluting individual importance.

### RQ3: Physical Activity as an Associational Moderator
- **Empirical Findings:**
  - `Physical_Activity_Hours` occupies rank 7 globally ($I = 0.0300\text{ pts}$, $2.21\%$ share), exhibiting a mild positive association with model predictions ($r = +0.235$, $\rho = +0.367$).
  - In the exploratory $N=10$ interaction sample, `Usage x Physical Activity` ranked 10th with a mean absolute interaction of $0.0171\text{ pts}$.
  - **Limitation & Inconclusive Boundary:** While physical activity exhibits a mild positive bivariate correlation and a minor pairwise interaction with screen time, a formal three-way moderation test across high-screen-time ($>6\text{h}$) and elevated-stress subgroups cannot be conclusively established without broader multi-way subgroup sampling.

### RQ5: TreeSHAP Non-Linear Attributions vs. Linear Regression Baseline
- **Empirical Comparison from Phase 2 Nested Cross-Validation:**
  - Baseline OLS Linear Regression (`M1_OLS_Full_Clip`): Outer test $R^2 = 0.7177 \pm 0.0167$, $\text{MAE} = 0.5240 \pm 0.0191$.
  - Tuned Random Forest (`M4a_RF_Full_Clip`): Outer test $R^2 = 0.8392 \pm 0.0128$, $\text{MAE} = 0.3831 \pm 0.0142$.
  - The non-linear ensemble achieved an error reduction of $26.9\%$ and an increase of $\Delta R^2 = +0.1215$ over the linear baseline.
- **Structural Discrepancies Revealed by TreeSHAP:**
  1. **Non-Linear Zero-Crossing:** Linear regression enforces a constant slope $\beta$, whereas TreeSHAP reveals an observed zero-crossing threshold between $4.8$ and $5.2\text{ hours/day}$ where negative attributions accelerate.
  2. **Sleep Plateau:** Linear regression assumes indefinite linear increases with sleep, whereas TreeSHAP reveals a plateau past $8.5\text{ hours/night}$.
  3. **Non-Additive Interactions:** OLS cannot model interactions without manual interaction terms. TreeSHAP captures joint conditioning (such as Usage $\times$ Sleep at $0.1775\text{ pts}$).
- **Reporting Limitation:** Individual OLS coefficient vectors were not persisted in the final model artifact (as Random Forest was selected as the production estimator). Detailed coefficient comparisons are therefore qualitative and structural based on nested CV metrics.

---

## 8. Model, Dataset and Governance Limitations

1. **Unverified Data Provenance:**
   - The underlying dataset is derived from an unverified public online release with potential self-reporting and selection biases.
   - Findings must not be generalized to broader student or adolescent populations.
2. **Undocumented Target Variable:**
   - `Mental_Health_Score` is an undocumented numerical index without clinical instrument mapping (such as PHQ-9 or GAD-7).
   - Predictions and explanations carry no clinical validity and must never be used for diagnosis, triage, or therapeutic evaluation.
3. **No Causal Inference:**
   - SHAP explains the statistical behavior of the fitted decision trees; it does not measure real-world causality.
4. **Game-Theoretic Aggregation Boundary:**
   - Grouping 38 transformed indicator attributions into 12 parent feature sums preserves additivity and symmetry, but is not mathematically equivalent to an atomic 12-player Shapley game. The same distinction applies to aggregated interaction values.
5. **Exploratory Scope of $N=10$ Interactions:**
   - Pairwise interaction rankings are derived from a benchmark sample of 10 records and are exploratory.
6. **Phase 2 Holdout Execution History Disclosed:**
   - During Phase 2, the sealed holdout (`artifacts/sealed_holdout_test.csv`) was opened and scored **3 times** (Tasks 640, 671, and 682) due to script errors before the final successful evaluation was persisted. Exactly-once evaluation is not claimed. However, zero holdout feedback was used to modify model parameters, feature selection, or preprocessing.
7. **Intentional Dual-Model Coexistence:**
   - In the API, `POST /predict` evaluates the legacy model (`Mental_Health_Model.pkl`, $25,694,899\text{ bytes}$) to maintain Phase 1 backwards compatibility.
   - `POST /explain` evaluates the modern Random Forest pipeline (`mindscope_rf_v1.joblib`) and decomposes that exact prediction via TreeSHAP.
   - Unification of `/predict` to the modern pipeline is scheduled for Phase 4.

---

## 9. Conclusion and Phase 4 Readiness

### 9.1 Summary of Phase 3 Deliverables
Phase 3 has delivered:
- `src/explain.py`: Production TreeSHAP explainer engine with dynamic $38 \to 12$ mapping.
- `src/global_shap.py`: Deterministic global analysis and visualization cache generation.
- `src/feature_analysis.py`: Dependence curves, interaction benchmarking, and $38 \to 12$ aggregation.
- API & Frontend Integration: `POST /explain` with neutral directional waterfall attributions.
- Automated Test Suite: 102 verified tests passing across 9 modules.
- Documentation: Complete design, step-wise reports, audit reports, and correction documents.

### 9.2 Remaining Dependencies & Transition to Phase 4
- **Step 3.4 Git Checkpoint:** Staging and committing the completed Step 3.4 and 3.5 milestones.
- **Formal Phase 3 Integrity Audit:** A final independent audit to verify complete milestone closure.
- **Phase 4 Preparation:** Transitioning to Modernization, Model Unification, and Dashboard Integration upon user authorization.

---

*End of Comprehensive Technical Report.*

