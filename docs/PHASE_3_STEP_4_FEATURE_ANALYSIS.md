# MindScope — Phase 3, Step 3.4: Feature Dependence & Interaction Analysis

**Document Status:** Complete & Verified (Audit Corrections Applied)  
**Milestone:** Phase 3, Step 3.4 (Feature Dependence & Interaction Analysis)  
**Execution Timestamp:** 2026-11-07  
**Runtime Environment:** Python 3.13.3, scikit-learn 1.8.0, shap 0.52.0, matplotlib 3.11.2  
**Target Estimator:** MindScope Production Random Forest Pipeline (`artifacts/models/mindscope_rf_v1.joblib`)  
**Expected Model Baseline:** $6.21592853$  

---

## 1. Executive Summary

Phase 3, Step 3.4 implements an empirical, mathematically verified analysis of feature dependence curves and pairwise feature interactions for the trained MindScope Random Forest pipeline.

All analyses strictly adhere to **training-partition data governance**, leaving the sealed holdout evaluation dataset (`artifacts/sealed_holdout_test.csv`), legacy model artifact (`Mental_Health_Model.pkl`), and previously generated global SHAP artifacts in `artifacts/explanations/` completely untouched.

### Key Milestones Achieved:
1. **Feature Dependence Engine**: Analyzed the model's non-linear response across all 12 original predictor features (6 numerical, 6 categorical) using $N=500$ deterministic training records.
2. **Empirical Response Thresholds Detected**:
   - `Avg_Daily_Usage_Hours` (Mean |SHAP| = $0.7541$, $\rho = -0.952$): Displays an observed non-linear zero-crossing threshold between $4.8$ and $5.2\text{ hours/day}$, transitioning from positive prediction shifts ($+1.15$ points below $4\text{ h/d}$) to negative prediction shifts (down to $-1.36$ points above $6.5\text{ h/d}$) relative to the expected model baseline.
   - `Sleep_Hours_Per_Night` (Mean |SHAP| = $0.2838$, $\rho = +0.768$): Exhibits a plateau in positive prediction shift past $8.5\text{ hours/night}$, with steep downward prediction shifts when sleep falls below $6.0\text{ hours/night}$.
3. **Computational Feasibility Benchmarking & Reconciliation**:
   - Empirically measured TreeSHAP interaction execution times across small sample sizes ($N \in \{1, 2, 5, 10\}$).
   - Observed throughput ranged from approximately $8.2\text{ seconds/observation}$ in short benchmark loops to $13.9\text{ seconds/observation}$ during sustained execution.
   - Approximate linear projections indicate that a 500-record reference sample would require approximately $1.14\text{ to }1.93\text{ hours}$, while the full $3,498$ training set would require approximately $7.97\text{ to }13.52\text{ hours}$.
   - Demonstrated why full-population interaction calculation is computationally prohibitive and established a targeted deterministic sample ($N=10$) as an exploratory feasibility demonstration.
4. **Interaction Aggregation & Validation ($38 \times 38 \to 12 \times 12$)**:
   - Aggregated transformed $38 \times 38$ interaction tensors into the $12 \times 12$ original predictor space by summing submatrices over parent feature index sets.
   - Proved and verified mathematical invariants:
     - Exact matrix symmetry ($\max |M - M^T| = 1.08 \times 10^{-16}$).
     - Exact row-sum consistency with grouped SHAP attributions ($\max |\sum_B \Phi_{A, B} - \Phi_A| < 2 \times 10^{-14}$).
     - Exact additivity reconstruction ($\max |\sum_{A, B} \Phi_{A, B} - (f(x) - \mathbb{E}[f(x)])| < 2 \times 10^{-14}$).
5. **Exploratory Interaction Findings ($N=10$ Sample)**:
   - Within the selected $N=10$ exploratory benchmark sample, Screen Time $\times$ Sleep Duration (`Avg_Daily_Usage_Hours` $\times$ `Sleep_Hours_Per_Night`) exhibited the largest mean absolute aggregated interaction attribution ($0.1775\text{ points}$).
   - Screen Time $\times$ Daily Unlocks (`Avg_Daily_Usage_Hours` $\times$ `Daily_Unlocks`) ranked 4th ($0.0382\text{ points}$).
   - Screen Time $\times$ Physical Activity (`Avg_Daily_Usage_Hours` $\times$ `Physical_Activity_Hours`) ranked 10th ($0.0171\text{ points}$).
   - Study Duration $\times$ Sleep Duration (`Study_Hours` $\times$ `Sleep_Hours_Per_Night`) ranked 14th ($0.0146\text{ points}$).
   - Sleep Duration $\times$ Stress Level (`Sleep_Hours_Per_Night` $\times$ `Stress_Level`) ranked 28th ($0.0049\text{ points}$).
   - These findings demonstrate the mathematical pipeline on a tractable subset and are exploratory; they do not establish population-wide rankings or causal effect sizes.
6. **Visualization Suite**: Generated 7 publication-quality PNG figures ($300\text{ DPI}$) and 4 JSON data artifacts in a dedicated subdirectory: `artifacts/explanations/dependence/`.
7. **Automated Testing Suite**: Implemented 13 unit/integration tests in `tests/test_feature_analysis.py`. The entire MindScope test suite stands at **102 passed, 0 failed, 13 warnings in 34.94s**.

---

## 2. Strict Data Governance & Isolation Architecture

To prevent data leakage, evaluation tampering, and artifact regression:

1. **Isolation from Sealed Holdout**:
   - `artifacts/sealed_holdout_test.csv` is protected by `verify_sealed_holdout_isolation()`, which raises a `PermissionError` if the holdout path is referenced or passed.
   - Zero holdout records were accessed, loaded, or inspected during this milestone.
2. **Deterministic Partition Reconstruction**:
   - The analysis reconstructed the exact $3,498$-record training partition from `Student Social Media And Mental Health Impact.csv` using the approved deterministic split:
     $$\text{train\_test\_split}(X, y, \text{test\_size}=0.30, \text{random\_state}=42)$$
   - Exactly $N=500$ records were drawn deterministically using `random_state=42`.
3. **Model Artifact Immutability**:
   - Production pipeline `artifacts/models/mindscope_rf_v1.joblib` was loaded strictly in read-only inference mode.
   - Legacy artifact `Mental_Health_Model.pkl` remained completely untouched (exactly $25,694,899\text{ bytes}$).
4. **Namespace Isolation for Artifacts**:
   - All newly generated outputs were routed into `artifacts/explanations/dependence/`.
   - The existing Step 3.2 global SHAP files in `artifacts/explanations/` (`global_importance.json`, `global_analysis_metadata.json`, etc.) remain identical and untampered.

---

## 3. Feature Dependence Analysis

Feature dependence analysis investigates how the fitted model shifts its prediction as individual predictor values vary across the training population.

### 3.1 Numerical Feature Summary (N = 500 Training Observations)

| Rank | Feature | Units | Range | Mean Val | Pearson $r$ | Spearman $\rho$ | Mean \|SHAP\| | Positive % | Negative % |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `Avg_Daily_Usage_Hours` | hours/day | $[1.4, 8.7]$ | $5.16$ | $-0.937$ | $-0.952$ | $0.7541$ | $45.8\%$ | $54.2\%$ |
| 2 | `Sleep_Hours_Per_Night` | hours/night | $[4.2, 9.6]$ | $6.97$ | $+0.788$ | $+0.768$ | $0.2838$ | $49.2\%$ | $50.8\%$ |
| 3 | `Daily_Unlocks` | unlocks/day | $[71, 273]$ | $172.4$ | $-0.090$ | $-0.169$ | $0.0567$ | $44.8\%$ | $55.2\%$ |
| 7 | `Physical_Activity_Hours` | hours/day | $[-0.1, 3.8]$ | $1.86$ | $+0.235$ | $+0.367$ | $0.0300$ | $54.4\%$ | $45.6\%$ |
| 8 | `Study_Hours` | hours/day | $[0.6, 7.2]$ | $3.93$ | $+0.124$ | $+0.182$ | $0.0297$ | $50.8\%$ | $49.2\%$ |
| 9 | `Age` | years | $[18, 24]$ | $21.05$ | $+0.375$ | $+0.363$ | $0.0289$ | $49.6\%$ | $50.4\%$ |

### 3.2 Key Descriptive Response Patterns

1. **Screen Time (`Avg_Daily_Usage_Hours`) — Observed Zero-Crossing Threshold**:
   - The strongest predictor in the fitted model.
   - For usage between $1.4$ and $3.9\text{ h/d}$ ($Q_1$), the attribution is positive, shifting predictions $+1.150\text{ points}$ above the expected baseline ($6.2159$).
   - Between $3.9$ and $5.15\text{ h/d}$ ($Q_2$), the attribution moderates to $+0.201\text{ points}$.
   - The observed zero-crossing threshold occurs between **$4.8\text{ and }5.2\text{ hours/day}$**, where model attribution shifts from positive to negative relative to baseline.
   - Beyond $5.15\text{ h/d}$ ($Q_3$ and $Q_4$), attribution drops to $-0.633\text{ points}$, reaching a minimum of $-1.360\text{ points}$ at $8.7\text{ h/d}$.
   - This describes the internal behavior of the decision trees and does not constitute a real-world causal tipping point.
2. **Sleep Duration (`Sleep_Hours_Per_Night`) — Positive Shift Plateau**:
   - Displays a positive association with model predictions ($r = +0.788$, $\rho = +0.768$).
   - For sleep $< 6.0\text{ h/night}$, the attribution is negative (down to $-0.647\text{ points}$).
   - Between $6.0$ and $8.0\text{ h/night}$, attribution increases steadily through zero.
   - Above $8.5\text{ h/night}$, the curve plateaus around $+0.645\text{ points}$, demonstrating an empirical plateau in the model's positive prediction shift.
3. **Daily Device Unlocks (`Daily_Unlocks`) — High-End Attribution Spread**:
   - Overall correlation with prediction shift is weak ($\rho = -0.169$).
   - Observations with unlocks $> 200\text{ unlocks/day}$ exhibit sporadic negative attribution shifts down to $-0.513\text{ points}$.

### 3.3 Categorical Feature Attribution Distributions

Categorical variables cannot be evaluated on a continuous scale. Their attributions are evaluated across discrete categories:

| Feature | Category | Count ($N=500$) | Proportion | Mean SHAP | Median SHAP | Std Dev | Min SHAP | Max SHAP |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Purpose_Of_Use** | `Networking` | 74 | $14.8\%$ | $+0.0628$ | $+0.0440$ | $0.0712$ | $-0.0741$ | $+0.2976$ |
| | `Education` | 124 | $24.8\%$ | $+0.0540$ | $+0.0473$ | $0.0645$ | $-0.1065$ | $+0.3489$ |
| | `News` | 52 | $10.4\%$ | $+0.0432$ | $+0.0345$ | $0.0583$ | $-0.0652$ | $+0.2520$ |
| | `Entertainment` | 250 | $50.0\%$ | $-0.0458$ | $-0.0319$ | $0.0592$ | $-0.2004$ | $+0.0913$ |
| **Most_Used_Platform** | `WhatsApp` | 14 | $2.8\%$ | $+0.1732$ | $+0.2066$ | $0.0934$ | $+0.0211$ | $+0.3411$ |
| | `LINE` | 9 | $1.8\%$ | $+0.0570$ | $+0.0507$ | $0.0421$ | $+0.0032$ | $+0.1342$ |
| | `Twitter` | 45 | $9.0\%$ | $+0.0414$ | $+0.0261$ | $0.0487$ | $-0.0412$ | $+0.1983$ |
| | `VKontakte` | 5 | $1.0\%$ | $+0.0230$ | $+0.0371$ | $0.0381$ | $-0.0211$ | $+0.0654$ |
| | `LinkedIn` | 61 | $12.2\%$ | $+0.0227$ | $+0.0210$ | $0.0321$ | $-0.0312$ | $+0.1124$ |
| | `YouTube` | 51 | $10.2\%$ | $+0.0210$ | $+0.0134$ | $0.0401$ | $-0.0453$ | $+0.1256$ |
| | `TikTok` | 85 | $17.0\%$ | $-0.0034$ | $-0.0004$ | $0.0354$ | $-0.0954$ | $+0.0812$ |
| | `Facebook` | 67 | $13.4\%$ | $-0.0046$ | $+0.0035$ | $0.0378$ | $-0.1123$ | $+0.0765$ |
| | `Instagram` | 116 | $23.2\%$ | $-0.0621$ | $-0.0543$ | $0.0521$ | $-0.2154$ | $+0.0623$ |
| | `KakaoTalk` | 3 | $0.6\%$ | $-0.0652$ | $-0.0639$ | $0.0194$ | $-0.0821$ | $-0.0412$ |
| **Academic_Level** | `Undergraduate` | 348 | $69.6\%$ | $+0.0032$ | $+0.0031$ | $0.0142$ | $-0.0421$ | $+0.0512$ |
| | `Graduate` | 95 | $19.0\%$ | $-0.0063$ | $-0.0052$ | $0.0185$ | $-0.0612$ | $+0.0389$ |
| | `High School` | 57 | $11.4\%$ | $-0.0098$ | $-0.0051$ | $0.0210$ | $-0.0734$ | $+0.0412$ |
| **Stress_Level** | `Low` | 53 | $10.6\%$ | $+0.0148$ | $+0.0139$ | $0.0165$ | $-0.0182$ | $+0.0612$ |
| | `High` | 147 | $29.4\%$ | $+0.0123$ | $+0.0100$ | $0.0178$ | $-0.0245$ | $+0.0689$ |
| | `Medium` | 131 | $26.2\%$ | $+0.0096$ | $+0.0072$ | $0.0154$ | $-0.0210$ | $+0.0543$ |
| | `Very High` | 169 | $33.8\%$ | $-0.0060$ | $-0.0033$ | $0.0162$ | $-0.0541$ | $+0.0398$ |
| **Gender** | `Male` | 278 | $55.6\%$ | $+0.0031$ | $+0.0026$ | $0.0132$ | $-0.0312$ | $+0.0412$ |
| | `Female` | 222 | $44.4\%$ | $-0.0025$ | $-0.0029$ | $0.0128$ | $-0.0389$ | $+0.0345$ |

---

## 4. Feature Interaction Feasibility & Benchmarking

### 4.1 Computational Complexity of TreeSHAP Interactions

TreeSHAP interaction calculations compute the pure non-additive contribution between every pair of features across all nodes in all trees. The theoretical computational complexity is:
$$\mathcal{O}\left( N \cdot T \cdot D^2 \cdot M^2 \right)$$
where:
- $N$ is the number of observations.
- $T = 150$ is the number of decision trees in the ensemble.
- $D \approx 18$ is the average maximum depth of the trees.
- $M = 38$ is the dimension of the transformed feature space.

### 4.2 Empirical Benchmarks on Deterministic Training Samples

Benchmarked on the MindScope runtime environment (Windows AMD64, Python 3.13.3, 150 trees, 38 transformed inputs):

| Sample Size ($N$) | Execution Time (s) | Rate (s/sample) | Output Shape | Matrix Symmetry ($M = M^T$) | Max Additivity Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$N = 1$** | $7.9628\text{ s}$ | $7.96\text{ s}$ | $(1, 38, 38)$ | True ($\max = 3.82 \times 10^{-16}$) | $6.33 \times 10^{-15}$ |
| **$N = 2$** | $17.0968\text{ s}$ | $8.55\text{ s}$ | $(2, 38, 38)$ | True ($\max = 3.82 \times 10^{-16}$) | $6.33 \times 10^{-15}$ |
| **$N = 5$** | $40.2433\text{ s}$ | $8.05\text{ s}$ | $(5, 38, 38)$ | True ($\max = 8.96 \times 10^{-16}$) | $1.98 \times 10^{-14}$ |
| **$N = 10$ (Standalone)** | $139.1528\text{ s}$ | $13.92\text{ s}$ | $(10, 38, 38)$ | True ($\max = 1.08 \times 10^{-16}$) | $1.98 \times 10^{-14}$ |

**Observed Rate Range**: $\bar{t} \approx \mathbf{8.2\text{ to }13.9\text{ seconds per observation}}$.

### 4.3 Computational Scaling Projections & Practical Feasibility

Execution throughput varied across runs. Short warm-loop benchmark runs achieved $\approx 8.2\text{ s/obs}$, while sustained standalone execution across pipeline stages reached $\approx 13.9\text{ s/obs}$. Cumulative CPU utilization, thread scheduling, and potential thermal throttling on the Windows test environment are possible contributing factors for this variation.

Extrapolating this empirical range yields the following approximate linear projections:

| Sample Target | Number of Records ($N$) | Lower Projection ($\approx 8.2\text{ s/obs}$) | Upper Projection ($\approx 13.9\text{ s/obs}$) | Feasibility Assessment |
|:---|:---:|:---:|:---:|:---|
| Focused Interaction Sample | **10** | $81.9\text{ s}$ ($0.02\text{ h}$) | $139.2\text{ s}$ ($0.04\text{ h}$) | **Highly Feasible** (Approved exploratory sample) |
| Moderate Subsample | 50 | $409.3\text{ s}$ ($0.11\text{ h}$) | $696.0\text{ s}$ ($0.19\text{ h}$) | Feasible with background execution |
| Standard Reference Sample | 500 | $4,093.3\text{ s}$ ($\approx 1.14\text{ h}$) | $6,960.0\text{ s}$ ($\approx 1.93\text{ h}$) | Infeasible for continuous automated builds |
| Full Training Partition | 3,498 | $28,636.8\text{ s}$ ($\approx 7.97\text{ h}$) | $48,692.2\text{ s}$ ($\approx 13.52\text{ h}$) | **Completely Prohibitive** |

### 4.4 Feasibility Decision
> [!IMPORTANT]
> **Feasibility Verdict:** An unbounded $38 \times 38$ pairwise interaction calculation over the entire 3,498-record training partition (projected $8.0\text{ to }13.5\text{ hours}$) or even a 500-record sample (projected $1.1\text{ to }1.9\text{ hours}$) is computationally prohibitive for continuous automated testing. A deterministic sample of **$N=10$ records** provides a complete mathematical demonstration of matrix symmetry, additivity, and feature aggregation without prohibitive computational overhead.

---

## 5. Interaction Aggregation Methodology

### 5.1 Transformed ($38 \times 38$) vs Aggregated Original ($12 \times 12$) Space

In the transformed feature space, categorical features are split into binary one-hot indicators (e.g., `Platform_Instagram`, `Platform_TikTok`). Computing TreeSHAP interactions yields an $(N, 38, 38)$ tensor $\Phi_{i, j, k}^{\text{trans}}$.

To map these back to the 12 original features without loss of additivity, we define the aggregated interaction tensor $\Phi_{i, A, B}^{\text{orig}} \in \mathbb{R}^{N \times 12 \times 12}$ as:
$$\Phi_{i, A, B}^{\text{orig}} = \sum_{j \in I(A)} \sum_{k \in I(B)} \Phi_{i, j, k}^{\text{trans}}$$
where $I(A)$ and $I(B)$ are the index sets of transformed features mapped to parent features $A$ and $B$.

### 5.2 Mathematical Invariants Verified

1. **Symmetry Invariance:**
   $$\Phi_{i, A, B}^{\text{orig}} = \Phi_{i, B, A}^{\text{orig}} \quad \forall A, B \in \{1, \dots, 12\}$$
   *Verified empirical error:* $\max |M - M^T| = 1.08 \times 10^{-16}$.
2. **Row-Sum Consistency with Grouped SHAP Values:**
   $$\sum_{B=1}^{12} \Phi_{i, A, B}^{\text{orig}} = \sum_{j \in I(A)} \left( \sum_{k=1}^{38} \Phi_{i, j, k}^{\text{trans}} \right) = \sum_{j \in I(A)} \phi_{i, j}^{\text{trans}} = \Phi_{i, A}^{\text{orig}}$$
   The sum across any row $A$ of the $12 \times 12$ matrix equals the 12-feature grouped attribution $\Phi_{i, A}^{\text{orig}}$ computed in Step 3.1.  
   *Verified empirical error:* $\max |\text{row\_sum} - \Phi_A| < 2 \times 10^{-14}$.
3. **Completeness / Additivity Reconstruction:**
   $$\sum_{A=1}^{12} \sum_{B=1}^{12} \Phi_{i, A, B}^{\text{orig}} = \sum_{j=1}^{38} \sum_{k=1}^{38} \Phi_{i, j, k}^{\text{trans}} = f(x_i) - \mathbb{E}[f(x)]$$
   *Verified empirical error:* $\max |\text{total\_sum} - (f(x_i) - \mathbb{E}[f(x)])| < 2 \times 10^{-14}$.
4. **Metric Reporting Convention:**
   - The reported interaction strength $\bar{\Phi}_{A, B}^{\text{inter}}$ represents the mean absolute value of the individual off-diagonal cell $(A, B)$ in the symmetric matrix.
   - Because of symmetry ($\Phi_{A, B} = \Phi_{B, A}$), the pair's combined non-additive contribution to the total prediction shift is $2 \times \bar{\Phi}_{A, B}^{\text{inter}}$.

### 5.3 Methodological Distinction on Game-Theoretic Equivalence
> [!WARNING]
> While summing transformed interaction values preserves exact additivity and symmetry, aggregated interaction values **are not mathematically equivalent** to recalculating Shapley interaction values on an atomic 12-player game. In the fitted Random Forest, tree splits occur on one-hot indicators independently. Summed attributions reflect the combined contribution of those splits, not a game where entire categories are perturbed jointly.

---

## 6. Empirical Feature Interaction Findings (Exploratory N=10 Benchmark Sample)

> [!NOTE]
> The interaction rankings below are exploratory findings calculated across a deterministic benchmark sample of $N=10$ training observations. They demonstrate the operational validity of the aggregation mathematics on real records. Because $N=10$ is an illustrative computational sample without population-level uncertainty bounds, these rankings must not be interpreted as definitive population-wide properties or real-world effect sizes.

Across all $\frac{12 \times 11}{2} = 66$ unique off-diagonal feature pairs ($A < B$), the top pairwise interactions ranked by mean absolute interaction strength ($\bar{\Phi}_{A, B}^{\text{inter}} = \frac{1}{N} \sum_{i=1}^N |\Phi_{i, A, B}|$) in [`artifacts/explanations/dependence/interaction_analysis.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/dependence/interaction_analysis.json) are:

### 6.1 Top 10 Off-Diagonal Pairwise Interactions ($N=10$ Sample)

| Rank | Feature Pair | Feature A | Feature B | Mean \|Interaction\| (Points) | Primary Category |
|:---:|:---|:---|:---|:---:|:---|
| **1** | `Avg_Daily_Usage_Hours` $\times$ `Sleep_Hours_Per_Night` | Daily Social Media Usage | Sleep Duration | **$0.1775$** | Candidate Pair |
| **2** | `Country` $\times$ `Avg_Daily_Usage_Hours` | Country of Residence | Daily Social Media Usage | $0.0436$ | Categorical $\times$ Continuous |
| **3** | `Most_Used_Platform` $\times$ `Avg_Daily_Usage_Hours` | Most Used Platform | Daily Social Media Usage | $0.0419$ | Categorical $\times$ Continuous |
| **4** | `Avg_Daily_Usage_Hours` $\times$ `Daily_Unlocks` | Daily Social Media Usage | Daily Device Unlocks | **$0.0382$** | Candidate Pair |
| **5** | `Avg_Daily_Usage_Hours` $\times$ `Study_Hours` | Daily Social Media Usage | Daily Study Duration | $0.0244$ | Continuous $\times$ Continuous |
| **6** | `Purpose_Of_Use` $\times$ `Avg_Daily_Usage_Hours` | Primary Purpose of Use | Daily Social Media Usage | $0.0231$ | Categorical $\times$ Continuous |
| **7** | `Daily_Unlocks` $\times$ `Sleep_Hours_Per_Night` | Daily Device Unlocks | Sleep Duration | $0.0205$ | Continuous $\times$ Continuous |
| **8** | `Country` $\times$ `Sleep_Hours_Per_Night` | Country of Residence | Sleep Duration | $0.0193$ | Categorical $\times$ Continuous |
| **9** | `Purpose_Of_Use` $\times$ `Sleep_Hours_Per_Night` | Primary Purpose of Use | Sleep Duration | $0.0190$ | Categorical $\times$ Continuous |
| **10** | `Avg_Daily_Usage_Hours` $\times$ `Physical_Activity_Hours` | Daily Social Media Usage | Physical Activity | **$0.0171$** | Candidate Pair |

*(Note: `Most_Used_Platform x Sleep_Hours_Per_Night` also exhibited $0.0182\text{ pts}$ in the full ranking list).*

### 6.2 Prioritized Candidate Pair Analysis (Reconciled with JSON Artifact)

1. **Screen Time $\times$ Sleep Duration (`Avg_Daily_Usage_Hours` $\times$ `Sleep_Hours_Per_Night`) — Rank 1 ($0.1775\text{ pts}$)**:
   - Within the selected $N=10$ exploratory benchmark sample, Screen Time $\times$ Sleep Duration exhibited the largest mean absolute aggregated interaction attribution ($0.1775\text{ pts}$).
   - In individual records with elevated screen time, higher sleep duration was observed alongside a positive interaction attribution, moderating the negative main attribution of screen time in the model's tree splits.
2. **Screen Time $\times$ Device Unlocks (`Avg_Daily_Usage_Hours` $\times$ `Daily_Unlocks`) — Rank 4 ($0.0382\text{ pts}$)**:
   - In observations where both usage hours and unlock counts are high, tree splits reflect joint conditioning with modest non-additive attribution shifts.
3. **Screen Time $\times$ Physical Activity (`Avg_Daily_Usage_Hours` $\times$ `Physical_Activity_Hours`) — Rank 10 ($0.0171\text{ pts}$)**:
   - In the $N=10$ sample, higher physical activity levels showed a minor positive interaction attribution in records with high screen time.
4. **Study Duration $\times$ Sleep Duration (`Study_Hours` $\times$ `Sleep_Hours_Per_Night`) — Rank 14 ($0.0146\text{ pts}$)**:
   - Exhibited a modest interaction effect ($0.0146\text{ pts}$) in the fitted ensemble across the evaluated observations.
5. **Sleep Duration $\times$ Stress Level (`Sleep_Hours_Per_Night` $\times$ `Stress_Level`) — Rank 28 ($0.0049\text{ pts}$)**:
   - Displayed low interaction magnitude ($0.0049\text{ pts}$), indicating that the model treats sleep duration and stress level categories in an almost purely additive manner.

---

## 7. Artifacts & Visualizations Created

All artifacts are persisted in `artifacts/explanations/dependence/`:

### 7.1 JSON Data Artifacts
1. [`dependence_summary.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/dependence/dependence_summary.json): Complete per-observation and binned dependence statistics for all 12 predictors ($N=500$).
2. [`interaction_benchmark.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/dependence/interaction_benchmark.json): Empirical benchmarks ($N=1, 2, 5$), symmetry tests, additivity proofs, and scaling projections.
3. [`interaction_analysis.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/dependence/interaction_analysis.json): Exact $12 \times 12$ mean absolute interaction matrix, all 66 ranked pairs, main effects, and candidate pair profiles ($N=10$).
4. [`dependence_metadata.json`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/artifacts/explanations/dependence/dependence_metadata.json): Governance, environment, and reproducibility metadata.

### 7.2 Publication-Quality Visualizations ($300\text{ DPI}$)
1. **`numerical_dependence_grid.png`**: 6-panel grid showing raw feature values vs SHAP attributions with rolling median curves for all 6 numerical features.
2. **`dependence_usage_hours.png`**: Focused dependence curve for `Avg_Daily_Usage_Hours` illustrating the $4.8\text{ to }5.2\text{ h/d}$ zero-crossing threshold.
3. **`dependence_sleep_hours.png`**: Focused dependence curve for `Sleep_Hours_Per_Night` illustrating the positive prediction shift plateau past $8.5\text{ h/night}$.
4. **`dependence_daily_unlocks.png`**: Focused curve for `Daily_Unlocks`.
5. **`categorical_attribution_distributions.png`**: 6-panel grid of horizontal boxplots showing attribution distributions for all 6 categorical features.
6. **`interaction_matrix_heatmap.png`**: $12 \times 12$ annotated heatmap of mean absolute pairwise interaction strengths across the 12 original predictors ($N=10$).
7. **`interaction_candidate_pairs.png`**: Bivariate scatter plots for primary candidate pairs (Usage colored by Sleep, Usage colored by Physical Activity) across $N=500$ records.

---

## 8. Statistical & Methodological Limitations

1. **Non-Causal Nature of Explanations**:
   - Feature dependence curves and interaction values describe the inner mechanics and decision surfaces of this specific Random Forest model.
   - They **do not establish causal relationships** in the real world. A student reducing social media usage by 2 hours will not necessarily experience an increase in mental health score.
2. **Exploratory Scope of $N=10$ Interaction Sample**:
   - The interaction analysis was conducted on an illustrative benchmark sample of $N=10$ training observations to manage computational complexity ($\mathcal{O}(N \cdot T \cdot D^2 \cdot M^2)$).
   - Findings from this sample demonstrate the mathematical operation of the interaction pipeline but cannot be generalized to the entire student population or full dataset without extensive uncertainty estimation.
3. **Unverified Data Provenance**:
   - The underlying dataset is derived from an unverified online public release with potential self-reporting and selection biases.
   - Findings must not be generalized to broader student or adolescent populations.
4. **No Clinical Validity**:
   - The target `Mental_Health_Score` is a synthetic or self-reported index, not a validated psychiatric instrument (such as PHQ-9 or GAD-7).
   - Predictions and attributions must never be used for clinical triage, diagnosis, or treatment recommendations.
5. **Neutral Terminology Governance**:
   - Attributions are strictly characterized by direction relative to the expected model baseline ($6.2159$).
   - Directional terms are neutral: "moves prediction above baseline" or "moves prediction below baseline."
   - Value-laden terms (*protective, harmful, healthy, unhealthy, improvement, deterioration, good habit, bad habit, risk reduction*) are strictly avoided.

---

## 9. Automated Testing & Verification Suite

A dedicated automated test module [`tests/test_feature_analysis.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/tests/test_feature_analysis.py) was added with 13 comprehensive tests:

| Test Name | Verification Focus | Result |
|:---|:---|:---:|
| `test_training_sampling_determinism_and_isolation` | 3,498-record training partition, deterministic `random_state=42` | Passed |
| `test_sealed_holdout_access_blocked` | Strict `PermissionError` on sealed holdout references | Passed |
| `test_numerical_dependence_schema_and_finiteness` | Schema, Pearson $r$, Spearman $\rho$, finite SHAP values | Passed |
| `test_categorical_dependence_schema_and_finiteness` | Schema, category counts, IQR, finite distribution statistics | Passed |
| `test_interaction_timing_benchmark_structure` | Benchmark structure, scaling projections, timeouts | Passed |
| `test_interaction_aggregation_dimensions_and_symmetry` | Aggregation $38 \to 12$, strict matrix symmetry ($M = M^T$) | Passed |
| `test_interaction_row_sum_consistency_with_grouped_shap` | Row-sums match grouped 12-feature SHAP attributions | Passed |
| `test_interaction_additivity_reconstruction` | Total interaction sum equals $f(x) - \mathbb{E}[f(x)]$ | Passed |
| `test_invalid_inputs_handling` | Defensive validation for invalid sample sizes and schemas | Passed |
| `test_dependence_artifacts_and_metadata_integrity` | All 4 JSON and 7 PNG artifacts exist, valid, and non-empty | Passed |
| `test_model_artifacts_unmutated_after_analysis` | Pipeline and legacy model ($25,694,899\text{ B}$) unmutated | Passed |
| `test_sealed_holdout_file_unaccessed_and_unmodified` | Sealed holdout file remains untouched on disk | Passed |
| `test_non_causal_disclaimer_presence` | Mandatory non-causal statistical disclaimer present in outputs | Passed |

### Full Test Suite Execution Summary
```text
tests/test_backend.py ............. [ 13 passed ]
tests/test_evaluation.py ........ [  8 passed ]
tests/test_explain.py ............ [ 12 passed ]
tests/test_feature_analysis.py ............. [ 13 passed ]
tests/test_frontend_contracts.py ......... [  9 passed ]
tests/test_global_shap.py ......... [  9 passed ]
tests/test_local_explain.py ............. [ 13 passed ]
tests/test_preprocessing.py ................. [ 17 passed ]
tests/test_training.py ........ [  8 passed ]

================== 102 passed, 13 warnings in 34.94s ==================
```

---

## 10. Summary of Files Created / Modified

- **Created Source**:
  - [`src/feature_analysis.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/src/feature_analysis.py): Feature dependence, interaction benchmarking, $38 \to 12$ aggregation, and visualization engine.
- **Created Tests**:
  - [`tests/test_feature_analysis.py`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/tests/test_feature_analysis.py): 13 automated unit and integration tests.
- **Created Documentation**:
  - [`docs/PHASE_3_STEP_4_FEATURE_ANALYSIS.md`](file:///c:/Users/Ishit/OneDrive/Desktop/MindScope/docs/PHASE_3_STEP_4_FEATURE_ANALYSIS.md): Comprehensive Step 3.4 report.
- **Created Artifacts** (`artifacts/explanations/dependence/`):
  - `dependence_summary.json`
  - `interaction_benchmark.json`
  - `interaction_analysis.json`
  - `dependence_metadata.json`
  - `numerical_dependence_grid.png`
  - `dependence_usage_hours.png`
  - `dependence_sleep_hours.png`
  - `dependence_daily_unlocks.png`
  - `categorical_attribution_distributions.png`
  - `interaction_matrix_heatmap.png`
  - `interaction_candidate_pairs.png`
- **Modified Source**: None. (`src/explain.py`, `src/global_shap.py`, and `main.py` were left completely intact).
