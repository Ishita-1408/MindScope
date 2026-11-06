# MindScope — Student Mental Health Analytics & Predictive Intelligence Platform
## Empirical Investigation of the Dataset and Target Variable (`Mental_Health_Score`)

- **Document Version:** 1.1.0
- **Status:** Refined Empirical Investigation Report
- **Target Variable:** `Mental_Health_Score`
- **Primary Data Source:** `Student Social Media And Mental Health Impact.csv` (5,000 records, 13 attributes)
- **Investigator:** MindScope Analytics Core Team
- **Date:** November 2026
- **Associated Documents:** `docs/PROJECT_REQUIREMENTS.md`, `docs/SYSTEM_ARCHITECTURE.md`, `mindscope_audit_report.md`

---

## 1. Executive Summary & Investigation Context

Before undertaking model training or API implementation for **MindScope**, this read-only investigation was conducted to analyze the primary dataset and its target variable: `Mental_Health_Score`. 

The objective of this investigation is to evaluate primary evidence from the dataset and codebase to determine:
1. The mathematical and distributional characteristics of `Mental_Health_Score`.
2. The nature of statistical relationships between predictors and the target variable.
3. Whether the target score is mathematically derived from the predictors or contains unexplained variance under evaluated models.
4. What provenance metadata exists across the codebase and documentation.
5. Realistic input domain boundaries, structural data couplings, and out-of-distribution risks.

### Summary of Key Findings
- **Discretization to 0.1 Increments:** `Mental_Health_Score` is not an unconstrained continuous float. It spans exactly **59 discrete values** between **3.6** and **9.4**, spaced at uniform $0.1$ intervals.
- **Concentration at Integer Values:** Scores ending in `.0` (`4.0`, `5.0`, `6.0`, `7.0`, `8.0`, `9.0`) account for **27.84%** (1,392 of 5,000) of all records. While this non-uniform concentration is empirically notable, determining whether it stems from human survey rounding, questionnaire aggregation, or a programmatic generation artifact requires further information.
- **Statistical Directionality vs. Construct Meaning:** High scores are strongly associated with higher sleep duration, more study hours, lower daily screen time, fewer device unlocks, and lower perceived stress. While these associations establish clear statistical directionality, they do not establish the original operational definition or psychometric construct of `Mental_Health_Score`, which remains undocumented.
- **Unexplained Variance Under Evaluated Models:** Fitting Ordinary Least Squares (OLS) regression yields an $R^2$ of $0.731$ ($\text{RMSE} = 0.665$). Evaluated non-linear tree ensembles (Random Forest and Gradient Boosting) achieve a 5-fold cross-validation $R^2$ of **$0.878 \pm 0.008$** ($\text{RMSE} \approx 0.45$). The remaining **$1 - R^2 \approx 12.2\%$** represents unexplained variance under the evaluated models and validation procedures; its underlying causes (e.g., omitted variables, unmodeled interactions, measurement error, or stochastic noise) cannot be conclusively identified from this experiment alone.
- **Exact Structural Couplings:** The dataset exhibits exact deterministic relationships:
  - `Age` is strictly partitioned by `Academic_Level` with zero overlap (High School $= 18$; Undergraduate $= 19–22$; Graduate $= 23–24$).
  - For the 7 major global platforms, `Purpose_Of_Use` is 100% deterministically paired with the platform (Facebook with `Networking`; Instagram, TikTok, Snapchat with `Entertainment`; LinkedIn, YouTube with `Education`; Twitter with `News`).
  - Regional messaging platforms map 100% to single countries (LINE $\rightarrow$ Japan; VKontakte $\rightarrow$ Russia; KakaoTalk $\rightarrow$ South Korea; WeChat $\rightarrow$ China).
- **Unverified Provenance:** Comprehensive searches across repository files, Colab metadata, and documentation confirm that no source citation, questionnaire instrument, author attribution, collection date, or institutional oversight (IRB) exists. The codebase originated as a classroom tutorial/demo titled *"Mansik-Santulan-Score"* with a greeting string referencing *"Sheryians AI School"*.
- **Informational & Educational Suitability:** The dataset provides an effective, well-conditioned regression problem for evaluating machine learning pipelines, cross-validation protocols, Explainable AI (SHAP), and REST API deployment, provided MindScope explicitly avoids clinical or diagnostic claims.

---

## 2. Target Variable Distribution & Discretization Analysis

### 2.1 Parametric & Non-Parametric Summary Statistics

Evaluating `Mental_Health_Score` across all 5,000 records yields the following statistical profile:

| Metric | Empirical Value | Analytical Interpretation |
| :--- | :---: | :--- |
| **Sample Size ($N$)** | 5,000 (4,998 unique) | Complete dataset without missing entries. |
| **Minimum Value** | **3.60** | Bounded lower extreme observed in sample. |
| **Maximum Value** | **9.40** | Bounded upper extreme observed in sample. |
| **Mean ($\mu$)** | **6.2310** | Central tendency centered slightly above mid-scale. |
| **Median ($Q_2$)** | **6.1000** | Close alignment with the mean indicates near-symmetry. |
| **Standard Deviation ($\sigma$)** | **1.2787** | Moderate dispersion across the score range. |
| **Interquartile Range ($\text{IQR}$)** | $5.10 - 7.10$ ($2.00$) | 50% of the population falls between 5.10 and 7.10. |
| **Skewness** | **$+0.2066$** | Mild right skewness; approximately symmetric distribution. |
| **Kurtosis (Fisher/Excess)** | **$-0.8033$** | Platykurtic distribution (flatter than a normal distribution, lighter tails). |

```
Quantile Progression:
 1st Percentile :  4.00        50th Percentile (Median): 6.10
 5th Percentile :  4.20        75th Percentile (Q3)    : 7.10
 10th Percentile:  4.60        90th Percentile         : 8.00
 25th Percentile (Q1): 5.10    95th Percentile         : 8.40
                               99th Percentile         : 8.90
```

### 2.2 Discretization and Decimals Analysis
- **Decimal Precision:** Every record in the dataset is recorded to **exactly 1 decimal digit** (100% of samples format as `X.Y`).
- **Unique Values:** There are exactly **59 unique values** in the dataset:
  $$\{3.6, 3.7, 3.8, 3.9, 4.0, 4.1, \dots, 9.3, 9.4\}$$
  This spans the range from $3.6$ to $9.4$ in uniform steps of $0.10$:
  $$\text{Count} = \frac{9.4 - 3.6}{0.1} + 1 = 59$$
  All 59 possible tenths-values within this range appear at least once in the dataset.

### 2.3 Tenths-Digit Distribution
Evaluating the distribution of the tenths digit ($\text{digit} = (\text{Score} \times 10) \pmod{10}$) reveals an empirical concentration at `.0`:

| Tenths Digit | Observed Count | Percentage of Dataset |
| :---: | :---: | :---: |
| **.0** | **1,392** | **27.84%** |
| **.1** | 491 | 9.82% |
| **.2** | 444 | 8.88% |
| **.3** | 368 | 7.36% |
| **.4** | 309 | 6.18% |
| **.5** | 331 | 6.62% |
| **.6** | 343 | 6.86% |
| **.7** | 360 | 7.20% |
| **.8** | 461 | 9.22% |
| **.9** | 501 | 10.02% |

**Most Frequent Individual Values:**
1. `6.0`: 424 occurrences (8.48%)
2. `5.0`: 346 occurrences (6.92%)
3. `7.0`: 315 occurrences (6.30%)
4. `8.0`: 203 occurrences (4.06%)
5. `5.9`: 134 occurrences (2.68%)

**Observation & Context:**  
Scores ending in `.0` account for over a quarter of the dataset. While non-uniform distributions across fractional digits can occur in survey responses (e.g., preference for whole numbers) or synthetic generation procedures (e.g., base integer generation with intermittent noise), this concentration cannot be definitively attributed to a specific cause without documentation of the data collection or generation protocol. It remains an empirical observation requiring further investigation.

---

## 3. Target-Feature Statistical Relationships

### 3.1 Bivariate Numerical Associations

Linear (Pearson $r$) and monotonic (Spearman rank $\rho$) correlations between numerical predictors and `Mental_Health_Score` were calculated alongside 95% Fisher-transformed confidence intervals and asymptotic $p$-values:

| Feature Identifier | Pearson $r$ | 95% Confidence Interval | Spearman $\rho$ | $p$-value | Statistical Direction & Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Avg_Daily_Usage_Hours` | **$-0.8165$** | $[-0.8255, -0.8070]$ | **$-0.8401$** | $< 10^{-300}$ | Strong negative association. Higher screen time tracks lower scores. |
| `Daily_Unlocks` | **$-0.7909$** | $[-0.8010, -0.7802]$ | **$-0.8128$** | $< 10^{-300}$ | Strong negative association. Higher unlocks track lower scores. |
| `Sleep_Hours_Per_Night` | **$+0.7664$** | $[+0.7547, +0.7776]$ | **$+0.7869$** | $< 10^{-300}$ | Strong positive association. Adequate sleep tracks higher scores. |
| `Study_Hours` | **$+0.7526$** | $[+0.7404, +0.7644]$ | **$+0.7567$** | $< 10^{-300}$ | Strong positive association. More study hours track higher scores. |
| `Physical_Activity_Hours`| **$+0.5214$** | $[+0.5009, +0.5413]$ | **$+0.5257$** | $< 10^{-300}$ | Moderate positive association. Regular exercise tracks higher scores. |
| `Age` | **$+0.1008$** | $[+0.0733, +0.1281]$ | **$+0.0965$** | $9.13 \times 10^{-13}$| Weak positive correlation within the observed $[18, 24]$ window. |

#### Methodological Observations:
1. **Collinearity Among Predictors:** Several predictors are strongly inter-correlated (e.g., `Avg_Daily_Usage_Hours` and `Daily_Unlocks` have a Pearson $r = 0.887$). Multivariable models must account for variance inflation and parameter instability.
2. **Monotonic Alignment:** For all features, Spearman rank correlation $\rho$ closely mirrors or slightly exceeds Pearson $r$, indicating consistent monotonic trends across the distribution rather than distortions driven by isolated outliers.
3. **Statistical Directionality vs. Construct Meaning:** These correlations confirm clear statistical directionality (protective lifestyle habits correlate positively with the score; intensive screen use and stress correlate negatively). However, statistical directionality does not define the psychometric construct or operational meaning of `Mental_Health_Score`, which remains unverified.

### 3.2 Categorical Associations & ANOVA

One-way Analysis of Variance (ANOVA) and effect size ($\eta^2$, proportion of total variance accounted for by group membership) were computed across categorical features:

| Categorical Feature | Distinct Levels | ANOVA $F$-Statistic | $p$-value | Effect Size ($\eta^2$) | Group Variance Summary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Stress_Level`** | 4 | **2,685.02** | $< 10^{-300}$ | **0.6172 (61.7%)** | Dominant categorical factor; stepwise score decline across tiers. |
| **`Country`** | 111 | **9.90** | $3.03 \times 10^{-142}$ | **0.1822 (18.2%)** | High cardinality; small sample counts inflate between-group variance. |
| **`Most_Used_Platform`** | 12 | **22.21** | $6.43 \times 10^{-45}$ | **0.0467 (4.7%)** | Moderate between-group variance; influenced by regional platforms. |
| **`Academic_Level`** | 3 | **26.02** | $5.72 \times 10^{-12}$ | **0.0103 (1.0%)** | Slight upward trend from High School to Graduate level. |
| **`Purpose_Of_Use`** | 4 | **14.59** | $1.84 \times 10^{-9}$ | **0.0087 (0.9%)** | Entertainment users average slightly lower scores than Networking users. |
| **`Gender`** | 2 | **5.74** | $1.67 \times 10^{-2}$ | **0.0011 (0.1%)** | Minor difference (Male: $6.27 \pm 1.26$; Female: $6.19 \pm 1.30$). |

#### Breakdown of Score by Stress Level:
- **Low Stress ($N=644$):** $\mu = 7.84 \pm 0.72$ (Min: 5.8, Max: 9.4)
- **Medium Stress ($N=1,295$):** $\mu = 7.14 \pm 0.92$ (Min: 4.7, Max: 9.2)
- **High Stress ($N=1,440$):** $\mu = 6.03 \pm 0.83$ (Min: 3.7, Max: 9.1)
- **Very High Stress ($N=1,621$):** $\mu = 5.05 \pm 0.66$ (Min: 3.6, Max: 7.1)

Perceived stress level alone accounts for **61.7%** of the sample variance in `Mental_Health_Score`.

---

## 4. Model Performance & Evaluation Protocol

### 4.1 Multivariable Linear Baselines

To examine whether linear combinations of the available predictors can reconstruct the score, Ordinary Least Squares (OLS) regressions were fitted:

#### 1. Simple OLS (6 Numerical Lifestyle Features + Ordinal Stress)
Fitting OLS on `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Study_Hours`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night`, and ordinal `Stress_Code`:
$$\hat{y} = 6.075 - 0.250(\text{Usage}) - 0.0012(\text{Unlocks}) + 0.085(\text{Study}) + 0.023(\text{Activity}) + 0.260(\text{Sleep}) - 0.212(\text{Stress})$$
- **$R^2$:** **$0.7093$** (70.93% variance explained)
- **Residual Standard Error (RMSE):** **$0.6898$**
- **Residual Range:** $[-2.55, +2.20]$
- **Residual Distribution:** Skewness $= -0.160$, Kurtosis $= +0.063$. Residuals are approximately centered around 0 with no extreme skewness.

#### 2. Full OLS with One-Hot Encoded Categories
- **$R^2$:** **$0.7309$**
- **Residual Standard Error (RMSE):** **$0.6654$**
Adding demographic, country, and platform categories increases explained variance by only $\approx 2.16\%$.

### 4.2 Non-Linear Ensemble Cross-Validation vs. Holdout Evaluation

To assess non-linear modeling capacity and evaluate generalizability, candidate estimators were evaluated:

| Estimator Pipeline | 5-Fold Cross-Validation $R^2$ (Mean $\pm$ Std) | Holdout Test $R^2$ (30% Split) | Holdout Test MAE | Holdout Test RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | $0.728 \pm 0.012$ | 0.7398 | 0.5362 | 0.6760 |
| **Random Forest (Default, unbounded depth)** | $0.8782 \pm 0.0082$ | 0.8776 | 0.3472 | 0.4637 |
| **Gradient Boosting (300 trees, max depth 6)** | $0.8787 \pm 0.0092$ | 0.8765 | 0.3490 | 0.4658 |
| **Random Forest (Tuned, max depth 15)** | $0.8685 \pm 0.0075$ | 0.8650 | 0.3689 | 0.4869 |

#### Validation Methodology Safeguards:
1. **Fold-Level Preprocessing Encapsulation:** To prevent data leakage, all preprocessing steps (scaling parameters, one-hot category vocabularies, and imputers) must be fitted strictly inside each training fold during cross-validation, and fitted only on `X_train` during holdout evaluation.
2. **Distinction Between CV and Holdout Performance:** 5-fold cross-validation estimates out-of-sample performance across multiple training folds ($R^2 \approx 0.878$). Holdout evaluation measures performance on a single fixed partition ($N=1,500$).
3. **Unbiased Model Selection & Nested CV:** Hyperparameter tuning must not share the same folds used for performance reporting. The system requires nested cross-validation or an equivalent unbiased selection procedure to prevent selection bias.
4. **Protection Against Holdout Overfitting:** The holdout test set must be reserved strictly for final model evaluation. It must not be repeatedly consulted for model selection or hyperparameter tuning, which would cause data snooping bias.
5. **Evaluation Metric Standards:** Primary regression metrics ($R^2$, MAE, RMSE) are calculated on continuous, unrounded predictions. Supplementary tolerance metrics ($\text{Tol}_{0.1}$, $\text{Tol}_{0.2}$) provide descriptive error diagnostics without claiming clinical or diagnostic validity.

### 4.3 Interpretation of Unexplained Variance
The evaluated non-linear ensembles achieve a cross-validated $R^2$ of approximately $0.878$, leaving approximately **$1 - R^2 \approx 12.2\%$** of target variance unexplained ($\text{RMSE} \approx 0.45$ score points).

> [!NOTE]
> **Scientific Interpretation of $1 - R^2$:**  
> The remaining $12.2\%$ represents **unexplained variance under the evaluated model architectures, selected features, and validation protocols**. This empirical result indicates that the target is not an exact deterministic formula of the recorded features. However, the specific source of this unexplained variance cannot be determined from this experiment alone; it could result from unmodeled feature interactions, omitted variables, functional misspecification, measurement noise, or inherent stochasticity.

---

## 5. Structural Dependencies & Coupling Analysis

Cross-tabulating categorical and demographic attributes reveals several exact structural couplings:

```
                      STRUCTURAL DATASET COUPLINGS IDENTIFIED
┌─────────────────────────┬──────────────────────────┬─────────────────────────┐
│     ACADEMIC LEVEL      │   MOST USED PLATFORM     │   REGIONAL PLATFORMS    │
│    VS. STUDENT AGE      │   VS. PURPOSE OF USE     │      VS. COUNTRY        │
├─────────────────────────┼──────────────────────────┼─────────────────────────┤
│ • High School:          │ • Facebook:              │ • LINE:                 │
│   Age 18 exactly (100%) │   100% Networking        │   100% Japan (50/50)    │
│ • Undergraduate:        │ • Instagram, TikTok,     │ • VKontakte:            │
│   Ages 19–22 (100%)     │   Snapchat:              │   100% Russia (40/40)   │
│ • Graduate:             │   100% Entertainment     │ • KakaoTalk:            │
│   Ages 23–24 (100%)     │ • LinkedIn, YouTube:     │   100% S. Korea (36/36) │
│                         │   100% Education         │ • WeChat:               │
│ [Zero age overlap]      │ • Twitter: 100% News     │   100% China (36/36)    │
└─────────────────────────┴──────────────────────────┴─────────────────────────┘
```

### 5.1 Observed Structural Relationships
1. **Age Strictly Partitioned by Academic Level:**  
   - High School students are strictly 18 years old ($N=450$; min=18, max=18).
   - Undergraduates are strictly 19 to 22 years old ($N=3,632$; min=19, max=22).
   - Graduate students are strictly 23 to 24 years old ($N=918$; min=23, max=24).
   There is zero age overlap across the three educational tiers.
2. **Platform Strictly Determines Purpose (Top 7 Major Platforms):**  
   - 100% of Facebook users reported `Networking` (719/719).
   - 100% of Instagram (1,130), TikTok (918), and Snapchat (420) users reported `Entertainment`.
   - 100% of LinkedIn (523) and YouTube (491) users reported `Education`.
   - 100% of Twitter users reported `News` (462/462).
   Only multi-purpose messaging applications (WhatsApp, LINE, VKontakte, KakaoTalk, WeChat) exhibit varied purpose selections.
3. **Regional Platforms Strictly Map to Single Countries:**  
   - 100% of LINE users are in Japan (50/50).
   - 100% of VKontakte users are in Russia (40/40).
   - 100% of KakaoTalk users are in South Korea (36/36).
   - 100% of WeChat users are in China (36/36).

### 5.2 Implications for Modeling, Feature Importance & Generalization
- **Feature Redundancy & Multicollinearity:** Including both `Age` and `Academic_Level`, or both `Most_Used_Platform` and `Purpose_Of_Use`, introduces structural redundancy. In linear models, this inflates variance; in tree-based models, split attribution is divided across collinear features.
- **Distortion of Feature Importance:** In tree-based models and SHAP attributions, credit for predictive splits may be shared or arbitrarily assigned between collinear feature pairs (e.g., between `Most_Used_Platform` and `Purpose_Of_Use`), diluting individual importance rankings.
- **Generalization Limitations:** A model trained on these rigid relationships may produce unreliable predictions if exposed to out-of-distribution inputs that violate these couplings (e.g., a 25-year-old undergraduate or an Instagram user whose primary purpose is education).
- **Modeling Recommendation:** These features should not be automatically dropped without careful trade-off analysis. Retaining them maintains compatibility with existing baseline pipelines and allows multi-perspective exploratory analysis, but their collinearity must be explicitly noted in interpretability reports. In Phase 2, the modeling protocol mandates an empirical comparison between a full-feature pipeline (retaining both `Most_Used_Platform` and `Purpose_Of_Use`) and a reduced-feature variant, evaluating predictive performance, feature importance stability, and SHAP attribution clarity.

---

## 6. Repository Provenance Search

A search across all repository files (`README.md`, `ML_Project.ipynb`, `ML Project.html`, `main.py`, `script.js`, `index.html`) was performed to evaluate data origin metadata:

| Investigation Scope | Artifact Examined | Verified Findings |
| :--- | :--- | :--- |
| **Dataset Provenance** | `README.md` | Contains only `# Mansik-Santulan-Score` (UTF-16LE, 52 bytes). No author, license, or collection source. |
| **Notebook Documentation** | `ML_Project.ipynb` | Cell 0 describes shape (5,000 students, 13 columns) but provides **no URL, author, survey citation, or data collection methodology**. Cell 68 shows `pip install --upgrade scikit-learn==1.9.0` run in Google Colab. |
| **API Endpoints** | `main.py` | Line 49 returns `{'Welcome to Sheryians AI School Guys'}` on root route `/`. |
| **Deployment Blueprint** | `ML Project.html` | Tutorial blueprint titled *"Build & Ship an ML-Powered Web App: A five-sheet construction guide"*. Notes include on-camera tutorial advice (*"On camera: send a bad request... this is the moment that sells Pydantic to viewers"*). |
| **Frontend Assets** | `script.js` | Line 4 hardcodes remote API: `https://mansik-santulan-score.onrender.com`. |

### Unverified Elements:
- The survey questionnaire items or psychometric instruments used (if any).
- Whether the data was collected from human participants or synthetically generated.
- The institutional identity, ethical review (IRB), or consent status of respondents.
- The date of data collection.

---

## 7. Age and Input Domain Specifications

### 7.1 Distinguishing Observed Ranges, Validated Ranges, and API Safeguards

To prevent invalid inputs and unwarranted model extrapolation, MindScope distinguishes three distinct domain definitions:
1. **Observed Training Domain:** The empirical range of values present in `Student Social Media And Mental Health Impact.csv`.
2. **Validated Model Range:** The envelope where model predictions are supported by empirical training density.
3. **API Accepted Range:** Input boundaries enforced at the API schema level as engineering safeguards against malformed data.

| Feature Identifier | Observed Training Range | Validated Model Range | Recommended Initial API Bounds | Boundary Classification |
| :--- | :---: | :---: | :---: | :--- |
| **`Age`** | $[18, 24]$ | **$[18, 24]$** | **$[18, 24]$** | **Initial deployment bound.** Strictly matches observed training data. Future extensions to non-traditional students ($25+$) must be treated as out-of-distribution until validated. |
| **`Avg_Daily_Usage_Hours`**| $[1.0, 8.8]$ | $[1.0, 8.8]$ | **$[0.0, 16.0]$** | **Engineering safeguard.** Extends beyond sample max ($8.8\text{h}$) to accept high user inputs without server crashes, while capping at 16h as an operational boundary. |
| **`Daily_Unlocks`** | $[62, 273]$ | $[62, 273]$ | **$[0, 500]$** | **Engineering safeguard.** Prevents unconstrained integers while accepting user inputs moderately above the observed sample max ($273$). |
| **`Study_Hours`** | $[0.3, 8.3]$ | $[0.3, 8.3]$ | **$[0.0, 16.0]$** | **Engineering safeguard.** Accommodates intensive study periods while capping input. |
| **`Physical_Activity_Hours`**| $[-0.4, 4.1]$ | $[0.0, 4.1]$ | **$[0.0, 8.0]$** | **Engineering safeguard.** Lower bound clipped at $0.0$; upper bound capped at 8.0h. |
| **`Sleep_Hours_Per_Night`** | $[3.6, 9.9]$ | $[3.6, 9.9]$ | **$[2.0, 16.0]$** | **Engineering safeguard.** Permits inputs outside training range while rejecting extreme values. |

> [!IMPORTANT]
> **Classification of Engineering Safeguards:**  
> Bounds such as 16 hours of screen time or 500 unlocks are **engineering safeguards** to ensure input sanitization and prevent server errors. They must not be described as empirically proven physiological or behavioral limits.

---

## 8. Research Conclusions & Methodological Framework

To maintain scientific integrity, the conclusions of this investigation are structured into four distinct categories:

### 8.1 What the Dataset Empirically Demonstrates
1. **Discretized Target:** `Mental_Health_Score` consists of exactly 59 distinct values between 3.6 and 9.4 at uniform 0.1 intervals.
2. **Empirical Tenths Concentration:** Values ending in `.0` constitute 27.84% of all records.
3. **High Bivariate Correlations:** Continuous lifestyle metrics exhibit strong correlations with the target score (screen time $r = -0.817$, sleep $r = +0.766$, study $r = +0.753$, unlocks $r = -0.791$).
4. **Dominant Categorical Factor:** Perceived stress level accounts for 61.7% of target variance ($\eta^2 = 0.617$).
5. **Structural Deterministic Couplings:** Exact categorical couplings exist between Age and Academic Level, between Platform and Purpose for major platforms, and between regional messaging apps and specific countries.

### 8.2 What the Evidence Suggests
1. **Directionality:** Statistical associations consistently suggest that higher scores reflect balanced, protective routines (higher sleep, regular study, moderate device use), while lower scores reflect behavioral strain (elevated screen exposure, high unlocks, elevated stress).
2. **Non-Deterministic Target Generation:** Because linear models ($R^2 \approx 0.73$) and non-linear ensembles ($R^2 \approx 0.878$) leave $\approx 12.2\%$ unexplained variance, the evidence suggests that `Mental_Health_Score` is not an exact deterministic mathematical function of the features.
3. **Programmatic Data Structuring:** The exact categorical pairings and absence of missing data suggest that the dataset was generated programmatically or subjected to rigid schema constraints.

### 8.3 What Remains Unknown
1. The original source, institutional author, collection methodology, and creation timeline remain undocumented.
2. Whether the data originated from human survey respondents or a programmatic generation algorithm is unverified.
3. The operational definition, psychometric validity, and scoring methodology of `Mental_Health_Score` remain unknown.
4. The exact cause of the $12.2\%$ unexplained variance cannot be isolated from this experiment alone.
5. Whether the tenths concentration at `.0` represents human digit preference, scoring aggregation, or generation artifacts is unknown.

### 8.4 What Cannot Currently Be Claimed
- MindScope **cannot claim clinical validity**, diagnostic capability, or screening efficacy for depression, anxiety, or medical pathology.
- MindScope **cannot claim causal intervention efficacy** (e.g., that altering screen time or sleep will cause an increase in real-world mental well-being).
- MindScope **cannot claim real-world population generalizability** beyond this specific benchmark dataset.
