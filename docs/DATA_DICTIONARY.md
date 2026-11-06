# MindScope — Student Mental Health Analytics & Predictive Intelligence Platform
## Comprehensive Data Dictionary & Exploratory Analysis Specification

- **Document Version:** 1.0.0
- **Status:** Approved Data Specification
- **Dataset File:** `Student Social Media And Mental Health Impact.csv`
- **Volume:** 5,000 records, 13 features (398 KB)
- **Scope:** Feature Schemas, Target Variable Investigation, Anomaly Log, and Transformation Logic

---

## 1. Dataset Overview & Provenance

### 1.1 Provenance Statement
The primary dataset used by MindScope is stored in `Student Social Media And Mental Health Impact.csv`. 

> [!WARNING]
> **Unverified Provenance & Synthetic Data Hypothesis:**  
> The original data source, institutional origin, survey instrument, sampling methodology, and collection timeline are **undocumented and unverified** in the codebase repository.  
> 
> While statistical indicators—such as unusually high linear correlations ($|r| > 0.75$) with the target variable and the complete absence of natural real-world missingness—suggest possible synthetic data generation, this remains an **unconfirmed hypothesis** rather than an established fact. The dataset is retained as an empirical research baseline without asserting that it represents independently verified observational survey data.

### 1.2 Structural Summary
- **Total Records:** 5,000 rows
- **Total Attributes:** 13 columns (1 target variable, 12 predictor features)
- **Missing Value Count:** 0 (100% complete across all columns)
- **Duplicate Records:** 2 duplicate rows identified (`df.duplicated().sum() == 2`). Dropping duplicates leaves **4,998 unique records**.
- **Data Types:** 5 continuous numerical floats, 2 discrete numerical integers, 6 categorical object strings.

---

## 2. Complete Feature Dictionary

| # | Column Identifier | Data Type | Measurement Unit | Empirical Range (Dataset) | System Boundary (API) | Nulls | Description & Analytical Role |
| :-: | :--- | :--- | :--- | :---: | :---: | :-: | :--- |
| **0** | `Age` | `int64` | Years | $[18, 24]$ | $[10, 100]$ | 0 | Age of respondent. Mean: 20.82, Std: 1.74. API validates realistic educational age range $[10, 100]$ while expecting student domain $[18, 35]$. |
| **1** | `Gender` | `object` (`str`) | Category | `{'Male', 'Female'}` | `{'Male', 'Female'}` | 0 | Binary gender or biological sex indicator. Distribution: Male (2,635; 52.7%), Female (2,365; 47.3%). |
| **2** | `Country` | `object` (`str`) | Category | 111 unique values | Any valid string | 0 | Geographic residence. High cardinality feature; preprocessed into Top 10 countries + `'Other'`. |
| **3** | `Academic_Level` | `object` (`str`) | Category | 3 tiers | `{'High School', 'Undergraduate', 'Graduate'}` | 0 | Educational level. Undergraduate (3,632; 72.6%), Graduate (918; 18.4%), High School (450; 9.0%). |
| **4** | `Most_Used_Platform` | `object` (`str`) | Category | 12 platforms | 12 supported platforms | 0 | Primary social media application utilized by the respondent. |
| **5** | `Purpose_Of_Use` | `object` (`str`) | Category | 4 categories | `{'Entertainment', 'Education', 'Networking', 'News'}` | 0 | Self-reported primary motivation for digital platform engagement. |
| **6** | `Avg_Daily_Usage_Hours` | `float64` | Hours / Day | $[1.0, 8.8]$ | $[0.0, 24.0]$ | 0 | Daily active screen time dedicated to social media. Mean: 5.08, Std: 1.65. |
| **7** | `Daily_Unlocks` | `int64` | Unlocks / Day | $[62, 273]$ | $[0, \infty)$ | 0 | Daily frequency of mobile device unlocks. Mean: 171.45, Std: 42.86. |
| **8** | `Study_Hours` | `float64` | Hours / Day | $[0.3, 8.3]$ | $[0.0, 24.0]$ | 0 | Daily time committed to academic study outside class. Mean: 3.01, Std: 1.64. Positively skewed ($+0.436$). |
| **9** | `Physical_Activity_Hours`| `float64` | Hours / Day | $[-0.4, 4.1]$ | $[0.0, 24.0]$ | 0 | Daily physical exercise. Contains 10 negative entry artifacts requiring clipping at 0.0. Mean: 1.75, Std: 0.67. |
| **10**| `Sleep_Hours_Per_Night` | `float64` | Hours / Night | $[3.6, 9.9]$ | $[0.0, 24.0]$ | 0 | Average nightly sleep duration. Mean: 6.63, Std: 1.22. |
| **11**| `Stress_Level` | `object` (`str`) | Ordinal Tier | 4 tiers | `{'Low', 'Medium', 'High', 'Very High'}` | 0 | Perceived stress tier: Very High (1,621; 32.4%), High (1,440; 28.8%), Medium (1,295; 25.9%), Low (644; 12.9%). |
| **12**| `Mental_Health_Score` | `float64` | Score Unit | $[3.6, 9.4]$ | $[0.0, 10.0]$ | 0 | **Target Variable.** Continuous mental health score. Mean: 6.23, Std: 1.28. Non-clinical metric. |

---

## 3. Target Variable Deep-Dive: `Mental_Health_Score`

### 3.1 Empirical Properties
- **Identifier:** `Mental_Health_Score`
- **Role:** Supervised Continuous Regression Target
- **Parametric Summary:**
  - Count: 5,000 (4,998 unique)
  - Mean ($\mu$): $6.2310$
  - Standard Deviation ($\sigma$): $1.2787$
  - Minimum: $3.6000$
  - 25th Percentile ($Q_1$): $5.1000$
  - Median (50%): $6.1000$
  - 75th Percentile ($Q_3$): $7.1000$
  - Maximum: $9.4000$
  - Skewness: $+0.2071$ (approximately symmetric, normal distribution without tail truncation)

### 3.2 Target Investigation Requirement
The exact mathematical formula, diagnostic instrument, or survey composite used to construct `Mental_Health_Score` is **undocumented**. MindScope establishes an explicit empirical investigation requirement:
1. **Directionality Analysis:** Linear correlation analysis demonstrates that:
   - High sleep duration correlates positively ($r = +0.7664$).
   - High academic study hours correlate positively ($r = +0.7526$).
   - High physical activity correlates positively ($r = +0.5214$).
   - High daily social media usage correlates negatively ($r = -0.8165$).
   - High daily phone unlocks correlate negatively ($r = -0.7909$).
   - Elevated stress levels correspond to stepwise decreases in the median score.
   *Conclusion:* Empirically, higher scores unambiguously represent positive, healthy behavioral equilibria, while lower scores reflect behavioral strain.
2. **Non-Clinical Context:** The score must not be interpreted as a clinical diagnostic scale (such as PHQ-9 or GAD-7). Predetermined clinical cutoffs are avoided; interpretations in the interface are limited to relative population percentile rankings.

---

## 4. Categorical Distributions & Cardinality Mapping

### 4.1 Most Used Platform Distribution (12 Classes)
| Platform | Sample Count | Percentage |
| :--- | :---: | :---: |
| **Instagram** | 1,130 | 22.60% |
| **TikTok** | 918 | 18.36% |
| **Facebook** | 719 | 14.38% |
| **LinkedIn** | 523 | 10.46% |
| **YouTube** | 491 | 9.82% |
| **Twitter** | 462 | 9.24% |
| **Snapchat** | 420 | 8.40% |
| **WhatsApp** | 175 | 3.50% |
| **LINE** | 50 | 1.00% |
| **VKontakte** | 40 | 0.80% |
| **KakaoTalk** | 36 | 0.72% |
| **WeChat** | 36 | 0.72% |

### 4.2 High-Cardinality Country Grouping (Top 10 Mapping)
The raw dataset contains **111 unique countries**. The raw column already contains `'Other'` as its largest class (1,880 samples). The pipeline maps the top 10 most frequent classes and buckets all others into `'Other'`:

```
Top 10 Classes:
1. Other      : 1,880 (raw) ──> 3,231 (post-grouping, 64.65%)
2. India      :   389 ( 7.78%)
3. USA        :   355 ( 7.10%)
4. Canada     :   230 ( 4.60%)
5. Australia  :   198 ( 3.96%)
6. UK         :   185 ( 3.70%)
7. Germany    :   136 ( 2.72%)
8. Mexico     :    94 ( 1.88%)
9. Turkey     :    94 ( 1.88%)
10. France    :    87 ( 1.74%)
[Remaining 101 countries bucketed into 'Other': +1,351 samples]
```

### 4.3 Ordinal Stress Level Mapping
`Stress_Level` is encoded with a strict ordinal hierarchy:
$$\text{Low (0)} < \text{Medium (1)} < \text{High (2)} < \text{Very High (3)}$$
Distribution: `Very High` (1,621; 32.4%), `High` (1,440; 28.8%), `Medium` (1,295; 25.9%), `Low` (644; 12.9%).

---

## 5. Anomaly & Data Quality Log

1. **Duplicate Records:**  
   - Count: 2 duplicate records identified.
   - Action: Dropped prior to model training (`drop_duplicates()`), resulting in 4,998 unique rows.
2. **Negative Physical Activity Hours:**  
   - Count: 10 records with `Physical_Activity_Hours < 0.0` (minimum value $-0.40$).
   - Reason: Data entry or simulation artifact; physical activity hours cannot be negative.
   - Action: Clipped at lower bound $0.0$ (`clip(lower=0)`).
3. **Pre-Split Leakage Note:**  
   - Audit Finding: In the initial experimentation, country grouping and clipping were executed globally before the train-test split.
   - Engineering Correction: Refactored into pipeline-level transformers fitted exclusively on the training partition.
4. **Missing Values:**  
   - Count: Zero missing values in raw CSV.
   - Engineering Safeguard: Defensive `SimpleImputer` instances (median for numerical, most frequent for categorical) added to all pipeline branches to protect production inference against malformed inputs.

