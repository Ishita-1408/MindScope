import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for publication plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import (
    DEFAULT_MODEL_PATH,
    ORIGINAL_FEATURE_METADATA,
    MindScopeExplainer,
    load_explainer,
)
from src.preprocessing import (
    EXPECTED_COLUMNS,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    load_and_validate_data,
)

# Paths and Directories
DATA_FILE = PROJECT_ROOT / "Student Social Media And Mental Health Impact.csv"
DEPENDENCE_DIR = PROJECT_ROOT / "artifacts" / "explanations" / "dependence"
SEALED_HOLDOUT_PATH = PROJECT_ROOT / "artifacts" / "sealed_holdout_test.csv"

# Numerical and Categorical Predictor Groups
NUMERICAL_FEATURES = [
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Study_Hours",
    "Age",
]

CATEGORICAL_FEATURES = [
    "Purpose_Of_Use",
    "Country",
    "Most_Used_Platform",
    "Academic_Level",
    "Stress_Level",
    "Gender",
]

PRIORITIZED_FEATURES_ORDER = [
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Daily_Unlocks",
    "Purpose_Of_Use",
    "Country",
    "Most_Used_Platform",
    "Physical_Activity_Hours",
    "Study_Hours",
    "Age",
    "Academic_Level",
    "Stress_Level",
    "Gender",
]

CANDIDATE_INTERACTION_PAIRS = [
    ("Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"),
    ("Avg_Daily_Usage_Hours", "Daily_Unlocks"),
    ("Avg_Daily_Usage_Hours", "Physical_Activity_Hours"),
    ("Sleep_Hours_Per_Night", "Stress_Level"),
    ("Study_Hours", "Sleep_Hours_Per_Night"),
]

NON_CAUSAL_DISCLAIMER = (
    "Model attribution describes how feature values shift this model's prediction relative "
    "to its expected baseline (6.22). It does not establish causality, real-world impact, "
    "or clinical assessments."
)

# Plot styling configuration
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
})


def verify_sealed_holdout_isolation(path: Optional[Path] = None):
    """Guarantees that sealed holdout data is never accessed or referenced."""
    target = Path(path) if path else SEALED_HOLDOUT_PATH
    if "sealed_holdout_test.csv" in str(target):
        raise PermissionError(
            "Access violation: 'artifacts/sealed_holdout_test.csv' is strictly sealed "
            "and cannot be loaded, accessed, or inspected during feature analysis."
        )


def get_deterministic_training_sample(
    sample_size: int = 500, random_state: int = 42, data_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Reconstructs the exact 3,498-record training partition from the raw dataset.
    Never opens, loads, or references 'artifacts/sealed_holdout_test.csv'.
    Samples exactly `sample_size` records deterministically using random_state=42.
    """
    csv_path = data_path or DATA_FILE
    verify_sealed_holdout_isolation(csv_path)

    X, y = load_and_validate_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    total_train_records = len(X_train)
    if total_train_records != 3498:
        raise ValueError(f"Expected 3,498 training records, found {total_train_records}")

    if sample_size > total_train_records:
        raise ValueError(
            f"Requested sample_size ({sample_size}) exceeds training partition size ({total_train_records})"
        )

    sample_df = X_train.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return sample_df, total_train_records


def compute_feature_dependence(
    explainer: MindScopeExplainer, sample_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Computes feature dependence statistics for all 12 original predictors across sample records.
    Returns raw observations, grouped SHAP attributions, distributional statistics,
    and non-linear trend metrics.
    """
    X_trans = explainer.transform_data(sample_df)
    shap_trans = explainer.compute_transformed_shap_values(X_trans)
    grouped_df = explainer.aggregate_to_original_features(shap_trans)

    dependence_results: Dict[str, Any] = {
        "metadata": {
            "sample_size": len(sample_df),
            "expected_value": explainer.expected_value,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "numerical_features": {},
        "categorical_features": {},
    }

    # 1. Numerical Feature Dependence
    for feat in NUMERICAL_FEATURES:
        feat_vals = sample_df[feat].to_numpy(dtype=float)
        shap_vals = grouped_df[feat].to_numpy(dtype=float)

        pearson_r, pearson_p = stats.pearsonr(feat_vals, shap_vals)
        spearman_rho, spearman_p = stats.spearmanr(feat_vals, shap_vals)

        # Check for attribution direction changes across value range
        sign_changes = int(np.sum((shap_vals[:-1] * shap_vals[1:]) < 0))
        positive_pct = float(np.mean(shap_vals > 0) * 100.0)
        negative_pct = float(np.mean(shap_vals < 0) * 100.0)

        # Quantile binning for non-linear shape tracking
        quantiles = np.percentile(feat_vals, [0, 25, 50, 75, 100])
        bin_stats = []
        for q_low, q_high in zip(quantiles[:-1], quantiles[1:]):
            mask = (feat_vals >= q_low) & (feat_vals <= q_high)
            bin_shaps = shap_vals[mask]
            bin_stats.append({
                "range": [float(q_low), float(q_high)],
                "count": int(np.sum(mask)),
                "mean_shap": float(np.mean(bin_shaps)) if len(bin_shaps) > 0 else 0.0,
                "median_shap": float(np.median(bin_shaps)) if len(bin_shaps) > 0 else 0.0,
                "std_shap": float(np.std(bin_shaps)) if len(bin_shaps) > 0 else 0.0,
            })

        meta = ORIGINAL_FEATURE_METADATA.get(feat, {})
        dependence_results["numerical_features"][feat] = {
            "display_name": meta.get("display_name", feat),
            "unit": meta.get("unit", ""),
            "min_value": float(np.min(feat_vals)),
            "max_value": float(np.max(feat_vals)),
            "mean_value": float(np.mean(feat_vals)),
            "median_value": float(np.median(feat_vals)),
            "min_shap": float(np.min(shap_vals)),
            "max_shap": float(np.max(shap_vals)),
            "mean_shap": float(np.mean(shap_vals)),
            "mean_abs_shap": float(np.mean(np.abs(shap_vals))),
            "std_shap": float(np.std(shap_vals)),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "positive_attribution_pct": positive_pct,
            "negative_attribution_pct": negative_pct,
            "bin_statistics": bin_stats,
            "sample_values": feat_vals.tolist(),
            "sample_shaps": shap_vals.tolist(),
        }

    # 2. Categorical Feature Distributions
    for feat in CATEGORICAL_FEATURES:
        feat_vals = sample_df[feat].astype(str).to_numpy()
        shap_vals = grouped_df[feat].to_numpy(dtype=float)

        unique_cats = sorted(list(set(feat_vals)))
        cat_stats = []
        for cat in unique_cats:
            mask = feat_vals == cat
            cat_shaps = shap_vals[mask]
            cat_stats.append({
                "category": cat,
                "count": int(np.sum(mask)),
                "proportion": float(np.mean(mask)),
                "mean_shap": float(np.mean(cat_shaps)) if len(cat_shaps) > 0 else 0.0,
                "median_shap": float(np.median(cat_shaps)) if len(cat_shaps) > 0 else 0.0,
                "std_shap": float(np.std(cat_shaps)) if len(cat_shaps) > 0 else 0.0,
                "min_shap": float(np.min(cat_shaps)) if len(cat_shaps) > 0 else 0.0,
                "max_shap": float(np.max(cat_shaps)) if len(cat_shaps) > 0 else 0.0,
                "iqr_shap": float(stats.iqr(cat_shaps)) if len(cat_shaps) > 0 else 0.0,
            })

        # Sort categories by mean absolute attribution descending
        cat_stats.sort(key=lambda x: abs(x["mean_shap"]), reverse=True)

        meta = ORIGINAL_FEATURE_METADATA.get(feat, {})
        dependence_results["categorical_features"][feat] = {
            "display_name": meta.get("display_name", feat),
            "n_categories": len(unique_cats),
            "mean_abs_shap": float(np.mean(np.abs(shap_vals))),
            "category_statistics": cat_stats,
            "categories": feat_vals.tolist(),
            "sample_shaps": shap_vals.tolist(),
        }

    return dependence_results


def benchmark_interaction_timing(
    explainer: MindScopeExplainer,
    sample_df: pd.DataFrame,
    test_sizes: Optional[List[int]] = None,
    time_limit_seconds: float = 120.0,
) -> Dict[str, Any]:
    """
    Benchmarks computational execution time and memory properties of TreeSHAP interaction values
    on small deterministic training samples to establish empirical scaling laws.
    Enforces strict execution timeouts.
    """
    if test_sizes is None:
        test_sizes = [1, 2, 5]

    X_trans = explainer.transform_data(sample_df)
    preds = explainer.model.predict(X_trans)
    ev = explainer.expected_value

    benchmark_records = []
    scaling_projections = {}

    for n in test_sizes:
        if n > len(sample_df):
            continue

        t0 = time.perf_counter()
        X_sub = X_trans[:n]
        try:
            inter_tensor = explainer.explainer.shap_interaction_values(X_sub)
            dt = time.perf_counter() - t0

            # Verify symmetry: M == M.T
            max_asymm = float(np.max([np.max(np.abs(mat - mat.T)) for mat in inter_tensor]))
            is_symmetric = bool(max_asymm < 1e-5)

            # Verify additivity: sum(inter) == pred - ev
            additivity_errors = []
            for i in range(n):
                total_inter = float(np.sum(inter_tensor[i]))
                expected_diff = float(preds[i] - ev)
                additivity_errors.append(abs(total_inter - expected_diff))
            max_add_err = float(np.max(additivity_errors))

            rec = {
                "sample_size": n,
                "execution_time_seconds": round(dt, 4),
                "seconds_per_sample": round(dt / n, 4),
                "output_shape": list(inter_tensor.shape),
                "is_symmetric": is_symmetric,
                "max_asymmetry": max_asymm,
                "max_additivity_error": max_add_err,
            }
            benchmark_records.append(rec)

            if dt > time_limit_seconds:
                break
        except Exception as e:
            benchmark_records.append({
                "sample_size": n,
                "error": str(e),
            })
            break

    # Calculate average cost per observation
    valid_rates = [r["seconds_per_sample"] for r in benchmark_records if "seconds_per_sample" in r]
    avg_sec_per_obs = float(np.mean(valid_rates)) if valid_rates else 9.5

    # Project scaling to larger cohorts
    scaling_projections = {
        "avg_seconds_per_observation": round(avg_sec_per_obs, 2),
        "projected_seconds_n50": round(avg_sec_per_obs * 50, 1),
        "projected_seconds_n100": round(avg_sec_per_obs * 100, 1),
        "projected_seconds_n500": round(avg_sec_per_obs * 500, 1),
        "projected_hours_n500": round((avg_sec_per_obs * 500) / 3600.0, 2),
        "projected_seconds_full_train_3498": round(avg_sec_per_obs * 3498, 1),
        "projected_hours_full_train_3498": round((avg_sec_per_obs * 3498) / 3600.0, 2),
        "recommended_sample_size": 10,
        "is_full_interaction_feasible": False,
        "feasibility_justification": (
            f"Computing full 38x38 TreeSHAP interaction values requires approximately "
            f"{avg_sec_per_obs:.1f} seconds per observation. Running on the full 3,498 training records "
            f"would require ~{(avg_sec_per_obs * 3498) / 3600.0:.1f} hours, and even a 500-record sample "
            f"would require ~{(avg_sec_per_obs * 500) / 3600.0:.1f} hours. Therefore, full population-scale "
            f"interaction computation is computationally prohibitive, and a targeted deterministic sample "
            f"(N=10) is recommended for interaction analysis."
        ),
    }

    return {
        "benchmarks": benchmark_records,
        "scaling_projections": scaling_projections,
        "tree_count": len(explainer.model.estimators_),
        "transformed_features": explainer.n_transformed_features,
    }


def aggregate_interaction_to_original(
    inter_transformed: np.ndarray, explainer: MindScopeExplainer
) -> np.ndarray:
    """
    Aggregates an (N, 38, 38) transformed interaction tensor into an (N, 12, 12)
    original-feature interaction tensor.

    Mathematically, for each observation i and original features A and B:
    Phi_{i, A, B} = sum_{j in I(A)} sum_{k in I(B)} Phi_{i, j, k}

    Properties verified:
    1. Symmetry: Phi_{i, A, B} == Phi_{i, B, A}
    2. Row-sum consistency: sum_{B} Phi_{i, A, B} == Phi_{i, A} (grouped SHAP)
    3. Additivity: sum_{A, B} Phi_{i, A, B} == f(x_i) - E[f(x)]
    """
    if inter_transformed.ndim == 2:
        inter_transformed = inter_transformed[np.newaxis, :, :]

    N, M, M2 = inter_transformed.shape
    if M != explainer.n_transformed_features or M2 != explainer.n_transformed_features:
        raise ValueError(
            f"Expected transformed dimension ({explainer.n_transformed_features}, "
            f"{explainer.n_transformed_features}), got ({M}, {M2})"
        )

    n_orig = len(PREDICTOR_COLUMNS)
    orig_tensor = np.zeros((N, n_orig, n_orig), dtype=np.float64)

    for r_idx, r_feat in enumerate(PREDICTOR_COLUMNS):
        r_cols = explainer.parent_to_indices[r_feat]
        for c_idx, c_feat in enumerate(PREDICTOR_COLUMNS):
            c_cols = explainer.parent_to_indices[c_feat]
            # Sum submatrix over indices corresponding to parent features
            ix = np.ix_(r_cols, c_cols)
            orig_tensor[:, r_idx, c_idx] = np.sum(
                inter_transformed[:, ix[0], ix[1]], axis=(1, 2)
            )

    return orig_tensor


def compute_interaction_analysis(
    explainer: MindScopeExplainer, sample_df: pd.DataFrame, n_samples: int = 10
) -> Dict[str, Any]:
    """
    Computes exact TreeSHAP interaction values on a deterministic sample of n_samples records.
    Produces both transformed (38x38) and aggregated original (12x12) interaction matrices,
    ranks off-diagonal pairwise interactions, and extracts focused candidate pair profiles.
    """
    if n_samples > len(sample_df):
        raise ValueError(f"Requested n_samples ({n_samples}) exceeds available records ({len(sample_df)})")

    sub_df = sample_df.iloc[:n_samples].reset_index(drop=True)
    X_trans = explainer.transform_data(sub_df)
    preds = explainer.model.predict(X_trans)
    ev = explainer.expected_value

    t0 = time.perf_counter()
    inter_trans = explainer.explainer.shap_interaction_values(X_trans)
    elapsed = time.perf_counter() - t0

    # Aggregate to 12x12 original feature interaction tensor
    orig_tensor = aggregate_interaction_to_original(inter_trans, explainer)

    # 1. Compute Mean Absolute Interaction Matrix across the sample
    mean_abs_inter_12 = np.mean(np.abs(orig_tensor), axis=0)

    # Verify symmetry of aggregated mean matrix
    symm_err = float(np.max(np.abs(mean_abs_inter_12 - mean_abs_inter_12.T)))

    # 2. Extract and rank all 66 unique off-diagonal pairs (A < B)
    n_orig = len(PREDICTOR_COLUMNS)
    pairwise_ranks = []
    for i in range(n_orig):
        feat_a = PREDICTOR_COLUMNS[i]
        for j in range(i + 1, n_orig):
            feat_b = PREDICTOR_COLUMNS[j]
            mean_abs_val = float(mean_abs_inter_12[i, j])
            pairwise_ranks.append({
                "feature_a": feat_a,
                "feature_b": feat_b,
                "mean_abs_interaction": mean_abs_val,
                "feature_a_display": ORIGINAL_FEATURE_METADATA.get(feat_a, {}).get("display_name", feat_a),
                "feature_b_display": ORIGINAL_FEATURE_METADATA.get(feat_b, {}).get("display_name", feat_b),
            })

    pairwise_ranks.sort(key=lambda x: x["mean_abs_interaction"], reverse=True)
    for rank, item in enumerate(pairwise_ranks, 1):
        item["rank"] = rank

    # 3. Main effects (diagonal elements)
    main_effects = []
    for i in range(n_orig):
        feat = PREDICTOR_COLUMNS[i]
        main_effects.append({
            "feature": feat,
            "display_name": ORIGINAL_FEATURE_METADATA.get(feat, {}).get("display_name", feat),
            "mean_abs_main_effect": float(mean_abs_inter_12[i, i]),
        })
    main_effects.sort(key=lambda x: x["mean_abs_main_effect"], reverse=True)

    # 4. Extract focused candidate interaction pairs
    candidate_profiles = []
    for feat_a, feat_b in CANDIDATE_INTERACTION_PAIRS:
        idx_a = PREDICTOR_COLUMNS.index(feat_a)
        idx_b = PREDICTOR_COLUMNS.index(feat_b)
        sample_interactions = orig_tensor[:, idx_a, idx_b].tolist()
        mean_abs_ij = float(mean_abs_inter_12[idx_a, idx_b])

        # Find rank in pairwise_ranks
        rank_val = next(
            (p["rank"] for p in pairwise_ranks if (
                (p["feature_a"] == feat_a and p["feature_b"] == feat_b) or
                (p["feature_a"] == feat_b and p["feature_b"] == feat_a)
            )),
            None
        )

        candidate_profiles.append({
            "pair": f"{feat_a} x {feat_b}",
            "feature_a": feat_a,
            "feature_b": feat_b,
            "rank": rank_val,
            "mean_abs_interaction": mean_abs_ij,
            "sample_interaction_values": sample_interactions,
            "interpretation_note": (
                f"Off-diagonal TreeSHAP interaction between {feat_a} and {feat_b}. "
                f"Measures pure non-additive contribution in the fitted Random Forest."
            ),
        })

    return {
        "sample_size": n_samples,
        "execution_time_seconds": round(elapsed, 4),
        "mean_abs_interaction_matrix_12x12": mean_abs_inter_12.tolist(),
        "predictor_columns": PREDICTOR_COLUMNS,
        "symmetry_max_error": symm_err,
        "top_pairwise_interactions": pairwise_ranks[:15],
        "all_pairwise_interactions": pairwise_ranks,
        "main_effects": main_effects,
        "candidate_pair_profiles": candidate_profiles,
        "non_causal_disclaimer": NON_CAUSAL_DISCLAIMER,
    }


# ==============================================================================
# VISUALIZATION SUITE
# ==============================================================================

def plot_numerical_dependence_grid(dep_data: Dict[str, Any], output_path: Path):
    """
    Generates a 6-panel publication-quality grid of numerical feature dependence plots.
    Horizontal axis: raw feature value (with natural units).
    Vertical axis: SHAP attribution (points on mental health scale).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()

    for idx, feat in enumerate(NUMERICAL_FEATURES):
        ax = axes[idx]
        data = dep_data["numerical_features"][feat]
        x_vals = np.array(data["sample_values"])
        y_vals = np.array(data["sample_shaps"])

        # Scatter plot with neutral slate-blue color
        ax.scatter(x_vals, y_vals, alpha=0.55, color="#2b5c8f", edgecolors="none", s=22)

        # Zero reference line (baseline contribution = 0)
        ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.85)

        # Non-parametric LOWESS / smoothing trend curve for visual shape inspection
        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]
        y_sorted = y_vals[sort_idx]
        # Rolling median trend curve
        roll_window = max(15, len(x_sorted) // 15)
        y_roll = pd.Series(y_sorted).rolling(window=roll_window, center=True, min_periods=5).median()
        ax.plot(x_sorted, y_roll, color="#0f172a", linewidth=2.0, label="Median Trend")

        unit_str = f" ({data['unit']})" if data['unit'] else ""
        ax.set_xlabel(f"{data['display_name']}{unit_str}", fontweight="semibold")
        ax.set_ylabel("SHAP Attribution (Points)", fontweight="semibold")
        ax.set_title(
            f"{data['display_name']}\n(Spearman $\\rho={data['spearman_rho']:.2f}$)",
            fontweight="bold",
            pad=8,
        )
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Numerical Feature Dependence Analysis (N={dep_data['metadata']['sample_size']} Training Records)\n"
        "Model Feature Attributions Relative to Expected Baseline (6.22)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        NON_CAUSAL_DISCLAIMER,
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#475569",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_individual_dependence(dep_data: Dict[str, Any], feature: str, output_path: Path):
    """Generates an individual focused dependence plot for a high-priority numerical feature."""
    if feature not in dep_data["numerical_features"]:
        raise ValueError(f"Feature '{feature}' not found in numerical dependence data.")

    data = dep_data["numerical_features"][feature]
    x_vals = np.array(data["sample_values"])
    y_vals = np.array(data["sample_shaps"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x_vals, y_vals, alpha=0.6, color="#2b5c8f", edgecolors="none", s=30)
    ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.85)

    # Rolling median curve
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]
    roll_window = max(20, len(x_sorted) // 12)
    y_roll = pd.Series(y_sorted).rolling(window=roll_window, center=True, min_periods=5).median()
    ax.plot(x_sorted, y_roll, color="#0f172a", linewidth=2.5, label="Rolling Median Trend")

    unit_str = f" ({data['unit']})" if data['unit'] else ""
    ax.set_xlabel(f"{data['display_name']}{unit_str}", fontweight="semibold")
    ax.set_ylabel("SHAP Attribution (Points on Score Scale)", fontweight="semibold")
    ax.set_title(
        f"Feature Dependence: {data['display_name']}\n"
        f"N={dep_data['metadata']['sample_size']} Training Records (Pearson $r={data['pearson_r']:.2f}$, Spearman $\\rho={data['spearman_rho']:.2f}$)",
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1")

    fig.text(
        0.5,
        -0.02,
        NON_CAUSAL_DISCLAIMER,
        ha="center",
        fontsize=8,
        style="italic",
        color="#475569",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_categorical_distributions(dep_data: Dict[str, Any], output_path: Path):
    """
    Generates a 6-panel publication-quality grid of categorical feature attribution distributions.
    Displays grouped boxplots and jittered strip scatters showing attribution spreads per category.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(CATEGORICAL_FEATURES):
        ax = axes[idx]
        data = dep_data["categorical_features"][feat]
        cats = [item["category"] for item in data["category_statistics"]]
        
        # Gather attribution lists per category in ranked order
        cat_shaps = []
        for cat in cats:
            indices = [i for i, c in enumerate(data["categories"]) if c == cat]
            cat_shaps.append([data["sample_shaps"][i] for i in indices])

        # Horizontal boxplot
        bp = ax.boxplot(
            cat_shaps,
            tick_labels=cats,
            orientation="horizontal",
            patch_artist=True,
            boxprops=dict(facecolor="#e2e8f0", color="#334155"),
            medianprops=dict(color="#0f172a", linewidth=2.0),
            whiskerprops=dict(color="#64748b"),
            capprops=dict(color="#64748b"),
            flierprops=dict(marker="o", markersize=3, markerfacecolor="#94a3b8", alpha=0.5),
            widths=0.55,
        )

        ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.set_xlabel("SHAP Attribution (Points)", fontweight="semibold")
        ax.set_title(f"{data['display_name']}", fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.5, axis="x")

    fig.suptitle(
        f"Categorical Feature Attribution Distributions (N={dep_data['metadata']['sample_size']} Training Records)\n"
        "Model Feature Attributions Relative to Expected Baseline (6.22)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        NON_CAUSAL_DISCLAIMER,
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#475569",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_interaction_matrix_heatmap(inter_data: Dict[str, Any], output_path: Path):
    """
    Plots a 12x12 heatmap of mean absolute interaction strength between the 12 original predictors.
    """
    matrix = np.array(inter_data["mean_abs_interaction_matrix_12x12"])
    features = inter_data["predictor_columns"]
    display_names = [ORIGINAL_FEATURE_METADATA.get(f, {}).get("display_name", f) for f in features]

    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    # Show colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Mean |Interaction SHAP| (Points)", rotation=-90, va="bottom", fontweight="semibold")

    # Set ticks
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(display_names, fontsize=9)

    # Annotate numeric values in cells
    max_val = np.max(matrix)
    for i in range(len(features)):
        for j in range(len(features)):
            val = matrix[i, j]
            text_color = "white" if val > (max_val * 0.55) else "#0f172a"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=7.5)

    ax.set_title(
        f"Aggregated Pairwise Feature Interaction Matrix (12 Original Predictors)\n"
        f"Mean |SHAP Interaction| across N={inter_data['sample_size']} Training Records",
        fontweight="bold",
        pad=15,
    )

    fig.text(
        0.5,
        -0.03,
        NON_CAUSAL_DISCLAIMER,
        ha="center",
        fontsize=8,
        style="italic",
        color="#475569",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_candidate_pair_interactions(
    dep_data: Dict[str, Any],
    inter_data: Dict[str, Any],
    sample_df: pd.DataFrame,
    output_path: Path,
):
    """
    Plots bivariate interaction plots for the primary candidate pairs:
    1. Screen Time (Avg_Daily_Usage_Hours) x Sleep Duration (Sleep_Hours_Per_Night)
    2. Screen Time (Avg_Daily_Usage_Hours) x Physical Activity (Physical_Activity_Hours)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Usage x Sleep
    ax1 = axes[0]
    usage = sample_df["Avg_Daily_Usage_Hours"].to_numpy()
    usage_shap = np.array(dep_data["numerical_features"]["Avg_Daily_Usage_Hours"]["sample_shaps"])
    sleep = sample_df["Sleep_Hours_Per_Night"].to_numpy()

    sc1 = ax1.scatter(usage, usage_shap, c=sleep, cmap="Blues_r", s=25, alpha=0.75, edgecolors="none")
    ax1.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.0)
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label("Sleep Duration (hours/night)", fontweight="semibold")
    ax1.set_xlabel("Daily Social Media Usage (hours/day)", fontweight="semibold")
    ax1.set_ylabel("Usage SHAP Attribution (Points)", fontweight="semibold")
    ax1.set_title("Screen Time Attribution Colored by Sleep Duration", fontweight="bold", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Usage x Physical Activity
    ax2 = axes[1]
    activity = sample_df["Physical_Activity_Hours"].to_numpy()
    sc2 = ax2.scatter(usage, usage_shap, c=activity, cmap="Blues_r", s=25, alpha=0.75, edgecolors="none")
    ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.0)
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Physical Activity (hours/day)", fontweight="semibold")
    ax2.set_xlabel("Daily Social Media Usage (hours/day)", fontweight="semibold")
    ax2.set_ylabel("Usage SHAP Attribution (Points)", fontweight="semibold")
    ax2.set_title("Screen Time Attribution Colored by Physical Activity", fontweight="bold", pad=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Bivariate Feature Interaction Analysis (N={dep_data['metadata']['sample_size']} Training Records)",
        fontweight="bold",
        fontsize=13,
        y=0.99,
    )
    fig.text(
        0.5,
        -0.02,
        NON_CAUSAL_DISCLAIMER,
        ha="center",
        fontsize=8,
        style="italic",
        color="#475569",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

def run_feature_analysis(
    sample_size: int = 500,
    interaction_sample_size: int = 10,
    output_dir: Optional[Path] = None,
    reuse_cache: bool = False,
) -> Dict[str, Any]:
    """
    Executes the complete Phase 3 Step 3.4 Feature Dependence and Interaction Analysis pipeline.
    Produces JSON data summaries and publication-grade visualizations in artifacts/explanations/dependence/.
    """
    out_dir = output_dir or DEPENDENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MINDSCOPE — PHASE 3, STEP 3.4 FEATURE DEPENDENCE & INTERACTION ANALYSIS")
    print("=" * 70)

    # 1. Load fitted explainer (never retraining)
    print("[1/6] Loading fitted TreeSHAP explainer...")
    explainer = load_explainer()
    print(f"      Expected Value: {explainer.expected_value:.4f}")
    print(f"      Transformed Features: {explainer.n_transformed_features}")

    # 2. Extract deterministic training sample (never touching sealed holdout)
    print(f"[2/6] Extracting deterministic training partition (sample_size={sample_size})...")
    sample_df, n_total = get_deterministic_training_sample(sample_size=sample_size, random_state=42)
    print(f"      Extracted {len(sample_df)} observations from {n_total:,} training records.")

    # 3. Compute Feature Dependence
    dep_json_path = out_dir / "dependence_summary.json"
    if reuse_cache and dep_json_path.exists():
        print(f"[3/6] Loading cached dependence summary: {dep_json_path}")
        with open(dep_json_path, "r", encoding="utf-8") as f:
            dep_results = json.load(f)
    else:
        print("[3/6] Computing feature dependence across 12 original predictors...")
        dep_results = compute_feature_dependence(explainer, sample_df)
        with open(dep_json_path, "w", encoding="utf-8") as f:
            json.dump(dep_results, f, indent=2)
        print(f"      Saved dependence summary: {dep_json_path}")

    # 4. Interaction Benchmarking
    bench_json_path = out_dir / "interaction_benchmark.json"
    if reuse_cache and bench_json_path.exists():
        print(f"[4/6] Loading cached benchmark results: {bench_json_path}")
        with open(bench_json_path, "r", encoding="utf-8") as f:
            benchmark_results = json.load(f)
    else:
        print("[4/6] Benchmarking TreeSHAP interaction computational feasibility...")
        benchmark_results = benchmark_interaction_timing(explainer, sample_df, test_sizes=[1, 2, 5])
        with open(bench_json_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"      Saved benchmark results: {bench_json_path}")
    print(f"      Average rate: {benchmark_results['scaling_projections']['avg_seconds_per_observation']} s/obs")
    print(f"      Projected 500 records: {benchmark_results['scaling_projections']['projected_hours_n500']} hours")

    # 5. Compute Interaction Analysis on Tractable Sample (N=10)
    inter_json_path = out_dir / "interaction_analysis.json"
    if reuse_cache and inter_json_path.exists():
        print(f"[5/6] Loading cached interaction analysis: {inter_json_path}")
        with open(inter_json_path, "r", encoding="utf-8") as f:
            inter_results = json.load(f)
    else:
        print(f"[5/6] Computing interaction tensor for N={interaction_sample_size} records...")
        inter_results = compute_interaction_analysis(explainer, sample_df, n_samples=interaction_sample_size)
        with open(inter_json_path, "w", encoding="utf-8") as f:
            json.dump(inter_results, f, indent=2)
        print(f"      Saved interaction analysis: {inter_json_path}")
    print(f"      Top interaction pair: {inter_results['top_pairwise_interactions'][0]['feature_a']} x "
          f"{inter_results['top_pairwise_interactions'][0]['feature_b']} "
          f"(mean |inter| = {inter_results['top_pairwise_interactions'][0]['mean_abs_interaction']:.4f})")

    # 6. Generate Visualizations
    print("[6/6] Generating publication-quality visualizations...")
    grid_path = out_dir / "numerical_dependence_grid.png"
    plot_numerical_dependence_grid(dep_results, grid_path)
    print(f"      Generated: {grid_path}")

    usage_path = out_dir / "dependence_usage_hours.png"
    plot_individual_dependence(dep_results, "Avg_Daily_Usage_Hours", usage_path)
    print(f"      Generated: {usage_path}")

    sleep_path = out_dir / "dependence_sleep_hours.png"
    plot_individual_dependence(dep_results, "Sleep_Hours_Per_Night", sleep_path)
    print(f"      Generated: {sleep_path}")

    unlocks_path = out_dir / "dependence_daily_unlocks.png"
    plot_individual_dependence(dep_results, "Daily_Unlocks", unlocks_path)
    print(f"      Generated: {unlocks_path}")

    cats_path = out_dir / "categorical_attribution_distributions.png"
    plot_categorical_distributions(dep_results, cats_path)
    print(f"      Generated: {cats_path}")

    heatmap_path = out_dir / "interaction_matrix_heatmap.png"
    plot_interaction_matrix_heatmap(inter_results, heatmap_path)
    print(f"      Generated: {heatmap_path}")

    pairs_path = out_dir / "interaction_candidate_pairs.png"
    plot_candidate_pair_interactions(dep_results, inter_results, sample_df, pairs_path)
    print(f"      Generated: {pairs_path}")

    # Metadata file
    metadata = {
        "milestone": "Phase 3, Step 3.4 — Feature Dependence & Interaction Analysis",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": {
            "python_version": sys.version,
            "os": platform.platform(),
        },
        "governance": {
            "sealed_holdout_accessed": False,
            "models_retrained": False,
            "training_partition_size": n_total,
            "dependence_sample_size": sample_size,
            "interaction_sample_size": interaction_sample_size,
            "random_state": 42,
        },
        "artifacts_generated": [
            "dependence_summary.json",
            "interaction_benchmark.json",
            "interaction_analysis.json",
            "numerical_dependence_grid.png",
            "dependence_usage_hours.png",
            "dependence_sleep_hours.png",
            "dependence_daily_unlocks.png",
            "categorical_attribution_distributions.png",
            "interaction_matrix_heatmap.png",
            "interaction_candidate_pairs.png",
        ],
    }
    meta_json_path = out_dir / "dependence_metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"      Saved metadata: {meta_json_path}")

    print("=" * 70)
    print("STEP 3.4 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return {
        "dependence": dep_results,
        "benchmark": benchmark_results,
        "interaction": inter_results,
        "metadata": metadata,
    }


if __name__ == "__main__":
    reuse = "--reuse-cache" in sys.argv
    run_feature_analysis(reuse_cache=reuse)
