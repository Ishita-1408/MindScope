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
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import DEFAULT_MODEL_PATH, MindScopeExplainer, load_explainer
from src.preprocessing import (
    EXPECTED_COLUMNS,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    load_and_validate_data,
)

# Directory configurations
DATA_FILE = PROJECT_ROOT / "Student Social Media And Mental Health Impact.csv"
EXPLANATIONS_DIR = PROJECT_ROOT / "artifacts" / "explanations"
GLOBAL_IMPORTANCE_JSON = EXPLANATIONS_DIR / "global_importance.json"
TRANSFORMED_IMPORTANCE_JSON = EXPLANATIONS_DIR / "transformed_importance.json"
METADATA_JSON = EXPLANATIONS_DIR / "global_analysis_metadata.json"
SUMMARY_BAR_PNG = EXPLANATIONS_DIR / "global_shap_summary.png"
GLOBAL_BEESWARM_PNG = EXPLANATIONS_DIR / "global_shap_beeswarm.png"
TRANSFORMED_BEESWARM_PNG = EXPLANATIONS_DIR / "transformed_shap_beeswarm.png"
DISTRIBUTION_PNG = EXPLANATIONS_DIR / "global_shap_distribution.png"

# Plot styling configuration
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
})


def get_deterministic_training_sample(
    sample_size: int = 500, random_state: int = 42
) -> Tuple[pd.DataFrame, int]:
    """
    Reconstructs the exact 3,498-record training partition from the raw dataset.
    Never opens, loads, or references 'artifacts/sealed_holdout_test.csv'.
    Samples exactly `sample_size` records deterministically using random_state=42.
    """
    X, y = load_and_validate_data(DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    total_train_records = len(X_train)
    if total_train_records != 3498:
        raise ValueError(f"Expected 3,498 training records, found {total_train_records}")

    sample_df = X_train.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return sample_df, total_train_records


def compute_global_shap_metrics(
    explainer: MindScopeExplainer, sample_df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes transformed SHAP values and aggregates them into the 12 original features.
    Verifies the Shapley efficiency axiom for all evaluated records.
    """
    X_trans = explainer.transform_data(sample_df)
    preds = explainer.model.predict(X_trans)
    shap_trans = explainer.compute_transformed_shap_values(X_trans)
    grouped_df = explainer.aggregate_to_original_features(shap_trans, predictions=preds, atol=1e-5)

    # 1. Original 12-feature calculations
    mean_abs_shap = grouped_df.abs().mean(axis=0)
    mean_signed_shap = grouped_df.mean(axis=0)
    median_abs_shap = grouped_df.abs().median(axis=0)
    std_shap = grouped_df.std(axis=0)
    min_shap = grouped_df.min(axis=0)
    max_shap = grouped_df.max(axis=0)

    original_records = []
    for feat in PREDICTOR_COLUMNS:
        original_records.append({
            "feature": feat,
            "mean_abs_shap": float(mean_abs_shap[feat]),
            "mean_signed_shap": float(mean_signed_shap[feat]),
            "median_abs_shap": float(median_abs_shap[feat]),
            "std_shap": float(std_shap[feat]),
            "min_shap": float(min_shap[feat]),
            "max_shap": float(max_shap[feat]),
        })
    # Sort descending by mean absolute SHAP
    original_records.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    for rank, item in enumerate(original_records, 1):
        item["rank"] = rank

    # 2. Transformed-space calculations
    trans_mean_abs = np.mean(np.abs(shap_trans), axis=0)
    trans_mean_signed = np.mean(shap_trans, axis=0)
    trans_std = np.std(shap_trans, axis=0)

    transformed_records = []
    for idx, name in enumerate(explainer.transformed_feature_names):
        transformed_records.append({
            "transformed_feature": name,
            "parent_feature": explainer.feature_mapping[name],
            "mean_abs_shap": float(trans_mean_abs[idx]),
            "mean_signed_shap": float(trans_mean_signed[idx]),
            "std_shap": float(trans_std[idx]),
        })
    transformed_records.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    for rank, item in enumerate(transformed_records, 1):
        item["rank"] = rank

    analysis_results = {
        "original_features": original_records,
        "transformed_features": transformed_records,
        "base_value": explainer.expected_value,
        "sample_size": len(sample_df),
    }

    return grouped_df, shap_trans, preds, analysis_results


def plot_global_summary_bar(original_records: List[Dict[str, Any]], output_path: Path):
    """
    Generates a publication-quality horizontal bar chart of mean absolute SHAP importance.
    """
    features = [item["feature"] for item in reversed(original_records)]
    scores = [item["mean_abs_shap"] for item in reversed(original_records)]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(features, scores, color="#2b5c8f", edgecolor="#1a365d", height=0.65, alpha=0.9)

    ax.set_xlabel("Mean |SHAP Value| (Points on Mental Health Score Scale)", fontweight="semibold")
    ax.set_title("Global Feature Importance (12 Original Predictors)\nRandom Forest Pipeline", pad=15, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Annotate numeric values on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(scores) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="semibold",
            color="#2d3748",
        )

    # Footnote
    fig.text(
        0.5,
        0.01,
        "Note: SHAP values quantify statistical model attribution within the fitted response surface, not real-world causal effect sizes.",
        ha="center",
        fontsize=8.5,
        fontstyle="italic",
        color="#718096",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path.name}")


def plot_global_grouped_beeswarm(
    grouped_df: pd.DataFrame, sample_df: pd.DataFrame, original_records: List[Dict[str, Any]], output_path: Path
):
    """
    Generates a grouped beeswarm / strip distribution plot for the 12 original predictors.
    Preserves all individual observation attributions.
    Colors continuous features by standardized value (blue = low, red = high).
    """
    sorted_features = [item["feature"] for item in reversed(original_records)]
    n_samples = len(grouped_df)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    np.random.seed(42)

    for y_idx, feat in enumerate(sorted_features):
        vals = grouped_df[feat].to_numpy()
        # Add slight vertical jitter for beeswarm visualization
        jitter = np.random.normal(0, 0.08, size=n_samples)
        y_positions = y_idx + jitter

        raw_feat_vals = sample_df[feat]
        if pd.api.types.is_numeric_dtype(raw_feat_vals):
            # Normalize continuous values for colormap
            norm_vals = (raw_feat_vals - raw_feat_vals.min()) / (raw_feat_vals.max() - raw_feat_vals.min() + 1e-8)
            scatter = ax.scatter(
                vals, y_positions, c=norm_vals, cmap="coolwarm", s=18, alpha=0.7, edgecolors="none"
            )
        else:
            # Categorical features colored with neutral slate
            ax.scatter(vals, y_positions, color="#4a5568", s=18, alpha=0.6, edgecolors="none")

    ax.axvline(0, color="#a0aec0", linestyle="-", linewidth=1.2, zorder=1)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("SHAP Value (Impact on Model Prediction Relative to Baseline E[y])", fontweight="semibold")
    ax.set_title("Grouped SHAP Beeswarm Distribution (Original 12 Features)\nPreserving Individual Observation Attributions", pad=15, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Colorbar for relative feature value
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=0.7)
    cbar.set_label("Relative Feature Value (Low -> High)", fontsize=9.5)
    cbar.set_ticks([0.1, 0.9])
    cbar.set_ticklabels(["Low Value", "High Value"])

    # Footnote
    fig.text(
        0.5,
        0.01,
        "Note: SHAP values quantify statistical model attribution, not causal effect sizes. Points represent 500 individual training records.",
        ha="center",
        fontsize=8.5,
        fontstyle="italic",
        color="#718096",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path.name}")


def plot_transformed_beeswarm(
    shap_trans: np.ndarray,
    X_trans: np.ndarray,
    transformed_feature_names: List[str],
    transformed_records: List[Dict[str, Any]],
    output_path: Path,
    top_n: int = 20,
):
    """
    Generates a publication-quality beeswarm plot across the top transformed features.
    """
    top_records = transformed_records[:top_n]
    top_feature_names = [item["transformed_feature"] for item in reversed(top_records)]
    top_indices = [transformed_feature_names.index(name) for name in top_feature_names]

    n_samples = shap_trans.shape[0]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    np.random.seed(42)

    for y_idx, (trans_idx, name) in enumerate(zip(top_indices, top_feature_names)):
        vals = shap_trans[:, trans_idx]
        col_vals = X_trans[:, trans_idx]
        jitter = np.random.normal(0, 0.08, size=n_samples)
        y_positions = y_idx + jitter

        norm_vals = (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-8)
        ax.scatter(vals, y_positions, c=norm_vals, cmap="coolwarm", s=18, alpha=0.7, edgecolors="none")

    ax.axvline(0, color="#a0aec0", linestyle="-", linewidth=1.2, zorder=1)
    ax.set_yticks(range(len(top_feature_names)))

    # Clean display labels: strip 'standard_numeric__', 'nominal__', 'skewed__', 'physical__'
    clean_labels = [
        name.replace("standard_numeric__", "")
            .replace("nominal__", "")
            .replace("skewed__", "")
            .replace("physical__", "")
            .replace("country__", "")
        for name in top_feature_names
    ]
    ax.set_yticklabels(clean_labels)
    ax.set_xlabel("SHAP Value (Transformed Space)", fontweight="semibold")
    ax.set_title(f"Top {top_n} Transformed Features: SHAP Beeswarm Distribution", pad=15, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=0.7)
    cbar.set_label("Transformed Feature Magnitude (Low -> High)", fontsize=9.5)
    cbar.set_ticks([0.1, 0.9])
    cbar.set_ticklabels(["Low", "High"])

    fig.text(
        0.5,
        0.01,
        "Note: SHAP values quantify statistical model attribution, not causal effect sizes.",
        ha="center",
        fontsize=8.5,
        fontstyle="italic",
        color="#718096",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path.name}")


def plot_global_distribution(
    grouped_df: pd.DataFrame, original_records: List[Dict[str, Any]], output_path: Path
):
    """
    Generates horizontal box & violin distribution plots for the 12 original predictors.
    Reveals median, interquartile range (IQR), and positive/negative skew.
    """
    sorted_features = [item["feature"] for item in reversed(original_records)]
    data_to_plot = [grouped_df[feat].to_numpy() for feat in sorted_features]

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Violin density representation
    violins = ax.violinplot(
        data_to_plot,
        orientation="horizontal",
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.75,
    )
    for body in violins["bodies"]:
        body.set_facecolor("#3182ce")
        body.set_edgecolor("#1a365d")
        body.set_alpha(0.4)
    violins["cmedians"].set_color("#2b6cb0")
    violins["cmedians"].set_linewidth(1.8)

    # Boxplot overlay showing exact quartiles and outliers
    ax.boxplot(
        data_to_plot,
        orientation="horizontal",
        widths=0.25,
        patch_artist=True,
        boxprops=dict(facecolor="#2b5c8f", color="#1a365d", alpha=0.8),
        medianprops=dict(color="#ecc94b", linewidth=2.0),
        whiskerprops=dict(color="#4a5568", linestyle="--"),
        capprops=dict(color="#4a5568"),
        flierprops=dict(marker="o", color="#718096", markersize=3, alpha=0.5),
    )

    ax.axvline(0, color="#e53e3e", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_yticks(range(1, len(sorted_features) + 1))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("SHAP Value Distribution (Points on Scale)", fontweight="semibold")
    ax.set_title("SHAP Value Dispersion & Spread (Original 12 Features)\nBox & Density Violin Distribution", pad=15, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Footnote
    fig.text(
        0.5,
        0.01,
        "Note: Boxes represent IQR with yellow median line; violins show kernel density. SHAP quantifies statistical attribution, not causation.",
        ha="center",
        fontsize=8.5,
        fontstyle="italic",
        color="#718096",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path.name}")


def execute_global_shap_pipeline(sample_size: int = 500, random_state: int = 42) -> None:
    """
    Master runner for Phase 3, Step 3.2:
    1. Loads the fitted production model.
    2. Samples training records deterministically.
    3. Computes transformed and grouped SHAP values.
    4. Persists structured JSON outputs.
    5. Generates 4 publication-quality PNG visualizations.
    6. Saves analysis metadata.
    """
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"--- Phase 3, Step 3.2: Global SHAP Analysis ---")
    print(f"Loading explainer from {DEFAULT_MODEL_PATH}...")
    explainer = load_explainer()

    print(f"Extracting deterministic training sample (N={sample_size}, seed={random_state})...")
    sample_df, total_train_records = get_deterministic_training_sample(
        sample_size=sample_size, random_state=random_state
    )

    print("Computing global SHAP values across evaluation sample...")
    grouped_df, shap_trans, preds, results = compute_global_shap_metrics(explainer, sample_df)
    X_trans = explainer.transform_data(sample_df)

    # 1. Save global_importance.json
    print(f"Persisting {GLOBAL_IMPORTANCE_JSON.name}...")
    with open(GLOBAL_IMPORTANCE_JSON, "w") as f:
        json.dump(results["original_features"], f, indent=2)

    # 2. Save transformed_importance.json
    print(f"Persisting {TRANSFORMED_IMPORTANCE_JSON.name}...")
    with open(TRANSFORMED_IMPORTANCE_JSON, "w") as f:
        json.dump(results["transformed_features"], f, indent=2)

    # 3. Generate Visualizations
    print("Generating publication-quality visualizations...")
    plot_global_summary_bar(results["original_features"], SUMMARY_BAR_PNG)
    plot_global_grouped_beeswarm(grouped_df, sample_df, results["original_features"], GLOBAL_BEESWARM_PNG)
    plot_transformed_beeswarm(shap_trans, X_trans, explainer.transformed_feature_names, results["transformed_features"], TRANSFORMED_BEESWARM_PNG)
    plot_global_distribution(grouped_df, results["original_features"], DISTRIBUTION_PNG)

    # 4. Save metadata JSON
    computation_time = time.time() - t_start
    metadata = {
        "analysis_identifier": "mindscope_global_shap_v1",
        "model_identifier": "mindscope_rf_v1",
        "model_artifact_path": str(DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "dataset_source": "Student Social Media And Mental Health Impact.csv",
        "training_partition_total_samples": total_train_records,
        "evaluation_sample_size": sample_size,
        "sample_selection_method": f"Deterministic random sample from X_train using random_state={random_state}",
        "sealed_holdout_access": False,
        "base_value_expected_score": explainer.expected_value,
        "feature_count_original": len(PREDICTOR_COLUMNS),
        "feature_count_transformed": explainer.n_transformed_features,
        "shap_version": "0.52.0",
        "python_version": sys.version,
        "platform": platform.platform(),
        "computation_time_seconds": round(computation_time, 2),
        "execution_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "additivity_verification": {
            "verified": True,
            "max_discrepancy_tolerance": 1e-5,
            "observed_max_discrepancy": float(np.max(np.abs(preds - (explainer.expected_value + grouped_df.sum(axis=1).to_numpy())))),
        },
        "feature_mapping_summary": {
            orig: len(indices) for orig, indices in explainer.parent_to_indices.items()
        },
    }
    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {METADATA_JSON.name}")
    print(f"Global SHAP analysis complete in {computation_time:.2f}s.")


if __name__ == "__main__":
    execute_global_shap_pipeline(sample_size=500, random_state=42)
