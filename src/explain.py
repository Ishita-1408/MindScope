import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    EXPECTED_COLUMNS,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
)

# Default model path
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "mindscope_rf_v1.joblib"

# Feature metadata mapping for human-interpretable explanations
ORIGINAL_FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "Avg_Daily_Usage_Hours": {
        "display_name": "Daily Social Media Usage",
        "unit": "hours/day",
        "type": "numerical",
    },
    "Sleep_Hours_Per_Night": {
        "display_name": "Sleep Duration",
        "unit": "hours/night",
        "type": "numerical",
    },
    "Daily_Unlocks": {
        "display_name": "Daily Device Unlocks",
        "unit": "unlocks/day",
        "type": "numerical",
    },
    "Purpose_Of_Use": {
        "display_name": "Primary Purpose of Use",
        "unit": "category",
        "type": "categorical",
    },
    "Country": {
        "display_name": "Country of Residence",
        "unit": "category",
        "type": "categorical",
    },
    "Most_Used_Platform": {
        "display_name": "Most Used Social Platform",
        "unit": "category",
        "type": "categorical",
    },
    "Physical_Activity_Hours": {
        "display_name": "Daily Physical Activity",
        "unit": "hours/day",
        "type": "numerical",
    },
    "Study_Hours": {
        "display_name": "Daily Study Duration",
        "unit": "hours/day",
        "type": "numerical",
    },
    "Age": {
        "display_name": "Student Age",
        "unit": "years",
        "type": "numerical",
    },
    "Academic_Level": {
        "display_name": "Academic Level",
        "unit": "category",
        "type": "categorical",
    },
    "Stress_Level": {
        "display_name": "Self-Reported Stress Level",
        "unit": "category",
        "type": "categorical",
    },
    "Gender": {
        "display_name": "Gender",
        "unit": "category",
        "type": "categorical",
    },
}


def _format_feature_value(feature: str, val: Any) -> str:
    """Formats raw feature values into user-friendly strings with units."""
    if val is None or pd.isna(val):
        return "N/A"
    meta = ORIGINAL_FEATURE_METADATA.get(feature, {})
    unit = meta.get("unit", "")
    if isinstance(val, (float, np.floating)):
        return f"{float(val):.1f} {unit}".strip()
    if isinstance(val, (int, np.integer)):
        return f"{int(val)} {unit}".strip()
    return str(val)


class MindScopeExplainer:
    """
    Core Explainable AI (SHAP) module for the MindScope platform.
    Couples tree-path dependent TreeSHAP to the fitted Random Forest pipeline.
    Provides mathematically verified, additive feature attributions in both
    the 38-dimensional transformed space and the 12-dimensional original feature space.
    """

    def __init__(self, pipeline: Pipeline, check_additivity: bool = True):
        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"Expected scikit-learn Pipeline instance, got {type(pipeline)}")

        if "preprocessor" not in pipeline.named_steps or "model" not in pipeline.named_steps:
            raise ValueError(
                "Pipeline must contain 'preprocessor' and 'model' named steps. "
                f"Found steps: {list(pipeline.named_steps.keys())}"
            )

        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.model = pipeline.named_steps["model"]
        self.check_additivity = check_additivity

        # 1. Dynamically obtain transformed feature names (never hardcode 38 or 39)
        self.transformed_feature_names: List[str] = list(self.preprocessor.get_feature_names_out())
        self.n_transformed_features: int = len(self.transformed_feature_names)

        # 2. Verify feature count matches model's expected inputs
        expected_features = getattr(self.model, "n_features_in_", None)
        if expected_features is not None and self.n_transformed_features != expected_features:
            raise ValueError(
                f"Transformer output dimension ({self.n_transformed_features}) does not match "
                f"estimator input dimension ({expected_features})."
            )

        # 3. Build robust mapping from transformed features to original 12 features
        self.feature_mapping, self.parent_to_indices = self._build_feature_mapping(
            self.transformed_feature_names, PREDICTOR_COLUMNS
        )

        # 4. Initialize TreeExplainer in tree_path_dependent mode
        self.explainer = shap.TreeExplainer(
            model=self.model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

        # 5. Extract expected value as a safe float scalar
        raw_ev = self.explainer.expected_value
        self.expected_value: float = float(np.squeeze(raw_ev))

    @staticmethod
    def _build_feature_mapping(
        transformed_names: List[str], predictor_columns: List[str]
    ) -> Tuple[Dict[str, str], Dict[str, List[int]]]:
        """
        Dynamically maps each transformed feature name to exactly one parent original predictor.
        Fails loudly if any transformed feature cannot be mapped or matches ambiguously.
        """
        mapping: Dict[str, str] = {}
        parent_to_indices: Dict[str, List[int]] = {col: [] for col in predictor_columns}

        for idx, trans_name in enumerate(transformed_names):
            # ColumnTransformer prefixes names as: '<branch_prefix>__<column_name>_<category>'
            matched_parents = [orig for orig in predictor_columns if orig in trans_name]

            if len(matched_parents) == 0:
                raise ValueError(
                    f"Transformed feature '{trans_name}' could not be mapped to any original predictor. "
                    f"Known predictors: {predictor_columns}"
                )
            elif len(matched_parents) > 1:
                # Disambiguate exact substring match if multiple candidates match
                exact_matches = [
                    orig for orig in matched_parents
                    if f"__{orig}" in trans_name or f"_{orig}_" in trans_name
                ]
                if len(exact_matches) == 1:
                    parent = exact_matches[0]
                else:
                    raise ValueError(
                        f"Ambiguous mapping for transformed feature '{trans_name}'. "
                        f"Matches multiple original predictors: {matched_parents}"
                    )
            else:
                parent = matched_parents[0]

            mapping[trans_name] = parent
            parent_to_indices[parent].append(idx)

        # Verify no original feature was omitted
        unmapped_parents = [p for p, indices in parent_to_indices.items() if len(indices) == 0]
        if unmapped_parents:
            raise ValueError(f"The following original predictors have no mapped transformed columns: {unmapped_parents}")

        # Verify total mapped transformed features equals exact length
        if len(mapping) != len(transformed_names):
            raise ValueError(
                f"Mapping size mismatch: mapped {len(mapping)} features, expected {len(transformed_names)}"
            )

        return mapping, parent_to_indices

    def transform_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Validates input schema and transforms raw DataFrame into model-ready numerical feature matrix.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input data must be a pandas DataFrame, got {type(X)}")

        missing_cols = set(PREDICTOR_COLUMNS) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required predictor columns: {missing_cols}")

        X_input = X[PREDICTOR_COLUMNS].copy()
        X_trans = self.preprocessor.transform(X_input)

        # Ensure output is dense 2D float array
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        return np.asarray(X_trans, dtype=np.float64)

    def compute_transformed_shap_values(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Calculates raw SHAP values in the transformed feature space.
        Returns a 2D float64 array of shape (N, n_transformed_features).
        """
        if not isinstance(X_transformed, np.ndarray):
            X_transformed = np.asarray(X_transformed, dtype=np.float64)

        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(1, -1)

        if X_transformed.shape[1] != self.n_transformed_features:
            raise ValueError(
                f"Transformed matrix has {X_transformed.shape[1]} features, "
                f"expected {self.n_transformed_features}"
            )

        shap_vals = self.explainer.shap_values(X_transformed)

        # In modern SHAP releases, regression shap_values can be an ndarray
        shap_vals = np.asarray(shap_vals, dtype=np.float64)
        if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
            shap_vals = np.squeeze(shap_vals, axis=-1)

        return shap_vals

    def aggregate_to_original_features(
        self,
        shap_transformed: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        atol: float = 1e-5,
    ) -> pd.DataFrame:
        """
        Aggregates transformed SHAP values into the 12 original input features.
        Mathematically verifies the efficiency axiom:
        base_value + sum(grouped_SHAP) == prediction (within atol=1e-5).
        """
        if shap_transformed.ndim == 1:
            shap_transformed = shap_transformed.reshape(1, -1)

        n_samples = shap_transformed.shape[0]
        grouped_dict: Dict[str, np.ndarray] = {}

        # Aggregate transformed indicator columns into parent features
        for parent, indices in self.parent_to_indices.items():
            if len(indices) == 1:
                grouped_dict[parent] = shap_transformed[:, indices[0]]
            else:
                grouped_dict[parent] = np.sum(shap_transformed[:, indices], axis=1)

        grouped_df = pd.DataFrame(grouped_dict, columns=PREDICTOR_COLUMNS)

        # Mathematical Invariance 1: Sum across 12 features must equal sum across transformed features
        sum_grouped = grouped_df.sum(axis=1).to_numpy()
        sum_transformed = np.sum(shap_transformed, axis=1)
        internal_diff = np.max(np.abs(sum_grouped - sum_transformed))
        if internal_diff > atol:
            raise RuntimeError(
                f"Internal aggregation inconsistency: grouped sum differs from transformed sum "
                f"by {internal_diff:.4e} (tolerance: {atol:.4e})"
            )

        # Mathematical Invariance 2: Efficiency Axiom (base_value + sum(SHAP) == model_prediction)
        if self.check_additivity and predictions is not None:
            predictions = np.asarray(predictions, dtype=np.float64)
            reconstructed = self.expected_value + sum_grouped
            discrepancies = np.abs(predictions - reconstructed)
            max_discrepancy = float(np.max(discrepancies))

            if max_discrepancy > atol:
                max_idx = int(np.argmax(discrepancies))
                raise RuntimeError(
                    f"SHAP efficiency axiom additivity violation for observation {max_idx}: "
                    f"model prediction = {predictions[max_idx]:.6f}, "
                    f"reconstructed (base + sum(SHAP)) = {reconstructed[max_idx]:.6f}, "
                    f"absolute discrepancy = {discrepancies[max_idx]:.6e} (tolerance: {atol:.4e})"
                )

        return grouped_df

    def explain_prediction(
        self,
        record: Union[Dict[str, Any], pd.DataFrame],
        atol: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive, verified individual SHAP explanation for a single student profile.
        Returns prediction, base_value, net_attribution, ranked feature contributions,
        and transformed indicator attributions.
        """
        if isinstance(record, dict):
            df = pd.DataFrame([record])
        elif isinstance(record, pd.DataFrame):
            if len(record) != 1:
                raise ValueError(f"explain_prediction expects a single record, got {len(record)} rows.")
            df = record.copy()
        else:
            raise TypeError(f"record must be a dict or 1-row DataFrame, got {type(record)}")

        missing_cols = set(PREDICTOR_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Input record is missing required predictor columns: {missing_cols}")

        # 1. Generate model prediction
        pred = float(self.pipeline.predict(df[PREDICTOR_COLUMNS])[0])

        # 2. Transform input features
        X_trans = self.transform_data(df)

        # 3. Compute transformed SHAP values
        shap_trans = self.compute_transformed_shap_values(X_trans)

        # 4. Aggregate to 12 original features and verify additivity
        grouped_df = self.aggregate_to_original_features(
            shap_trans, predictions=np.array([pred]), atol=atol
        )

        # 5. Build structured feature attributions list with metadata
        raw_attributions = []
        for feature in PREDICTOR_COLUMNS:
            raw_val = df[feature].iloc[0]
            val_shap = float(grouped_df[feature].iloc[0])
            meta = ORIGINAL_FEATURE_METADATA.get(feature, {})
            # Ensure standard JSON-serializable Python scalar
            clean_val = raw_val
            if hasattr(raw_val, "item"):
                clean_val = raw_val.item()
            elif pd.isna(raw_val):
                clean_val = None

            raw_attributions.append({
                "feature": feature,
                "display_name": meta.get("display_name", feature),
                "value": clean_val,
                "unit": meta.get("unit", ""),
                "formatted_value": _format_feature_value(feature, clean_val),
                "shap_value": val_shap,
                "direction": "positive" if val_shap >= 0.0 else "negative",
                "abs_impact": float(abs(val_shap)),
            })

        # Rank features strictly by absolute impact descending
        raw_attributions.sort(key=lambda x: x["abs_impact"], reverse=True)

        # 6. Compute cumulative waterfall steps
        feature_attributions = []
        running_score = self.expected_value
        waterfall_steps = [
            {
                "step_type": "baseline",
                "label": "Expected Baseline E[y]",
                "feature": None,
                "delta": 0.0,
                "cumulative_score": round(self.expected_value, 4),
            }
        ]

        for rank_idx, item in enumerate(raw_attributions, 1):
            step_start = running_score
            running_score += item["shap_value"]
            item["rank"] = rank_idx
            item["step_start"] = float(step_start)
            item["cumulative_score"] = float(running_score)
            feature_attributions.append(item)

            waterfall_steps.append({
                "step_type": "attribution",
                "label": item["display_name"],
                "feature": item["feature"],
                "delta": item["shap_value"],
                "cumulative_score": round(running_score, 4),
            })

        waterfall_steps.append({
            "step_type": "prediction",
            "label": "Final Model Prediction",
            "feature": None,
            "delta": float(pred - self.expected_value),
            "cumulative_score": round(pred, 4),
        })

        # Detailed transformed-space dictionary
        transformed_attributions = {
            name: float(shap_trans[0, idx])
            for idx, name in enumerate(self.transformed_feature_names)
        }

        # Verification metrics
        sum_shap = float(grouped_df.iloc[0].sum())
        reconstructed = self.expected_value + sum_shap
        abs_err = abs(pred - reconstructed)

        return {
            "prediction": pred,
            "base_value": self.expected_value,
            "net_attribution": float(pred - self.expected_value),
            "feature_attributions": feature_attributions,
            "waterfall_steps": waterfall_steps,
            "transformed_attributions": transformed_attributions,
            "additivity_verified": bool(abs_err <= atol),
            "additivity_error": abs_err,
            "disclaimer": (
                "Contribution direction describes how a feature shifts this model's "
                "prediction relative to its expected baseline. It does not indicate "
                "better or worse mental health, establish causality, or provide a clinical assessment."
            ),
        }

    def compute_global_importance(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates global feature importance (mean absolute SHAP) across a dataset of observations.
        Returns ranked importance for both the 12 original predictors and the 38 transformed features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

        X_trans = self.transform_data(X)
        preds = self.model.predict(X_trans)
        shap_trans = self.compute_transformed_shap_values(X_trans)
        grouped_df = self.aggregate_to_original_features(shap_trans, predictions=preds)

        # 1. Original 12-feature global importance
        original_mean_abs = grouped_df.abs().mean(axis=0)
        original_importance = [
            {"feature": feat, "mean_abs_shap": float(original_mean_abs[feat])}
            for feat in PREDICTOR_COLUMNS
        ]
        original_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        # 2. Transformed-space global importance
        trans_mean_abs = np.mean(np.abs(shap_trans), axis=0)
        transformed_importance = [
            {"feature": name, "mean_abs_shap": float(trans_mean_abs[idx])}
            for idx, name in enumerate(self.transformed_feature_names)
        ]
        transformed_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        return {
            "sample_count": int(len(X)),
            "base_value": self.expected_value,
            "original_feature_importance": original_importance,
            "transformed_feature_importance": transformed_importance,
        }


def load_explainer(model_path: Optional[Union[str, Path]] = None) -> MindScopeExplainer:
    """
    Convenience factory to load the fitted production pipeline and initialize the explainer.
    Defaults to 'artifacts/models/mindscope_rf_v1.joblib'.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Fitted model artifact not found at {model_path}")

    pipeline = joblib.load(model_path)
    return MindScopeExplainer(pipeline)

