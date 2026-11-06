import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    EXPECTED_COLUMNS,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    load_and_validate_data,
)

# Directory configurations
DATA_FILE = PROJECT_ROOT / "Student Social Media And Mental Health Impact.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"
SEALED_HOLDOUT_FILE = PROJECT_ROOT / "artifacts" / "sealed_holdout_test.csv"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes unrounded primary regression metrics and supplementary tolerance metrics.
    Primary metrics: R², MAE, RMSE (unrounded float64).
    Tolerance metrics: Tol_0.1 (±0.10 points), Tol_0.2 (±0.20 points).
    """
    diff = np.abs(y_true - y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "tol_0.1": float(np.mean(diff <= 0.10) * 100.0),
        "tol_0.2": float(np.mean(diff <= 0.20) * 100.0),
    }


def partition_and_seal_data(
    data_path: Path = DATA_FILE,
    random_state: int = 42,
    test_size: float = 0.30,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads dataset, removes exact duplicate records, and partitions into:
    - 70% Training Set (N=3,498)
    - 30% Sealed Holdout Test Set (N=1,500)
    Saves sealed holdout to disk; training functions strictly receive training splits.
    """
    X, y = load_and_validate_data(data_path, exclude_negative_activity=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Persist sealed holdout split
    SEALED_HOLDOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    holdout_df = X_test.copy()
    holdout_df[TARGET_COLUMN] = y_test
    holdout_df.to_csv(SEALED_HOLDOUT_FILE, index=False)

    return X_train, y_train, X_test, y_test


def get_candidate_configurations() -> List[Dict[str, Any]]:
    """
    Defines the matrix of candidate models, feature schemas, and activity treatments.
    Each configuration specifies:
    - config_id: Unique string identifier.
    - algorithm: Family name.
    - estimator: Scikit-learn estimator instance.
    - param_grid: Bounded hyperparameter distributions for inner CV tuning.
    - drop_purpose: True for Reduced (11 features), False for Full (12 features).
    - activity_strategy: 'clip', 'nan', or 'passthrough'.
    - exclude_negative: True for sensitivity analysis excluding negative activity rows.
    """
    return [
        # 1. Baseline Ordinary Least Squares (Full Schema, Bounded Clip)
        {
            "config_id": "M1_OLS_Full_Clip",
            "algorithm": "LinearRegression",
            "estimator": LinearRegression(),
            "param_grid": {},
            "drop_purpose": False,
            "activity_strategy": "clip",
            "exclude_negative": False,
        },
        # 2. Regularized Ridge (Full Schema, Bounded Clip)
        {
            "config_id": "M2a_Ridge_Full_Clip",
            "algorithm": "Ridge",
            "estimator": Ridge(),
            "param_grid": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
            "drop_purpose": False,
            "activity_strategy": "clip",
            "exclude_negative": False,
        },
        # 3. Regularized Ridge (Reduced Schema, Bounded Clip)
        {
            "config_id": "M2b_Ridge_Reduced_Clip",
            "algorithm": "Ridge",
            "estimator": Ridge(),
            "param_grid": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
            "drop_purpose": True,
            "activity_strategy": "clip",
            "exclude_negative": False,
        },
        # 4. HistGradientBoosting (Full Schema, Bounded Clip)
        {
            "config_id": "M3a_HistGBR_Full_Clip",
            "algorithm": "HistGradientBoostingRegressor",
            "estimator": HistGradientBoostingRegressor(random_state=42),
            "param_grid": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [100, 150],
                "model__max_depth": [4, 6, 8],
                "model__min_samples_leaf": [15, 25],
                "model__l2_regularization": [0.0, 1.0, 5.0],
            },
            "drop_purpose": False,
            "activity_strategy": "clip",
            "exclude_negative": False,
        },
        # 5. HistGradientBoosting (Full Schema, NaN-Impute Sensitivity)
        {
            "config_id": "M3b_HistGBR_Full_NaNImpute",
            "algorithm": "HistGradientBoostingRegressor",
            "estimator": HistGradientBoostingRegressor(random_state=42),
            "param_grid": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [100, 150],
                "model__max_depth": [4, 6, 8],
                "model__min_samples_leaf": [15, 25],
                "model__l2_regularization": [0.0, 1.0, 5.0],
            },
            "drop_purpose": False,
            "activity_strategy": "nan",
            "exclude_negative": False,
        },
        # 6. HistGradientBoosting (Reduced Schema, Bounded Clip)
        {
            "config_id": "M3c_HistGBR_Reduced_Clip",
            "algorithm": "HistGradientBoostingRegressor",
            "estimator": HistGradientBoostingRegressor(random_state=42),
            "param_grid": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [100, 150],
                "model__max_depth": [4, 6, 8],
                "model__min_samples_leaf": [15, 25],
                "model__l2_regularization": [0.0, 1.0, 5.0],
            },
            "drop_purpose": True,
            "activity_strategy": "clip",
            "exclude_negative": False,
        },
        # 7. Random Forest (Full Schema, Bounded Clip)
        {
            "config_id": "M4a_RF_Full_Clip",
            "algorithm": "RandomForestRegressor",
            "estimator": RandomForestRegressor(random_state=42),
            "param_grid": {
                "model__n_estimators": [100, 150],
                "model__max_depth": [10, 15, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [2, 4],
            },
            "drop_purpose": False,
            "activity_strategy": "clip",
            "exclude_negative": False,
        },
        # 8. Random Forest (Reduced Schema, Bounded Clip)
        {
            "config_id": "M4b_RF_Reduced_Clip",
            "algorithm": "RandomForestRegressor",
            "estimator": RandomForestRegressor(random_state=42),
            "param_grid": {
                "model__n_estimators": [100, 150],
                "model__max_depth": [10, 15, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [2, 4],
            },
            "drop_purpose": True,
            "activity_strategy": "clip",
            "exclude_negative": False,
        },
        # 9. Random Forest (Full Schema, Observation Exclusion Sensitivity)
        {
            "config_id": "M4c_RF_Full_Exclude",
            "algorithm": "RandomForestRegressor",
            "estimator": RandomForestRegressor(random_state=42),
            "param_grid": {
                "model__n_estimators": [100, 150],
                "model__max_depth": [10, 15, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [2, 4],
            },
            "drop_purpose": False,
            "activity_strategy": "clip",
            "exclude_negative": True,
        },
    ]


def run_inner_cv_selection(
    X_outer_tr: pd.DataFrame,
    y_outer_tr: pd.Series,
    candidate_configs: List[Dict[str, Any]],
    n_inner_splits: int = 5,
    search_n_iter: int = 8,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Pipeline, Dict[str, Any], float]:
    """
    Executes 5-fold inner CV across all candidate modeling configurations on outer training data.
    Selects the complete configuration (family, hyperparameters, feature set, activity treatment)
    achieving the highest mean validation R².
    """
    inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)

    best_score = -float("inf")
    best_config = None
    best_pipeline = None
    best_params = {}

    for cfg in candidate_configs:
        # If this configuration tests row exclusion sensitivity, filter outer train slice
        if cfg.get("exclude_negative", False):
            mask = X_outer_tr["Physical_Activity_Hours"] >= 0.0
            X_tr_inner = X_outer_tr[mask].reset_index(drop=True)
            y_tr_inner = y_outer_tr[mask].reset_index(drop=True)
        else:
            X_tr_inner = X_outer_tr
            y_tr_inner = y_outer_tr

        preprocessor = build_preprocessor(
            drop_purpose=cfg["drop_purpose"],
            activity_strategy=cfg["activity_strategy"],
        )
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", clone(cfg["estimator"])),
        ])

        if cfg["param_grid"]:
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=cfg["param_grid"],
                n_iter=search_n_iter,
                cv=inner_cv,
                scoring="r2",
                random_state=seed,
                n_jobs=-1,
            )
            search.fit(X_tr_inner, y_tr_inner)
            score = float(search.best_score_)
            fitted_pipe = search.best_estimator_
            params = search.best_params_
        else:
            # Deterministic evaluation across inner CV
            fold_r2s = []
            for i_tr_idx, i_va_idx in inner_cv.split(X_tr_inner, y_tr_inner):
                i_X_tr, i_y_tr = X_tr_inner.iloc[i_tr_idx], y_tr_inner.iloc[i_tr_idx]
                i_X_va, i_y_va = X_tr_inner.iloc[i_va_idx], y_tr_inner.iloc[i_va_idx]
                f_pipe = clone(pipe)
                f_pipe.fit(i_X_tr, i_y_tr)
                fold_r2s.append(r2_score(i_y_va, f_pipe.predict(i_X_va)))
            score = float(np.mean(fold_r2s))
            pipe.fit(X_tr_inner, y_tr_inner)
            fitted_pipe = pipe
            params = {}

        if score > best_score:
            best_score = score
            best_config = cfg
            best_pipeline = fitted_pipe
            best_params = params

    return best_config, best_pipeline, best_params, best_score


def run_nested_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    search_n_iter: int = 8,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes rigorous 5-fold outer and 5-fold inner nested cross-validation.
    Inner CV selects the complete configuration for each outer fold.
    Outer CV measures the unbiased generalization of the complete selection procedure.
    """
    outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
    candidate_configs = get_candidate_configurations()

    outer_results = []
    all_config_fold_metrics: Dict[str, List[Dict[str, Any]]] = {
        cfg["config_id"]: [] for cfg in candidate_configs
    }

    start_total_time = time.time()

    for fold_idx, (outer_tr_idx, outer_va_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
        print(f"--- Outer Fold {fold_idx}/{n_outer_splits} ---")
        X_outer_tr = X_train.iloc[outer_tr_idx].reset_index(drop=True)
        y_outer_tr = y_train.iloc[outer_tr_idx].reset_index(drop=True)
        X_outer_va = X_train.iloc[outer_va_idx].reset_index(drop=True)
        y_outer_va = y_train.iloc[outer_va_idx].reset_index(drop=True)

        # 1. Run Complete Inner CV Model Selection
        t0 = time.time()
        best_cfg, best_pipe, best_params, inner_score = run_inner_cv_selection(
            X_outer_tr=X_outer_tr,
            y_outer_tr=y_outer_tr,
            candidate_configs=candidate_configs,
            n_inner_splits=n_inner_splits,
            search_n_iter=search_n_iter,
            seed=random_state + fold_idx,
        )
        inner_duration = time.time() - t0

        # 2. Evaluate Selected Winner on Outer Validation Split
        t_pred_start = time.time()
        val_preds = best_pipe.predict(X_outer_va)
        pred_duration = time.time() - t_pred_start

        train_preds = best_pipe.predict(X_outer_tr)
        train_r2 = float(r2_score(y_outer_tr, train_preds))

        fold_metrics = calculate_metrics(y_outer_va.to_numpy(), val_preds)
        fold_metrics["train_r2"] = train_r2
        fold_metrics["overfit_gap"] = train_r2 - fold_metrics["r2"]
        fold_metrics["selected_config"] = best_cfg["config_id"]
        fold_metrics["selected_algorithm"] = best_cfg["algorithm"]
        fold_metrics["selected_features"] = "Reduced" if best_cfg["drop_purpose"] else "Full"
        fold_metrics["selected_activity_strategy"] = best_cfg["activity_strategy"]
        fold_metrics["inner_cv_best_r2"] = inner_score
        fold_metrics["best_params"] = best_params
        fold_metrics["training_time_sec"] = round(inner_duration, 2)
        fold_metrics["prediction_time_sec"] = round(pred_duration, 4)
        fold_metrics["outer_fold"] = fold_idx

        outer_results.append(fold_metrics)

        print(
            f"  Winner: {best_cfg['config_id']} | Inner R²: {inner_score:.4f} | "
            f"Outer Val R²: {fold_metrics['r2']:.4f} | MAE: {fold_metrics['mae']:.4f} | "
            f"RMSE: {fold_metrics['rmse']:.4f} | Gap: {fold_metrics['overfit_gap']:.4f}"
        )

        # 3. Transparently evaluate every individual configuration on this fold for detailed reporting
        for cfg in candidate_configs:
            cfg_id = cfg["config_id"]
            if cfg.get("exclude_negative", False):
                tr_mask = X_outer_tr["Physical_Activity_Hours"] >= 0.0
                va_mask = X_outer_va["Physical_Activity_Hours"] >= 0.0
                f_X_tr, f_y_tr = X_outer_tr[tr_mask].reset_index(drop=True), y_outer_tr[tr_mask].reset_index(drop=True)
                f_X_va, f_y_va = X_outer_va[va_mask].reset_index(drop=True), y_outer_va[va_mask].reset_index(drop=True)
            else:
                f_X_tr, f_y_tr = X_outer_tr, y_outer_tr
                f_X_va, f_y_va = X_outer_va, y_outer_va

            p = Pipeline([
                ("preprocessor", build_preprocessor(drop_purpose=cfg["drop_purpose"], activity_strategy=cfg["activity_strategy"])),
                ("model", clone(cfg["estimator"])),
            ])
            t_fit = time.time()
            if cfg["param_grid"]:
                s = RandomizedSearchCV(
                    estimator=p,
                    param_distributions=cfg["param_grid"],
                    n_iter=search_n_iter,
                    cv=KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state + fold_idx),
                    scoring="r2",
                    random_state=random_state + fold_idx,
                    n_jobs=-1,
                )
                s.fit(f_X_tr, f_y_tr)
                f_pipe = s.best_estimator_
            else:
                p.fit(f_X_tr, f_y_tr)
                f_pipe = p
            fit_time = time.time() - t_fit

            t_p = time.time()
            preds_va = f_pipe.predict(f_X_va)
            p_time = time.time() - t_p
            preds_tr = f_pipe.predict(f_X_tr)

            m = calculate_metrics(f_y_va.to_numpy(), preds_va)
            tr_r2 = float(r2_score(f_y_tr, preds_tr))
            m["train_r2"] = tr_r2
            m["overfit_gap"] = tr_r2 - m["r2"]
            m["training_time_sec"] = round(fit_time, 2)
            m["prediction_time_sec"] = round(p_time, 4)
            m["outer_fold"] = fold_idx
            all_config_fold_metrics[cfg_id].append(m)

    total_time = time.time() - start_total_time

    # Aggregate Nested CV Generalization
    nested_summary = {
        "outer_r2_mean": float(np.mean([m["r2"] for m in outer_results])),
        "outer_r2_std": float(np.std([m["r2"] for m in outer_results])),
        "outer_mae_mean": float(np.mean([m["mae"] for m in outer_results])),
        "outer_mae_std": float(np.std([m["mae"] for m in outer_results])),
        "outer_rmse_mean": float(np.mean([m["rmse"] for m in outer_results])),
        "outer_rmse_std": float(np.std([m["rmse"] for m in outer_results])),
        "outer_tol_0.1_mean": float(np.mean([m["tol_0.1"] for m in outer_results])),
        "outer_tol_0.2_mean": float(np.mean([m["tol_0.2"] for m in outer_results])),
        "outer_train_r2_mean": float(np.mean([m["train_r2"] for m in outer_results])),
        "outer_overfit_gap_mean": float(np.mean([m["overfit_gap"] for m in outer_results])),
        "total_nested_cv_time_sec": round(total_time, 2),
        "outer_folds": outer_results,
    }

    # Aggregate Configuration-Level Benchmarks
    config_benchmarks = []
    for cfg in candidate_configs:
        cid = cfg["config_id"]
        ms = all_config_fold_metrics[cid]
        config_benchmarks.append({
            "config_id": cid,
            "algorithm": cfg["algorithm"],
            "features": "Reduced (11)" if cfg["drop_purpose"] else "Full (12)",
            "activity_strategy": cfg["activity_strategy"],
            "exclude_negative": cfg.get("exclude_negative", False),
            "r2_mean": float(np.mean([m["r2"] for m in ms])),
            "r2_std": float(np.std([m["r2"] for m in ms])),
            "mae_mean": float(np.mean([m["mae"] for m in ms])),
            "mae_std": float(np.std([m["mae"] for m in ms])),
            "rmse_mean": float(np.mean([m["rmse"] for m in ms])),
            "rmse_std": float(np.std([m["rmse"] for m in ms])),
            "tol_0.1_mean": float(np.mean([m["tol_0.1"] for m in ms])),
            "tol_0.2_mean": float(np.mean([m["tol_0.2"] for m in ms])),
            "train_r2_mean": float(np.mean([m["train_r2"] for m in ms])),
            "overfit_gap_mean": float(np.mean([m["overfit_gap"] for m in ms])),
            "mean_fit_time_sec": float(np.mean([m["training_time_sec"] for m in ms])),
            "mean_pred_time_sec": float(np.mean([m["prediction_time_sec"] for m in ms])),
        })

    return {
        "nested_selection_procedure_summary": nested_summary,
        "configuration_benchmarks": config_benchmarks,
        "all_config_fold_metrics": all_config_fold_metrics,
    }


def main():
    print("=" * 80)
    print("MindScope Phase 2, Step 2.2: Model Training & Nested Cross-Validation")
    print("=" * 80)

    # 1. Partition Data and Seal Holdout Test Set
    X_train, y_train, X_test, y_test = partition_and_seal_data()
    print(f"Data Partitioned:")
    print(f"  - Training Set (N={len(X_train)}): Used strictly for 5x5 Nested CV")
    print(f"  - Sealed Holdout Test Set (N={len(X_test)}): Saved to {SEALED_HOLDOUT_FILE}")
    print("  [CRITICAL SAFEGUARD] Holdout split will NOT be evaluated during this step.\n")

    # 2. Run 5-Fold Outer and 5-Fold Inner Nested CV
    print("Launching 5-Fold Outer and 5-Fold Inner Nested Cross-Validation...")
    experiment_results = run_nested_cross_validation(
        X_train=X_train,
        y_train=y_train,
        n_outer_splits=5,
        n_inner_splits=5,
        search_n_iter=6,
        random_state=42,
    )

    # 3. Save Structured Experiment Outputs
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    nested_summary_file = EXPERIMENTS_DIR / "nested_cv_summary.json"
    with open(nested_summary_file, "w") as f:
        json.dump(experiment_results["nested_selection_procedure_summary"], f, indent=2)

    benchmarks_file = EXPERIMENTS_DIR / "all_configurations_benchmarks.json"
    with open(benchmarks_file, "w") as f:
        json.dump(experiment_results["configuration_benchmarks"], f, indent=2)

    # Flatten fold-level metrics for CSV export
    fold_rows = []
    for cid, folds in experiment_results["all_config_fold_metrics"].items():
        for m in folds:
            row = {"config_id": cid, **m}
            fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(EXPERIMENTS_DIR / "fold_level_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("Nested Cross-Validation Unbiased Generalisation Estimate:")
    print("=" * 80)
    proc_summary = experiment_results["nested_selection_procedure_summary"]
    print(f"Outer Validation R² (Unbiased) : {proc_summary['outer_r2_mean']:.4f} +/- {proc_summary['outer_r2_std']:.4f}")
    print(f"Outer Validation MAE           : {proc_summary['outer_mae_mean']:.4f} +/- {proc_summary['outer_mae_std']:.4f}")
    print(f"Outer Validation RMSE          : {proc_summary['outer_rmse_mean']:.4f} +/- {proc_summary['outer_rmse_std']:.4f}")
    print(f"Outer Validation Tol ±0.1      : {proc_summary['outer_tol_0.1_mean']:.2f}%")
    print(f"Outer Validation Tol ±0.2      : {proc_summary['outer_tol_0.2_mean']:.2f}%")
    print(f"Mean Overfit Gap (Train - Val) : {proc_summary['outer_overfit_gap_mean']:.4f}")
    print(f"Total Nested CV Execution Time : {proc_summary['total_nested_cv_time_sec']:.2f} seconds")

    print("\n" + "=" * 80)
    print("Individual Configuration Benchmarks (Ranked by Outer Fold R²):")
    print("=" * 80)
    bench_df = pd.DataFrame(experiment_results["configuration_benchmarks"])
    bench_df = bench_df.sort_values(by="r2_mean", ascending=False).reset_index(drop=True)
    display_cols = ["config_id", "algorithm", "features", "activity_strategy", "r2_mean", "r2_std", "mae_mean", "rmse_mean", "overfit_gap_mean", "mean_fit_time_sec"]
    print(bench_df[display_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("Step 2.2 Complete: Model Experimentation & Nested CV Finished.")
    print("Results persisted to artifacts/experiments/.")
    print("Per instructions: Model packaging and holdout evaluation are DEFERRED to subsequent steps.")
    print("=" * 80)


if __name__ == "__main__":
    main()
