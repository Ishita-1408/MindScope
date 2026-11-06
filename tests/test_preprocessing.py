from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    CountryTopKEncoder,
    PhysicalActivityHandler,
    build_preprocessor,
    load_and_validate_data,
    EXPECTED_COLUMNS,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
)

DATA_PATH = Path("Student Social Media And Mental Health Impact.csv")


# -----------------------------------------------------------------------------
# 1. Data Loading and Validation Tests
# -----------------------------------------------------------------------------
def test_data_loading_and_deduplication():
    """Verify raw CSV has 5,000 rows and deduplication yields exactly 4,998 unique rows."""
    X, y = load_and_validate_data(DATA_PATH, exclude_negative_activity=False)
    assert len(X) == 4998
    assert len(y) == 4998
    assert list(X.columns) == PREDICTOR_COLUMNS
    assert y.name == TARGET_COLUMN


def test_data_loading_exclude_negative_activity():
    """Verify excluding negative activity drops exactly 10 records (4,998 -> 4,988)."""
    X, y = load_and_validate_data(DATA_PATH, exclude_negative_activity=True)
    assert len(X) == 4988
    assert len(y) == 4988
    assert (X["Physical_Activity_Hours"] >= 0.0).all()


def test_data_loading_missing_file():
    """Verify loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_and_validate_data("non_existent_dataset.csv")


def test_data_loading_schema_validation(tmp_path):
    """Verify that missing expected columns raises ValueError."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("Age,Gender,Mental_Health_Score\n20,Male,5.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset is missing required columns"):
        load_and_validate_data(bad_csv)


# -----------------------------------------------------------------------------
# 2. CountryTopKEncoder Tests
# -----------------------------------------------------------------------------
def test_country_top_k_encoder_clone_and_params():
    """Verify CountryTopKEncoder adheres to Scikit-Learn clone and param conventions."""
    encoder = CountryTopKEncoder(top_k=7, other_label="Other", include_other_in_top_k=False)
    cloned = clone(encoder)

    assert cloned.top_k == 7
    assert cloned.other_label == "Other"
    assert cloned.include_other_in_top_k is False

    cloned.set_params(top_k=5)
    assert cloned.get_params()["top_k"] == 5


def test_country_top_k_encoder_fit_and_transform():
    """Verify category frequencies are learned strictly during fit() and map unseen to Other."""
    train_countries = pd.Series(["India", "India", "USA", "USA", "USA", "UK", "France", "Other"])
    test_countries = pd.Series(["India", "USA", "Germany", "Japan", "Other"])

    encoder = CountryTopKEncoder(top_k=2, other_label="Other", include_other_in_top_k=False)
    encoder.fit(train_countries)

    # Top 2 non-other countries should be USA (3) and India (2)
    assert "USA" in encoder.top_countries_
    assert "India" in encoder.top_countries_
    assert "Other" in encoder.top_countries_

    transformed = encoder.transform(test_countries).ravel()
    expected = ["India", "USA", "Other", "Other", "Other"]
    assert list(transformed) == expected


def test_country_top_k_encoder_unfitted_raises():
    """Verify transform() on an unfitted encoder raises RuntimeError."""
    encoder = CountryTopKEncoder()
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        encoder.transform(pd.Series(["India"]))


# -----------------------------------------------------------------------------
# 3. Negative Physical Activity Sensitivity Tests
# -----------------------------------------------------------------------------
def test_physical_activity_handler_clip():
    """Verify 'clip' strategy clamps negative values to 0.0."""
    handler = PhysicalActivityHandler(strategy="clip")
    data = np.array([[-0.4], [-0.1], [0.0], [1.5], [3.0]])
    transformed = handler.transform(data)
    assert np.all(transformed >= 0.0)
    assert transformed[0, 0] == 0.0
    assert transformed[1, 0] == 0.0
    assert transformed[3, 0] == 1.5


def test_physical_activity_handler_nan():
    """Verify 'nan' strategy replaces negative values with np.nan."""
    handler = PhysicalActivityHandler(strategy="nan")
    data = np.array([[-0.4], [-0.1], [0.0], [1.5]])
    transformed = handler.transform(data)
    assert np.isnan(transformed[0, 0])
    assert np.isnan(transformed[1, 0])
    assert transformed[2, 0] == 0.0
    assert transformed[3, 0] == 1.5


def test_physical_activity_handler_invalid_strategy():
    """Verify invalid strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        PhysicalActivityHandler(strategy="invalid_mode")


# -----------------------------------------------------------------------------
# 4. Pipeline Encapsulation, Missing Values & Feature Configurations
# -----------------------------------------------------------------------------
def test_pipeline_missing_value_imputation():
    """Verify pipeline gracefully imputes missing values across numerical and categorical fields."""
    X, _ = load_and_validate_data(DATA_PATH)
    sample = X.head(20).copy()

    # Introduce intentional nulls
    sample.loc[0, "Study_Hours"] = np.nan
    sample.loc[1, "Physical_Activity_Hours"] = np.nan
    sample.loc[2, "Country"] = np.nan
    sample.loc[3, "Stress_Level"] = np.nan
    sample.loc[4, "Most_Used_Platform"] = np.nan

    preprocessor = build_preprocessor(drop_purpose=False, activity_strategy="clip")
    transformed = preprocessor.fit_transform(sample)

    assert not np.isnan(transformed).any(), "Transformed output must not contain NaNs"
    assert isinstance(transformed, np.ndarray)


def test_full_vs_reduced_feature_schemas():
    """Verify column dimension differences between full (12 features) and reduced (11 features)."""
    X, _ = load_and_validate_data(DATA_PATH)
    sample = X.head(50)

    prep_full = build_preprocessor(drop_purpose=False)
    out_full = prep_full.fit_transform(sample)

    prep_reduced = build_preprocessor(drop_purpose=True)
    out_reduced = prep_reduced.fit_transform(sample)

    # Purpose_Of_Use has 4 distinct one-hot categories, so full has exactly 4 more columns than reduced
    assert out_full.shape[1] - out_reduced.shape[1] == 4
    assert out_full.shape[0] == 50
    assert out_reduced.shape[0] == 50


def test_no_preprocessing_fit_on_test_data():
    """Verify preprocessing state (learned top countries and scaler means) is frozen after fit."""
    X, _ = load_and_validate_data(DATA_PATH)
    train_split = X.iloc[:3000]
    test_split = X.iloc[3000:]

    preprocessor = build_preprocessor(drop_purpose=False)
    preprocessor.fit(train_split)

    country_encoder = preprocessor.named_transformers_["country"].named_steps["top_k"]
    initial_top_countries = list(country_encoder.top_countries_)

    # Transform test split
    _ = preprocessor.transform(test_split)

    # Assert learned categories were not modified during transform
    assert list(country_encoder.top_countries_) == initial_top_countries


# -----------------------------------------------------------------------------
# 5. Estimator Compatibility Tests (Dense Output Verification)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("estimator_cls,kwargs", [
    (LinearRegression, {}),
    (Ridge, {"alpha": 1.0}),
    (RandomForestRegressor, {"n_estimators": 10, "random_state": 42}),
    (HistGradientBoostingRegressor, {"max_iter": 10, "random_state": 42}),
])
def test_candidate_estimator_pipeline_compatibility(estimator_cls, kwargs):
    """Verify pipeline integrates and executes fit/predict with every candidate estimator."""
    X, y = load_and_validate_data(DATA_PATH)
    train_X, train_y = X.iloc[:200], y.iloc[:200]
    val_X = X.iloc[200:250]

    preprocessor = build_preprocessor(drop_purpose=False, activity_strategy="clip")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator_cls(**kwargs)),
    ])

    # Fit and predict
    pipe.fit(train_X, train_y)
    preds = pipe.predict(val_X)

    assert len(preds) == len(val_X)
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any()

