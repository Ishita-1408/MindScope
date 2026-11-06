from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# Schema Definitions
EXPECTED_COLUMNS = [
    "Age",
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
    "Mental_Health_Score",
]

PREDICTOR_COLUMNS = [col for col in EXPECTED_COLUMNS if col != "Mental_Health_Score"]
TARGET_COLUMN = "Mental_Health_Score"


def load_and_validate_data(
    filepath: Union[str, Path] = "Student Social Media And Mental Health Impact.csv",
    exclude_negative_activity: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads dataset from disk, validates schema, removes exact duplicates,
    and returns separated predictors (X) and continuous target (y).

    Parameters:
    -----------
    filepath : str or Path
        Path to the raw CSV file. The original file remains untouched.
    exclude_negative_activity : bool, default=False
        If True (used for sensitivity analysis), excludes rows with Physical_Activity_Hours < 0.0.
        If False, retains all deduplicated rows for pipeline-level treatment.

    Returns:
    --------
    X : pd.DataFrame
        Predictor DataFrame containing the 12 features.
    y : pd.Series
        Continuous Mental_Health_Score target.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path)

    # Validate schema
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Remove exact duplicate records
    dedup_df = df.drop_duplicates().reset_index(drop=True)

    # Sensitivity option: exclude observations with negative physical activity
    if exclude_negative_activity:
        dedup_df = dedup_df[dedup_df["Physical_Activity_Hours"] >= 0.0].reset_index(drop=True)

    X = dedup_df[PREDICTOR_COLUMNS].copy()
    y = dedup_df[TARGET_COLUMN].copy()
    return X, y


class CountryTopKEncoder(BaseEstimator, TransformerMixin):
    """
    Leakage-free category grouping transformer for high-cardinality country values.
    Learns category frequencies exclusively during fit() on training data.
    Preserves existing 'Other' values and maps infrequent or unseen categories to `other_label`.

    Parameters:
    -----------
    top_k : int, default=10
        The number of top categories to retain.
    other_label : str, default='Other'
        The category label used for aggregating infrequent, unlearned, or existing 'Other' categories.
    include_other_in_top_k : bool, default=True
        Whether the existing `other_label` counts toward the top_k budget during fit().
        If True (matching baseline notebook behavior), 'Other' is counted as one of the top_k categories.
        If False, top_k represents k named countries excluding 'Other', and 'Other' is appended as the residual label.
    """

    def __init__(
        self,
        top_k: int = 10,
        other_label: str = "Other",
        include_other_in_top_k: bool = True,
    ):
        self.top_k = top_k
        self.other_label = other_label
        self.include_other_in_top_k = include_other_in_top_k
        self.top_countries_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0].dropna()
        elif isinstance(X, pd.Series):
            series = X.dropna()
        else:
            series = pd.Series(np.asarray(X).ravel()).dropna()

        if self.include_other_in_top_k:
            counts = series.value_counts()
            self.top_countries_ = counts.index[: self.top_k].tolist()
        else:
            counts = series[series != self.other_label].value_counts()
            self.top_countries_ = counts.index[: self.top_k].tolist()

        # Ensure other_label is present in recognized categories
        if self.other_label not in self.top_countries_:
            self.top_countries_.append(self.other_label)

        return self

    def transform(self, X):
        if self.top_countries_ is None:
            raise RuntimeError("CountryTopKEncoder has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            series = X
        else:
            series = pd.Series(np.asarray(X).ravel())

        top_set = set(self.top_countries_)
        # Preserve valid top countries; aggregate others and NaN into other_label
        transformed = series.apply(
            lambda val: val if pd.notna(val) and val in top_set else self.other_label
        )
        return transformed.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["Country_Grouped"], dtype=object)


class PhysicalActivityHandler(BaseEstimator, TransformerMixin):
    """
    Transformer supporting sensitivity experiments for negative Physical_Activity_Hours values:
    - 'clip': Clamps negative values to 0.0 (max(x, 0.0)).
    - 'nan': Replaces negative values with np.nan for fold-level median imputation.
    - 'passthrough': Leaves values unchanged (used when observations are excluded prior to pipeline).
    """

    def __init__(self, strategy: str = "clip"):
        if strategy not in ("clip", "nan", "passthrough"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. Must be 'clip', 'nan', or 'passthrough'."
            )
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float).copy()
        if self.strategy == "clip":
            arr = np.clip(arr, 0.0, None)
        elif self.strategy == "nan":
            arr[arr < 0.0] = np.nan
        return arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.array(["Physical_Activity_Hours_Handled"], dtype=object)


def build_preprocessor(
    drop_purpose: bool = False,
    activity_strategy: str = "clip",
    top_k_countries: int = 10,
    include_other_in_top_k: bool = True,
) -> ColumnTransformer:
    """
    Constructs an encapsulated, leakage-free Scikit-Learn ColumnTransformer.
    Ensures dense 2D float outputs for universal compatibility with HistGradientBoostingRegressor
    and all linear/ensemble models.

    Parameters:
    -----------
    drop_purpose : bool, default=False
        If True, excludes 'Purpose_Of_Use' to evaluate the reduced feature schema.
    activity_strategy : str, default='clip'
        Strategy for handling negative physical activity values ('clip', 'nan', or 'passthrough').
    top_k_countries : int, default=10
        Number of top countries retained by CountryTopKEncoder.
    include_other_in_top_k : bool, default=True
        Whether 'Other' is counted towards the top_k budget.
    """
    # 1. Country Branch: Top-K grouping -> Impute -> OneHot (dense)
    country_pipeline = Pipeline([
        ("top_k", CountryTopKEncoder(
            top_k=top_k_countries,
            other_label="Other",
            include_other_in_top_k=include_other_in_top_k,
        )),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # 2. Skewed Study Hours Branch: Impute -> Log1p -> StandardScaler
    study_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])

    # 3. Physical Activity Branch: ActivityHandler -> Impute -> StandardScaler
    physical_pipeline = Pipeline([
        ("handler", PhysicalActivityHandler(strategy=activity_strategy)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # 4. Standard Numeric Branch: Impute -> StandardScaler
    standard_numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # 5. Ordinal Stress Branch: Impute -> OrdinalEncoder
    ordinal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[["Low", "Medium", "High", "Very High"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # 6. Nominal Categorical Branch: Impute -> OneHot (dense)
    nominal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Column mappings
    skewed_col = ["Study_Hours"]
    physical_col = ["Physical_Activity_Hours"]
    standard_numeric_cols = [
        "Age",
        "Avg_Daily_Usage_Hours",
        "Daily_Unlocks",
        "Sleep_Hours_Per_Night",
    ]
    ordinal_col = ["Stress_Level"]
    country_col = ["Country"]

    if drop_purpose:
        nominal_cols = ["Gender", "Academic_Level", "Most_Used_Platform"]
    else:
        nominal_cols = ["Gender", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use"]

    transformers = [
        ("country", country_pipeline, country_col),
        ("skewed", study_pipeline, skewed_col),
        ("physical", physical_pipeline, physical_col),
        ("standard_numeric", standard_numeric_pipeline, standard_numeric_cols),
        ("ordinal", ordinal_pipeline, ordinal_col),
        ("nominal", nominal_pipeline, nominal_cols),
    ]

    return ColumnTransformer(transformers=transformers, remainder="drop")
