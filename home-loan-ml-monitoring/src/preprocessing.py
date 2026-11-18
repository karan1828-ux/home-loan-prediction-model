"""Data preprocessing pipeline for the Home Loan Prediction project.

This module provides Scikit-learn ColumnTransformer pipelines to impute
missing numerical and categorical features, encode categorical variables,
and prepare model-ready feature matrices for tree-based classifiers.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data_loader import ID_COLUMN, TARGET_COLUMN

# Try importing Scikit-learn components
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.exceptions import NotFittedError
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.utils.validation import check_is_fitted

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False
    ColumnTransformer = None  # type: ignore
    NotFittedError = Exception  # type: ignore
    SimpleImputer = None  # type: ignore
    Pipeline = None  # type: ignore
    OneHotEncoder = None  # type: ignore
    check_is_fitted = None  # type: ignore

# Numerical feature configuration (5 features)
NUMERICAL_FEATURES: List[str] = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Categorical feature configuration (6 features)
# Dependents is treated as categorical because it contains values such as '3+'
CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

# All 11 expected predictor features in consistent order
ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def validate_and_clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Validate that required feature columns exist and return a clean copy.

    Ensures that 'Loan_ID' and 'Loan_Status' are strictly excluded, all
    required 11 feature columns are present, and feature ordering is preserved.
    The original input DataFrame is never modified.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame containing application records.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing only the 11 feature columns in consistent order.

    Raises
    ------
    TypeError
        If X is not a pandas DataFrame.
    ValueError
        If X is empty or missing any required feature columns.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(X).__name__}")
    if X.empty:
        raise ValueError("Input feature DataFrame is empty.")

    missing_cols = [col for col in ALL_FEATURES if col not in X.columns]
    if missing_cols:
        raise ValueError(
            f"Input DataFrame is missing required feature column(s): {missing_cols}"
        )

    # Return a copy strictly restricted to the 11 feature columns
    return X[ALL_FEATURES].copy()


def create_preprocessor() -> "ColumnTransformer":
    """Create and return an unfitted ColumnTransformer preprocessor.

    Numerical pipeline:
        - SimpleImputer(strategy="median")

    Categorical pipeline:
        - SimpleImputer(strategy="most_frequent")
        - OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    Note:
        No feature scaling is applied, as tree-based baseline models (such as
        Decision Trees or Random Forests) are invariant to monotonic feature scaling.

    Returns
    -------
    ColumnTransformer
        An unfitted ColumnTransformer instance.

    Raises
    ------
    ImportError
        If scikit-learn is not installed in the environment.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required to create preprocessor. "
            "Please install dependencies with: pip install -r requirements.txt"
        )

    # Numerical pipeline: Median imputation for continuous & numerical features
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical pipeline: Most frequent imputation + OneHotEncoder with handle_unknown='ignore'
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        # Compatibility fallback for older scikit-learn versions
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def fit_preprocessor(X_train: pd.DataFrame) -> "ColumnTransformer":
    """Create a fresh ColumnTransformer and fit it only on X_train.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features DataFrame.

    Returns
    -------
    ColumnTransformer
        A fitted ColumnTransformer preprocessor.

    Raises
    ------
    TypeError
        If X_train is not a pandas DataFrame.
    ValueError
        If X_train is empty or missing required columns.
    ImportError
        If scikit-learn is not installed.
    """
    X_clean = validate_and_clean_features(X_train)
    preprocessor = create_preprocessor()
    preprocessor.fit(X_clean)
    return preprocessor


def transform_features(
    preprocessor: "ColumnTransformer", X: pd.DataFrame
) -> np.ndarray:
    """Transform input features using an already-fitted preprocessor.

    Does not refit the preprocessor. Handles unseen categories and missing values.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        An already-fitted ColumnTransformer instance.
    X : pd.DataFrame
        Input features DataFrame to transform.

    Returns
    -------
    np.ndarray
        Transformed 2D numpy array of imputed numerical and one-hot encoded features.

    Raises
    ------
    TypeError
        If X is not a pandas DataFrame or preprocessor is invalid.
    ValueError
        If preprocessor is unfitted or X is missing required columns.
    ImportError
        If scikit-learn is not installed.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required to transform features. "
            "Please install dependencies with: pip install -r requirements.txt"
        )
    if preprocessor is None:
        raise TypeError("Expected preprocessor instance, got None")

    # Verify that the preprocessor is fitted
    if check_is_fitted is not None:
        try:
            check_is_fitted(preprocessor)
        except NotFittedError as exc:
            raise ValueError(
                "Preprocessor must be fitted before transforming features."
            ) from exc
    elif not hasattr(preprocessor, "transformers_"):
        raise ValueError("Preprocessor must be fitted before transforming features.")

    X_clean = validate_and_clean_features(X)
    transformed = preprocessor.transform(X_clean)

    # Convert sparse matrix to dense array if applicable
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return np.asarray(transformed)


def fit_transform_features(
    X_train: pd.DataFrame,
) -> Tuple["ColumnTransformer", np.ndarray]:
    """Fit a new preprocessor on X_train and return both preprocessor and transformed features.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features DataFrame.

    Returns
    -------
    tuple[ColumnTransformer, np.ndarray]
        Tuple of (fitted_preprocessor, transformed_training_features).

    Raises
    ------
    TypeError
        If X_train is not a pandas DataFrame.
    ValueError
        If X_train is empty or missing required columns.
    ImportError
        If scikit-learn is not installed.
    """
    preprocessor = fit_preprocessor(X_train)
    transformed = transform_features(preprocessor, X_train)
    return preprocessor, transformed


def get_transformed_feature_names(
    preprocessor: "ColumnTransformer",
) -> List[str]:
    """Retrieve output feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer instance.

    Returns
    -------
    list[str]
        List of feature column names output by the preprocessor.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required. Please install dependencies with: pip install -r requirements.txt"
        )
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())

    # Fallback name construction
    names: List[str] = list(NUMERICAL_FEATURES)
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat" and hasattr(trans.named_steps.get("encoder"), "get_feature_names_out"):
            names.extend(list(trans.named_steps["encoder"].get_feature_names_out(cols)))
    return names
