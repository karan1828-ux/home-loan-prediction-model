"""Data loading and validation module for the Home Loan Prediction project.

This module provides reusable functions to load raw datasets (training and
prediction), validate their schemas, check for missing values and duplicates,
and cleanly separate model input features from target labels.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd

# Define project root relative to this module file (src/data_loader.py -> project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Default dataset locations relative to project root
DEFAULT_TRAIN_PATH: Path = PROJECT_ROOT / "data" / "raw" / "train.csv"
DEFAULT_PREDICTION_PATH: Path = PROJECT_ROOT / "data" / "raw" / "test.csv"

# Identifier and Target column definitions
ID_COLUMN: str = "Loan_ID"
TARGET_COLUMN: str = "Loan_Status"

# Expected 11 predictor features
FEATURE_COLUMNS: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

# Expected 12 columns for prediction data (Loan_ID + 11 features, no target)
RAW_PREDICTION_COLUMNS: List[str] = [ID_COLUMN] + FEATURE_COLUMNS

# Expected 13 columns for training data (Loan_ID + 11 features + Loan_Status)
RAW_TRAINING_COLUMNS: List[str] = RAW_PREDICTION_COLUMNS + [TARGET_COLUMN]


def _resolve_dataset_path(
    filepath: Optional[Union[str, Path]], default_path: Path
) -> Path:
    """Resolve a dataset path relative to the project root.

    Parameters
    ----------
    filepath : str | Path | None
        Supplied path to the dataset file, or None to use default.
    default_path : Path
        Default path resolved relative to PROJECT_ROOT.

    Returns
    -------
    Path
        Absolute or resolved Path to the target file.
    """
    if filepath is None:
        return default_path

    path_obj = Path(filepath)
    if path_obj.is_absolute():
        return path_obj

    # If relative, check project root first
    project_relative = (PROJECT_ROOT / path_obj).resolve()
    if project_relative.exists():
        return project_relative

    return path_obj.resolve()


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Calculate the number of missing (null/NaN) values per column.

    This function only inspects the DataFrame and does not alter, fill,
    or drop any records.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.

    Returns
    -------
    dict[str, int]
        Dictionary mapping column names to missing value counts.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    return {str(col): int(count) for col, count in df.isna().sum().items()}


def check_duplicate_rows(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> int:
    """Count duplicate rows in the DataFrame without removing them.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    subset : list[str] | None, default=None
        Columns to consider for duplicate detection. If None, considers all columns.

    Returns
    -------
    int
        Count of duplicate rows detected.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    return int(df.duplicated(subset=subset).sum())


def validate_training_data(df: pd.DataFrame) -> bool:
    """Validate that the training DataFrame matches the expected schema.

    Ensures the DataFrame is non-empty and contains all 13 required columns
    including the target column ('Loan_Status').

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame to validate.

    Returns
    -------
    bool
        True if validation succeeds.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty or missing required training columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Training dataset is empty.")

    missing_cols = [col for col in RAW_TRAINING_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Training dataset is missing required column(s): {missing_cols}"
        )

    return True


def validate_prediction_data(df: pd.DataFrame) -> bool:
    """Validate that the prediction DataFrame matches the expected schema.

    Ensures the DataFrame is non-empty and contains all 12 required prediction
    columns. Does NOT require the 'Loan_Status' target column. If 'Loan_Status'
    is present, a warning is issued to prevent treating test data as labeled data.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction DataFrame to validate.

    Returns
    -------
    bool
        True if validation succeeds.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty or missing required prediction columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Prediction dataset is empty.")

    missing_cols = [col for col in RAW_PREDICTION_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Prediction dataset is missing required column(s): {missing_cols}"
        )

    if TARGET_COLUMN in df.columns:
        warnings.warn(
            f"Prediction dataset contains target column '{TARGET_COLUMN}'. "
            "Prediction data should be unlabeled and not used for model evaluation.",
            UserWarning,
            stacklevel=2,
        )

    return True


def load_training_data(
    filepath: Optional[Union[str, Path]] = None, validate: bool = True
) -> pd.DataFrame:
    """Load the home loan training dataset from CSV.

    Parameters
    ----------
    filepath : str | Path | None, default=None
        Path to the training CSV file. If None, defaults to data/raw/train.csv
        resolved relative to the project root.
    validate : bool, default=True
        Whether to validate expected columns upon loading.

    Returns
    -------
    pd.DataFrame
        Loaded training dataset (614 rows expected in full dataset).

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the resolved path.
    ValueError
        If reading fails or validation detects schema issues.
    """
    resolved_path = _resolve_dataset_path(filepath, DEFAULT_TRAIN_PATH)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Training dataset file not found at: '{resolved_path}'. "
            "Please ensure 'train.csv' is placed in 'data/raw/'."
        )

    try:
        df = pd.read_csv(resolved_path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at '{resolved_path}': {exc}") from exc

    if validate:
        validate_training_data(df)

    return df


def load_prediction_data(
    filepath: Optional[Union[str, Path]] = None, validate: bool = True
) -> pd.DataFrame:
    """Load the home loan prediction dataset from CSV.

    Parameters
    ----------
    filepath : str | Path | None, default=None
        Path to the prediction CSV file. If None, defaults to data/raw/test.csv
        resolved relative to the project root.
    validate : bool, default=True
        Whether to validate expected columns upon loading.

    Returns
    -------
    pd.DataFrame
        Loaded prediction dataset (367 rows expected in full dataset).

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the resolved path.
    ValueError
        If reading fails or validation detects schema issues.
    """
    resolved_path = _resolve_dataset_path(filepath, DEFAULT_PREDICTION_PATH)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Prediction dataset file not found at: '{resolved_path}'. "
            "Please ensure 'test.csv' is placed in 'data/raw/'."
        )

    try:
        df = pd.read_csv(resolved_path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at '{resolved_path}': {exc}") from exc

    if validate:
        validate_prediction_data(df)

    return df


def get_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
    drop_id: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features and target from a training DataFrame.

    Excludes the target column and optionally the ID column ('Loan_ID')
    from the returned features DataFrame. The original input DataFrame
    is not modified.

    Parameters
    ----------
    df : pd.DataFrame
        Input training DataFrame.
    target_column : str, default='Loan_Status'
        Name of the target column to extract.
    id_column : str, default='Loan_ID'
        Name of the identifier column to exclude from model features.
    drop_id : bool, default=True
        Whether to exclude the identifier column from the returned features.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A tuple of (features_df, target_series).

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    ValueError
        If target_column is not present in the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            "Ensure you are passing labeled training data."
        )

    # Extract target as a copy to prevent accidental mutation
    target = df[target_column].copy()

    # Drop target and identifier from model input features
    cols_to_drop = [target_column]
    if drop_id and id_column in df.columns:
        cols_to_drop.append(id_column)

    features = df.drop(columns=cols_to_drop).copy()
    return features, target


def get_model_features(
    df: pd.DataFrame,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Extract model input features by dropping ID and target columns if present.

    Useful for preparing features from either training or prediction DataFrames
    without modifying the original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_column : str, default='Loan_ID'
        Identifier column to exclude.
    target_column : str, default='Loan_Status'
        Target column to exclude if present.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only predictor feature columns.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    cols_to_drop = [col for col in [id_column, target_column] if col in df.columns]
    return df.drop(columns=cols_to_drop).copy()
