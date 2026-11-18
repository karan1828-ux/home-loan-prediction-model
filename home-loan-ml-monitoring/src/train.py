"""Model training pipeline for the Home Loan Prediction project.

This module provides reusable functions to train a DecisionTreeClassifier
with an isolated preprocessing pipeline, avoiding data leakage, construct
an end-to-end inference Pipeline, and execute the full training, validation,
and MLflow tracking workflow.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import get_features_and_target, load_training_data
from src.evaluate import evaluate_model, get_classification_report
from src.mlflow_tracking import build_pipeline, log_training_run
from src.preprocessing import fit_transform_features, transform_features


def train_model(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a DecisionTreeClassifier with an isolated preprocessing pipeline.

    Fits a fresh Scikit-learn ColumnTransformer on X_train only and trains a
    DecisionTreeClassifier on the transformed feature matrix.

    This function is completely decoupled from file loading and can be reused
    directly by future retraining and challenger-model pipelines.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training predictor features DataFrame (must contain the 11 expected features).
    y_train : pd.Series | np.ndarray
        Training target labels ('Y' / 'N').
    random_state : int, default=42
        Random seed for the DecisionTreeClassifier.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'model': Fitted DecisionTreeClassifier instance.
        - 'preprocessor': Fitted ColumnTransformer instance.

    Raises
    ------
    TypeError
        If X_train is not a pandas DataFrame.
    ValueError
        If X_train is empty or length of X_train does not match y_train.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for X_train, got {type(X_train).__name__}")
    if X_train.empty:
        raise ValueError("X_train cannot be empty.")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Sample size mismatch: X_train has {len(X_train)} samples, "
            f"but y_train has {len(y_train)} labels."
        )

    # 1. Fit preprocessing pipeline strictly on X_train (prevents data leakage)
    preprocessor, X_train_transformed = fit_transform_features(X_train)

    # 2. Train baseline DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train_transformed, y_train)

    return {
        "model": model,
        "preprocessor": preprocessor,
    }


def run_training(
    filepath: Optional[Union[str, Path]] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    enable_mlflow: bool = False,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "home-loan-experiments",
    model_name: Optional[str] = None,
    model_alias: Optional[str] = "champion",
) -> Dict[str, Any]:
    """Orchestrate the end-to-end training, evaluation, and optional MLflow tracking workflow.

    Workflow:
    1. Load the training dataset using load_training_data().
    2. Separate features (X) and target (y) using get_features_and_target().
    3. Perform stratified train-validation split (default 80/20).
    4. Fit preprocessing ONLY on X_train and train model using train_model().
    5. Construct an end-to-end Pipeline without refitting components.
    6. Transform X_validation using the fitted preprocessor (no refitting).
    7. Generate validation predictions and evaluate metrics.
    8. Optionally log experiment run, artifacts, and register champion model with MLflow.

    Parameters
    ----------
    filepath : str | Path | None, default=None
        Custom path to the training CSV file. If None, uses default data/raw/train.csv.
    test_size : float, default=0.20
        Proportion of dataset to reserve for validation.
    random_state : int, default=42
        Random seed for train_test_split and model initialization.
    enable_mlflow : bool, default=False
        Whether to log parameters, metrics, artifacts, and model to MLflow.
    tracking_uri : str | None, default=None
        Custom MLflow tracking URI. If None, uses default from configuration.
    experiment_name : str, default='home-loan-experiments'
        MLflow experiment name.
    model_name : str | None, default=None
        Registered model name. If None, defaults to MODEL_NAME from config.
    model_alias : str | None, default=None
        Model alias (e.g. 'champion'). If None, defaults to MODEL_ALIAS from config.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'model': Trained DecisionTreeClassifier.
        - 'preprocessor': Fitted ColumnTransformer.
        - 'pipeline': Fitted end-to-end Scikit-learn Pipeline.
        - 'metrics': Evaluation metrics dictionary (accuracy, precision, recall, f1, cm).
        - 'X_train': Training features DataFrame.
        - 'X_val': Validation features DataFrame.
        - 'y_train': Training target labels.
        - 'y_val': Validation target labels.
        - 'y_pred': Model predictions on validation data.
        - 'mlflow': Optional dict of MLflow run details (if enable_mlflow=True).
    """
    # 1. Ingest raw data
    df = load_training_data(filepath=filepath)

    # 2. Extract features and target (stripping Loan_ID)
    X, y = get_features_and_target(df)

    # 3. Stratified split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 4. Train model with isolated preprocessing on X_train only
    training_artifacts = train_model(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
    )
    model = training_artifacts["model"]
    preprocessor = training_artifacts["preprocessor"]

    # 5. Assemble end-to-end Pipeline without refitting
    pipeline = build_pipeline(preprocessor=preprocessor, model=model)

    # 6. Transform validation features with the pre-fitted preprocessor
    X_val_transformed = transform_features(preprocessor, X_val)

    # 7. Generate predictions on validation set
    y_pred = model.predict(X_val_transformed)

    # 8. Evaluate validation performance
    metrics = evaluate_model(y_true=y_val, y_pred=y_pred)

    results: Dict[str, Any] = {
        "model": model,
        "preprocessor": preprocessor,
        "pipeline": pipeline,
        "metrics": metrics,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "y_pred": y_pred,
    }

    # 9. Optionally log to MLflow tracking server and register model
    if enable_mlflow:
        params: Dict[str, Any] = {
            "random_state": random_state,
            "test_size": test_size,
            "model_type": type(model).__name__,
            "criterion": model.criterion,
            "max_depth": str(model.max_depth),
            "min_samples_split": model.min_samples_split,
        }
        report_text = get_classification_report(y_true=y_val, y_pred=y_pred)
        mlflow_info = log_training_run(
            pipeline=pipeline,
            metrics=metrics,
            params=params,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            model_name=model_name or "HomeLoanClassifier",
            model_alias=model_alias,
            register_model=True,
            classification_report_text=report_text,
        )
        results["mlflow"] = mlflow_info

    return results


if __name__ == "__main__":
    # Convenience execution for standalone verification
    results = run_training()
    print("Training finished successfully.")
    print("Validation Metrics:")
    for metric_name, val in results["metrics"].items():
        if metric_name != "confusion_matrix":
            print(f"  {metric_name}: {val:.4f}")
        else:
            print(f"  {metric_name}:\n{val}")
