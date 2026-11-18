"""MLflow experiment tracking and model registry integration module.

This module provides functions to assemble an end-to-end Scikit-learn Pipeline
from pre-fitted components, log experiment runs (parameters, metrics, artifacts),
and register the champion model in the MLflow Model Registry.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)

# Default experiment name
DEFAULT_EXPERIMENT_NAME: str = "home-loan-experiments"

# Try importing MLflow components gracefully
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    MLFLOW_AVAILABLE = False


def build_pipeline(
    preprocessor: ColumnTransformer,
    model: DecisionTreeClassifier,
) -> Pipeline:
    """Combine an already-fitted preprocessor and classifier into an end-to-end Pipeline.

    Does NOT refit the preprocessor or retrain the classifier.
    The resulting Pipeline accepts raw feature DataFrames containing the 11
    predictor feature columns and supports both `predict()` and `predict_proba()`.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted Scikit-learn ColumnTransformer instance.
    model : DecisionTreeClassifier
        Fitted Scikit-learn DecisionTreeClassifier instance.

    Returns
    -------
    Pipeline
        Fitted Scikit-learn Pipeline object ready for raw feature inference.

    Raises
    ------
    TypeError
        If preprocessor or model are not valid Scikit-learn estimators.
    ValueError
        If either component is not yet fitted.
    """
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"Expected ColumnTransformer for preprocessor, got {type(preprocessor).__name__}"
        )
    if not isinstance(model, DecisionTreeClassifier):
        raise TypeError(
            f"Expected DecisionTreeClassifier for model, got {type(model).__name__}"
        )

    # Verify both components are already fitted
    try:
        check_is_fitted(preprocessor)
    except Exception as exc:
        raise ValueError(
            "Preprocessor must be fitted before building Pipeline."
        ) from exc

    try:
        check_is_fitted(model)
    except Exception as exc:
        raise ValueError(
            "Classifier model must be fitted before building Pipeline."
        ) from exc

    # Construct pipeline with pre-fitted steps without refitting
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def log_training_run(
    pipeline: Pipeline,
    metrics: Dict[str, Any],
    params: Dict[str, Any],
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    model_name: str = MODEL_NAME,
    model_alias: Optional[str] = MODEL_ALIAS,
    register_model: bool = True,
    classification_report_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Log an experiment run to MLflow and optionally register the champion model.

    Logs:
    - Training parameters (e.g. random_state, test_size, model hyperparameters).
    - Validation metrics (accuracy, precision, recall, f1_score).
    - Confusion matrix as a JSON artifact.
    - Classification report as a text artifact (if provided).
    - Fitted end-to-end Pipeline.
    - Registers model under `model_name` and sets the `model_alias` (e.g. 'champion').

    Parameters
    ----------
    pipeline : Pipeline
        Fitted end-to-end Scikit-learn Pipeline.
    metrics : dict[str, Any]
        Dictionary of validation metrics from evaluate_model().
    params : dict[str, Any]
        Dictionary of parameters to log (random_state, test_size, etc.).
    tracking_uri : str | None, default=None
        MLflow tracking server URI. If None, uses MLFLOW_TRACKING_URI.
    experiment_name : str, default='home-loan-experiments'
        Name of the MLflow experiment.
    model_name : str, default='HomeLoanClassifier'
        Registered model name.
    model_alias : str, default='champion'
        Alias to assign to the registered model version.
    register_model : bool, default=True
        Whether to register the model in the MLflow Model Registry.
    classification_report_text : str | None, default=None
        Optional text string containing the classification report.

    Returns
    -------
    dict[str, Any]
        Dictionary containing run metadata:
        - 'run_id': Active MLflow run ID.
        - 'experiment_id': MLflow experiment ID.
        - 'model_uri': URI of the logged model.
        - 'model_name': Registered model name.
        - 'model_version': Registered version number (if registered).
        - 'model_alias': Assigned alias (if registered).

    Raises
    ------
    ImportError
        If mlflow is not installed in the environment.
    """
    if not MLFLOW_AVAILABLE:
        raise ImportError(
            "mlflow is required for experiment tracking. "
            "Please install dependencies with: pip install mlflow"
        )

    active_tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(active_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        # 1. Log training and model parameters
        flat_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                flat_params[k] = v
            else:
                flat_params[k] = str(v)
        mlflow.log_params(flat_params)

        # 2. Log validation metrics (accuracy, precision, recall, f1_score)
        scalar_metrics = {}
        for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
            if metric_name in metrics:
                scalar_metrics[metric_name] = float(metrics[metric_name])
        mlflow.log_metrics(scalar_metrics)

        # 3. Log confusion matrix artifact
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            cm_list = cm.tolist() if hasattr(cm, "tolist") else cm
            cm_payload = {
                "confusion_matrix": cm_list,
                "labels": ["N", "Y"],
            }
            try:
                mlflow.log_dict(cm_payload, "metrics/confusion_matrix.json")
            except Exception as e_dict:
                # Fallback if log_dict is unavailable
                mlflow.log_text(json.dumps(cm_payload, indent=2), "metrics/confusion_matrix.json")

        # 4. Log classification report artifact if provided
        if classification_report_text:
            mlflow.log_text(classification_report_text, "metrics/classification_report.txt")

        # 5. Log end-to-end Pipeline model
        registered_name = model_name if register_model else None
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=registered_name,
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

        model_version: Optional[str] = None
        # 6. Assign champion alias if registered
        if register_model:
            client = MlflowClient(tracking_uri=active_tracking_uri)

            # Determine registered version
            if hasattr(model_info, "registered_model_version") and model_info.registered_model_version is not None:
                model_version = str(model_info.registered_model_version)
            else:
                try:
                    latest = client.get_latest_versions(model_name)
                    if latest:
                        model_version = str(latest[-1].version)
                except Exception as exc:
                    logger.warning("Could not automatically retrieve latest model version: %s", exc)

            if model_version and model_alias:
                try:
                    client.set_registered_model_alias(
                        name=model_name,
                        alias=model_alias,
                        version=model_version,
                    )
                    logger.info(
                        "Assigned alias '%s' to model '%s' version '%s'",
                        model_alias,
                        model_name,
                        model_version,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not set model alias '%s' for '%s': %s",
                        model_alias,
                        model_name,
                        exc,
                    )

        return {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "model_uri": model_info.model_uri if hasattr(model_info, "model_uri") else f"runs:/{run_id}/model",
            "model_name": model_name if register_model else None,
            "model_version": model_version,
            "model_alias": model_alias if register_model else None,
        }

