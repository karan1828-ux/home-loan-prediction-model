"""Prediction pipeline and champion model inference service.

This module is responsible for loading the registered champion model
from the MLflow Model Registry, preparing incoming inference payloads,
generating predictions, computing prediction probabilities (if supported),
and tracking serving performance metrics.
"""

from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_MODEL_URI,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
)
from src.data_loader import FEATURE_COLUMNS, ID_COLUMN
from monitoring.metrics import metrics_tracker

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when an inference request is made but no champion model is loaded."""


class ChampionModelPredictor:
    """Manages the champion model lifecycle and inference requests."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        model_name: Optional[str] = None,
        model_alias: Optional[str] = None,
        model_uri: Optional[str] = None,
    ) -> None:
        self.tracking_uri: str = tracking_uri or MLFLOW_TRACKING_URI
        self.model_name: str = model_name or MODEL_NAME
        self.model_alias: str = model_alias or MODEL_ALIAS
        self.model_uri: str = model_uri or MLFLOW_MODEL_URI

        self._model: Optional[Any] = None
        self._is_loaded: bool = False
        self._model_version: Optional[str] = None
        self._last_loaded_at: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is currently loaded in memory."""
        return self._is_loaded and self._model is not None

    @property
    def model_version(self) -> Optional[str]:
        """Return the version string of the loaded model."""
        return self._model_version

    @property
    def last_loaded_at(self) -> Optional[str]:
        """Return the ISO timestamp of the last successful model load."""
        return self._last_loaded_at

    def set_model(
        self,
        model: Any,
        model_name: Optional[str] = None,
        model_alias: Optional[str] = None,
        model_version: Optional[str] = "test-v1",
    ) -> None:
        """Explicitly set a model instance in memory (useful for testing and dependency injection)."""
        self._model = model
        self._is_loaded = True
        if model_name:
            self.model_name = model_name
        if model_alias:
            self.model_alias = model_alias
        self._model_version = model_version
        self._last_loaded_at = datetime.now(timezone.utc).isoformat()

    def unload_model(self) -> None:
        """Unload the active model from memory."""
        self._model = None
        self._is_loaded = False
        self._model_version = None
        self._last_loaded_at = None

    def load_model(self, force_reload: bool = False) -> Any:
        """Load the champion model registered in the MLflow Model Registry.

        Parameters
        ----------
        force_reload : bool, default=False
            Whether to force reload even if the model is already in memory.

        Returns
        -------
        Any
            The loaded MLflow model object.

        Raises
        ------
        ModelNotLoadedError
            If the model cannot be retrieved from MLflow.
        """
        if self.is_loaded and not force_reload:
            return self._model

        mlflow.set_tracking_uri(self.tracking_uri)

        # Candidate URI patterns: alias syntax (@champion), stage syntax (/champion), or direct URI
        candidate_uris = [
            self.model_uri,
            f"models:/{self.model_name}@{self.model_alias}",
            f"models:/{self.model_name}/{self.model_alias}",
        ]
        # Remove duplicates while preserving order
        candidate_uris = list(dict.fromkeys(candidate_uris))

        last_error: Optional[Exception] = None
        loaded_model = None

        for uri in candidate_uris:
            # 1. Try mlflow.sklearn first (preserves predict_proba directly)
            try:
                logger.info("Attempting to load model via mlflow.sklearn from '%s'...", uri)
                loaded_model = mlflow.sklearn.load_model(uri)
                logger.info("Successfully loaded sklearn model from '%s'", uri)
                break
            except Exception as e_sklearn:
                logger.debug("mlflow.sklearn failed for '%s': %s", uri, e_sklearn)

            # 2. Try mlflow.pyfunc as fallback
            try:
                logger.info("Attempting to load model via mlflow.pyfunc from '%s'...", uri)
                loaded_model = mlflow.pyfunc.load_model(uri)
                logger.info("Successfully loaded pyfunc model from '%s'", uri)
                break
            except Exception as e_pyfunc:
                logger.debug("mlflow.pyfunc failed for '%s': %s", uri, e_pyfunc)
                last_error = e_pyfunc

        if loaded_model is None:
            self._is_loaded = False
            self._model = None
            msg = (
                f"Failed to load champion model '{self.model_name}' (alias: '{self.model_alias}') "
                f"from MLflow tracking server at '{self.tracking_uri}'. Reason: {last_error}"
            )
            logger.error(msg)
            raise ModelNotLoadedError(msg)

        # Attempt to resolve the registered version number
        version_str: Optional[str] = None
        try:
            client = MlflowClient(tracking_uri=self.tracking_uri)
            try:
                model_version_info = client.get_model_version_by_alias(
                    self.model_name, self.model_alias
                )
                version_str = str(model_version_info.version)
            except Exception:
                pass
        except Exception:
            pass

        self._model = loaded_model
        self._is_loaded = True
        self._model_version = version_str or "active"
        self._last_loaded_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Champion model '%s' (alias: '%s', version: '%s') loaded successfully.",
            self.model_name,
            self.model_alias,
            self._model_version,
        )
        return self._model

    def _extract_probability(
        self, model: Any, features_df: pd.DataFrame
    ) -> Optional[float]:
        """Attempt to extract prediction probability for the positive class (approval).

        Parameters
        ----------
        model : Any
            The loaded model estimator or wrapper.
        features_df : pd.DataFrame
            Single-row DataFrame of predictor features.

        Returns
        -------
        float | None
            Probability of loan approval between 0.0 and 1.0, or None if unsupported.
        """
        estimator = model

        # If wrapped inside MLflow PyFuncModel, attempt to access the underlying estimator
        if hasattr(model, "unwrap_python_model"):
            try:
                unwrapped = model.unwrap_python_model()
                if hasattr(unwrapped, "predict_proba"):
                    estimator = unwrapped
                elif hasattr(unwrapped, "_model_impl") and hasattr(
                    unwrapped._model_impl, "predict_proba"
                ):
                    estimator = unwrapped._model_impl
            except Exception:
                pass
        elif hasattr(model, "_model_impl") and hasattr(model._model_impl, "predict_proba"):
            estimator = model._model_impl

        if not hasattr(estimator, "predict_proba"):
            return None

        try:
            proba_matrix = estimator.predict_proba(features_df)
            if proba_matrix is None or len(proba_matrix) == 0:
                return None

            row_proba = proba_matrix[0]

            # Identify positive class index ('Y' or 1)
            pos_idx = 1 if len(row_proba) > 1 else 0
            if hasattr(estimator, "classes_"):
                classes = list(estimator.classes_)
                if "Y" in classes:
                    pos_idx = classes.index("Y")
                elif 1 in classes:
                    pos_idx = classes.index(1)
                elif "1" in classes:
                    pos_idx = classes.index("1")

            return round(float(row_proba[pos_idx]), 4)
        except Exception as err:
            logger.debug("Failed to calculate predict_proba: %s", err)
            return None

    def _prepare_features(
        self, application_data: Union[Dict[str, Any], pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Validate and format input into a DataFrame matching FEATURE_COLUMNS."""
        if hasattr(application_data, "to_feature_dict"):
            features_dict = application_data.to_feature_dict()
            loan_id = getattr(application_data, "Loan_ID", None)
            df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)
            return df, loan_id

        if isinstance(application_data, dict):
            loan_id = application_data.get("Loan_ID") or application_data.get("loan_id")
            # Map case-insensitively if needed
            row = {}
            for col in FEATURE_COLUMNS:
                if col in application_data:
                    row[col] = application_data[col]
                elif col.lower() in application_data:
                    row[col] = application_data[col.lower()]
                else:
                    row[col] = None
            df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
            return df, loan_id

        if isinstance(application_data, pd.DataFrame):
            loan_id = None
            if ID_COLUMN in application_data.columns:
                loan_id = str(application_data[ID_COLUMN].iloc[0])
            missing = [c for c in FEATURE_COLUMNS if c not in application_data.columns]
            if missing:
                raise ValueError(f"Input DataFrame is missing required features: {missing}")
            return application_data[FEATURE_COLUMNS].copy(), loan_id

        raise TypeError(
            f"Unsupported application data type: {type(application_data).__name__}"
        )

    def predict(
        self, application_data: Union[Dict[str, Any], pd.DataFrame]
    ) -> Dict[str, Any]:
        """Perform real-time prediction for a single loan application.

        Parameters
        ----------
        application_data : dict | pd.DataFrame | LoanApplication
            Applicant attributes to score.

        Returns
        -------
        dict[str, Any]
            Dictionary containing predicted loan_status ('Y'/'N'), raw prediction,
            approval probability, and model metadata.

        Raises
        ------
        ModelNotLoadedError
            If no champion model is loaded in memory.
        """
        start_time = time.perf_counter()

        if not self.is_loaded:
            # Attempt lazy loading
            self.load_model()

        features_df, loan_id = self._prepare_features(application_data)

        try:
            preds = self._model.predict(features_df)
            if hasattr(preds, "tolist"):
                preds = preds.tolist()

            raw_pred = preds[0] if isinstance(preds, (list, tuple, np.ndarray)) else preds

            # Map raw output to standardized 'Y' or 'N'
            if raw_pred in (1, 1.0, "1", "Y", "y", "Approved", True):
                loan_status = "Y"
            elif raw_pred in (0, 0.0, "0", "N", "n", "Rejected", False):
                loan_status = "N"
            else:
                loan_status = str(raw_pred)

            probability = self._extract_probability(self._model, features_df)

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            metrics_tracker.record_prediction(
                loan_status=loan_status,
                latency_ms=latency_ms,
                success=True,
            )

            return {
                "loan_id": loan_id,
                "loan_status": loan_status,
                "prediction": raw_pred,
                "probability": probability,
                "model_name": self.model_name,
                "model_alias": self.model_alias,
                "model_version": self._model_version,
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            metrics_tracker.record_prediction(
                loan_status=None,
                latency_ms=latency_ms,
                success=False,
            )
            logger.error("Inference execution failed: %s", exc)
            raise

    def reload(self) -> Dict[str, Any]:
        """Force reload the champion model from MLflow registry."""
        self.load_model(force_reload=True)
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Return the current model status and metadata."""
        return {
            "model_loaded": self.is_loaded,
            "model_name": self.model_name,
            "model_alias": self.model_alias,
            "model_version": self._model_version,
            "last_loaded_at": self._last_loaded_at,
        }


# Global predictor singleton
predictor = ChampionModelPredictor()


def predict_loan(
    application_data: Union[Dict[str, Any], pd.DataFrame]
) -> Dict[str, Any]:
    """Module-level convenience wrapper for model prediction."""
    return predictor.predict(application_data)


def reload_champion_model() -> Dict[str, Any]:
    """Module-level convenience wrapper to reload champion model."""
    return predictor.reload()


def get_model_status() -> Dict[str, Any]:
    """Module-level convenience wrapper to inspect model status."""
    return predictor.get_status()
