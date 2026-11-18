"""Automated model retraining and champion-challenger evaluation pipeline.

This module coordinates:
1. Verification of drift detection signals (with support for forced overrides).
2. Automated invocation of the reusable training pipeline in src/train.py as a 'candidate' model.
3. Loading of the existing champion model from the MLflow Model Registry via supported interfaces.
4. Champion-versus-candidate evaluation on identical labeled evaluation datasets.
5. Conditional promotion of the candidate only when strictly outperforming the champion.
6. Fail-safe handling: preserving existing champion if training, evaluation, loading, or alias assignment fails.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
)
from src.data_loader import get_features_and_target, load_training_data
from src.drift_detection import DriftReport
from src.evaluate import evaluate_model
from src.predict import predictor
from src.train import run_training

# Optional / safe MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

SUPPORTED_METRICS: List[str] = ["f1_score", "accuracy", "precision", "recall"]


def _build_payload(
    status: str,
    message: str,
    promoted: bool = False,
    drift_detected: Optional[bool] = None,
    drift_percentage: Optional[float] = None,
    candidate_version: Optional[str] = None,
    active_model_version: Optional[str] = None,
    candidate_metrics: Optional[Dict[str, Any]] = None,
    champion_metrics: Optional[Dict[str, Any]] = None,
    candidate_score: Optional[float] = None,
    champion_score: Optional[float] = None,
    metric_evaluated: str = "f1_score",
    evaluation_source: Optional[str] = None,
    comparison: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct the standardized retraining response dictionary.

    Ensures consistent payload structure across all success, skip, and error return paths.
    """
    return {
        "status": status,
        "drift_detected": drift_detected,
        "drift_percentage": drift_percentage,
        "promoted": promoted,
        "candidate_version": candidate_version,
        "active_model_version": active_model_version,
        "candidate_metrics": candidate_metrics,
        "champion_metrics": champion_metrics,
        "candidate_score": candidate_score,
        "champion_score": champion_score,
        "metric_evaluated": metric_evaluated,
        "evaluation_source": evaluation_source,
        "message": message,
        "comparison": comparison,
    }


def _validate_retraining_inputs(
    primary_metric: str,
    min_improvement: float,
    drift_report: Optional[Union[DriftReport, Dict[str, Any]]],
) -> None:
    """Validate primary evaluation metric, minimum improvement, and drift report type.

    Parameters
    ----------
    primary_metric : str
        Target metric for champion-candidate comparison. Must be in SUPPORTED_METRICS.
    min_improvement : float
        Minimum performance threshold improvement required for promotion.
    drift_report : DriftReport | dict | None
        Input drift report. Must be DriftReport, dict, or None.

    Raises
    ------
    TypeError
        If drift_report or min_improvement are of unsupported types.
    ValueError
        If primary_metric is unsupported or min_improvement is negative, NaN, or non-finite.
    """
    if drift_report is not None and not isinstance(drift_report, (DriftReport, dict)):
        raise TypeError(
            f"Unsupported drift_report type: {type(drift_report).__name__}. "
            "Must be an instance of DriftReport, dict, or None."
        )

    if primary_metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported primary_metric '{primary_metric}'. "
            f"Must be one of {SUPPORTED_METRICS}."
        )

    if isinstance(min_improvement, bool) or not isinstance(min_improvement, (int, float)):
        raise TypeError(
            f"min_improvement must be a non-negative real number, got {type(min_improvement).__name__}."
        )

    if math.isnan(min_improvement) or math.isinf(min_improvement) or min_improvement < 0.0:
        raise ValueError(
            f"min_improvement must be a non-negative finite real number, got {min_improvement}."
        )


def _load_champion_from_mlflow(
    client: Any,
    model_name: str,
    alias: str = "champion",
) -> Tuple[Optional[Any], Optional[str]]:
    """Load the existing champion model and version from MLflow Model Registry.

    Parameters
    ----------
    client : MlflowClient
        Active MLflow client instance.
    model_name : str
        Registered model name in MLflow.
    alias : str, default='champion'
        Alias to resolve.

    Returns
    -------
    tuple[Any | None, str | None]
        (champion_pipeline, champion_version_string) or (None, None) if no champion exists.

    Raises
    ------
    RuntimeError
        If MLflow reports an unexpected error (connection failure, artifact corruption, etc.)
        other than legitimate non-existence of the model or alias.
    """
    import mlflow.sklearn

    try:
        version_info = client.get_model_version_by_alias(model_name, alias)
        champion_version = str(version_info.version)
    except Exception as exc:
        exc_str = str(exc)
        # Check if the model or alias legitimately does not exist yet (first-time training)
        is_not_found = (
            "RESOURCE_DOES_NOT_EXIST" in exc_str
            or "not found" in exc_str.lower()
            or "does not exist" in exc_str.lower()
        )
        if is_not_found:
            logger.info(
                "No registered champion model found for '%s' (alias '%s').",
                model_name,
                alias,
            )
            return None, None

        # Network, connection, or unexpected server error
        logger.error(
            "Failed to query champion model alias '%s' for '%s': %s",
            alias,
            model_name,
            exc,
        )
        raise RuntimeError(
            f"Failed to query champion model from MLflow registry: {exc}"
        ) from exc

    champion_uri = f"models:/{model_name}@{alias}"
    try:
        champion_pipeline = mlflow.sklearn.load_model(champion_uri)
        logger.info(
            "Loaded existing champion model version '%s' from '%s'.",
            champion_version,
            champion_uri,
        )
        return champion_pipeline, champion_version
    except Exception as exc:
        logger.error(
            "Failed to load champion model artifact from '%s': %s",
            champion_uri,
            exc,
        )
        raise RuntimeError(
            f"Failed to load champion model from MLflow at '{champion_uri}': {exc}"
        ) from exc


def _orchestrate_retraining_impl(
    drift_report: Optional[Union[DriftReport, Dict[str, Any]]] = None,
    new_training_data_path: Optional[Union[str, Path]] = None,
    evaluation_data_path: Optional[Union[str, Path]] = None,
    primary_metric: str = "f1_score",
    min_improvement: float = 0.0,
    force_retrain: bool = False,
    enable_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    model_name: Optional[str] = None,
    experiment_name: Optional[str] = "home-loan-experiments",
) -> Dict[str, Any]:
    """Execute end-to-end retraining gatekeeper, candidate training, evaluation, and promotion.

    Workflow:
    1. Validate input parameters (primary_metric, min_improvement, and drift_report type).
    2. Gating: verify drift status or force_retrain override. Skip retraining if no drift
       or inconclusive.
    3. Fail-safe: ensure MLflow is available; do not promote offline.
    4. Query existing champion model from MLflow registry via supported interface.
    5. Train candidate model via run_training(model_alias="candidate").
    6. Evaluate both candidate and champion on the exact same labeled evaluation rows.
    7. Promote candidate if performance strictly exceeds champion + min_improvement.
       - Reassign 'champion' alias in MLflow.
       - Update serving predictor if supported public interface is available.
       - Accurately report partial failure if predictor reload fails after MLflow promotion.

    Parameters
    ----------
    drift_report : DriftReport | dict | None
        Evaluated drift report. Required unless force_retrain=True.
    new_training_data_path : str | Path | None
        Path to training data for the candidate model.
    evaluation_data_path : str | Path | None
        Path to dedicated labeled evaluation dataset. If None, falls back to validation split.
    primary_metric : str, default='f1_score'
        Metric used for promotion decision ('f1_score', 'accuracy', 'precision', 'recall').
    min_improvement : float, default=0.0
        Minimum threshold difference required for candidate to win promotion.
    force_retrain : bool, default=False
        If True, bypasses drift gating and executes retraining regardless of drift state.
    enable_mlflow : bool, default=True
        Whether MLflow tracking and model registry are enabled.
    tracking_uri : str | None
        Custom MLflow tracking URI. If None, uses default from config.
    model_name : str | None
        Registered model name. If None, uses MODEL_NAME from config.
    experiment_name : str | None
        MLflow experiment name.

    Returns
    -------
    dict[str, Any]
        Structured dictionary matching project specification:
        - status: 'promoted', 'partial_promotion_reload_failed', 'rejected_no_improvement',
                  'skipped_no_drift', 'skipped_inconclusive', or 'failed'
        - drift_detected: bool | None
        - drift_percentage: float | None
        - promoted: bool
        - candidate_version: str | None
        - active_model_version: str | None
        - candidate_metrics: dict | None
        - champion_metrics: dict | None
        - candidate_score: float | None
        - champion_score: float | None
        - metric_evaluated: str
        - evaluation_source: str | None
        - message: str
        - comparison: dict | None
    """
    # 1. Validate inputs
    _validate_retraining_inputs(primary_metric, min_improvement, drift_report)

    # 2. Extract drift details
    drift_detected: Optional[bool] = None
    drift_percentage: Optional[float] = None
    drift_status: Optional[str] = None

    if isinstance(drift_report, DriftReport):
        drift_detected = drift_report.is_drift_detected
        drift_percentage = drift_report.drift_percentage
        drift_status = drift_report.status
    elif isinstance(drift_report, dict):
        drift_detected = drift_report.get("is_drift_detected")
        drift_percentage = drift_report.get("drift_percentage")
        drift_status = drift_report.get("status")

    # 3. Gating logic
    if not force_retrain:
        if drift_report is None:
            msg = (
                "Missing drift report. Automated retraining is not triggered "
                "without confirmed drift unless force mode is enabled."
            )
            logger.info(msg)
            return _build_payload(
                status="skipped_no_drift",
                message=msg,
                metric_evaluated=primary_metric,
            )

        if drift_status == "INCONCLUSIVE":
            msg = (
                "Drift report status is INCONCLUSIVE (no valid feature tests). "
                "Automated retraining skipped to avoid retraining on unobservable data."
            )
            logger.info(msg)
            return _build_payload(
                status="skipped_inconclusive",
                message=msg,
                drift_detected=False,
                drift_percentage=drift_percentage,
                metric_evaluated=primary_metric,
            )

        if not drift_detected:
            msg = (
                f"Drift threshold not exceeded (status: {drift_status}, "
                f"score: {drift_percentage}%). Automated retraining skipped."
            )
            logger.info(msg)
            return _build_payload(
                status="skipped_no_drift",
                message=msg,
                drift_detected=False,
                drift_percentage=drift_percentage,
                metric_evaluated=primary_metric,
            )

    logger.info(
        "Proceeding with automated retraining (drift_detected=%s, force_retrain=%s)...",
        drift_detected,
        force_retrain,
    )

    # 4. Fail-safe MLflow validation: do not perform offline fallback promotions
    if not enable_mlflow:
        msg = (
            "MLflow must be enabled for champion-candidate retraining and conditional promotion. "
            "Offline promotion is disabled to ensure registry integrity."
        )
        logger.error(msg)
        return _build_payload(
            status="failed",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            metric_evaluated=primary_metric,
        )

    if not MLFLOW_AVAILABLE:
        msg = "MLflow dependency unavailable. Retraining cannot proceed."
        logger.error(msg)
        return _build_payload(
            status="failed",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            metric_evaluated=primary_metric,
        )

    active_tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
    target_model_name = model_name or MODEL_NAME
    mlflow.set_tracking_uri(active_tracking_uri)

    try:
        client = MlflowClient(tracking_uri=active_tracking_uri)
    except Exception as exc:
        msg = f"Failed to initialize MLflowClient at '{active_tracking_uri}': {exc}"
        logger.error(msg)
        return _build_payload(
            status="failed",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            metric_evaluated=primary_metric,
        )

    # 5. Load existing champion model from MLflow registry
    try:
        champion_pipeline, champion_version = _load_champion_from_mlflow(
            client=client,
            model_name=target_model_name,
            alias="champion",
        )
    except Exception as exc:
        msg = f"Champion model loading failed: {exc}. Preserving current registry state."
        logger.error(msg)
        return _build_payload(
            status="failed",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            metric_evaluated=primary_metric,
        )

    # 6. Train candidate model via run_training (assigning 'candidate' alias)
    try:
        logger.info(
            "Training candidate model with alias 'candidate' under model name '%s'...",
            target_model_name,
        )
        training_results = run_training(
            filepath=new_training_data_path,
            enable_mlflow=True,
            tracking_uri=active_tracking_uri,
            experiment_name=experiment_name or "home-loan-experiments",
            model_name=target_model_name,
            model_alias="candidate",
        )
    except Exception as exc:
        msg = f"Candidate model training failed: {exc}. Champion remains unchanged."
        logger.error(msg)
        return _build_payload(
            status="failed",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            active_model_version=champion_version,
            metric_evaluated=primary_metric,
        )

    candidate_pipeline = training_results.get("pipeline")
    mlflow_info = training_results.get("mlflow") or {}
    candidate_version = mlflow_info.get("model_version")

    # Validate candidate model version
    if candidate_pipeline is None or not candidate_version or not str(candidate_version).strip():
        msg = "Candidate model version missing or invalid from training output. Promotion aborted."
        logger.error(msg)
        return _build_payload(
            status="failed",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            active_model_version=champion_version,
            metric_evaluated=primary_metric,
        )

    # 7. Evaluate candidate and champion on the EXACT same labeled evaluation dataset
    evaluation_source: str
    try:
        if evaluation_data_path is not None:
            logger.info("Loading dedicated evaluation dataset from '%s'...", evaluation_data_path)
            eval_df = load_training_data(evaluation_data_path)
            if eval_df.empty:
                msg = f"Dedicated evaluation dataset at '{evaluation_data_path}' is empty."
                logger.error(msg)
                return _build_payload(
                    status="failed",
                    message=msg,
                    drift_detected=drift_detected,
                    drift_percentage=drift_percentage,
                    candidate_version=str(candidate_version),
                    active_model_version=champion_version,
                    metric_evaluated=primary_metric,
                )

            X_eval, y_eval = get_features_and_target(eval_df)
            if len(X_eval) == 0:
                msg = f"Dedicated evaluation dataset at '{evaluation_data_path}' contains zero samples."
                logger.error(msg)
                return _build_payload(
                    status="failed",
                    message=msg,
                    drift_detected=drift_detected,
                    drift_percentage=drift_percentage,
                    candidate_version=str(candidate_version),
                    active_model_version=champion_version,
                    metric_evaluated=primary_metric,
                )
            evaluation_source = "dedicated_file"
        else:
            logger.info("Using training validation split as evaluation dataset fallback.")
            X_eval = training_results["X_val"]
            y_eval = training_results["y_val"]
            evaluation_source = "validation_split_fallback"

        # Generate predictions for candidate
        cand_preds = candidate_pipeline.predict(X_eval)
        candidate_metrics = evaluate_model(y_true=y_eval, y_pred=cand_preds)
        cand_score = float(candidate_metrics[primary_metric])

        # Generate predictions for champion if it exists
        if champion_pipeline is not None:
            champ_preds = champion_pipeline.predict(X_eval)
            champion_metrics = evaluate_model(y_true=y_eval, y_pred=champ_preds)
            champ_score: Optional[float] = float(champion_metrics[primary_metric])
        else:
            champion_metrics = None
            champ_score = None
    except Exception as exc:
        msg = f"Model evaluation failed: {exc}. Champion remains unchanged."
        logger.error(msg)
        return _build_payload(
            status="failed",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            candidate_version=str(candidate_version),
            active_model_version=champion_version,
            metric_evaluated=primary_metric,
        )

    # 8. Champion vs Candidate Comparison
    diff: Optional[float] = None
    should_promote: bool = False
    decision_text: str = ""

    if champion_pipeline is None:
        # First-time training: candidate automatically qualifies as the first champion
        should_promote = True
        decision_text = "PROMOTE (First champion registration; no prior champion existed)"
    else:
        diff = round(cand_score - champ_score, 4)  # type: ignore[operator]
        should_promote = bool(diff > min_improvement)
        if should_promote:
            decision_text = (
                f"PROMOTE (Candidate {primary_metric} {cand_score:.4f} exceeded "
                f"champion {champ_score:.4f} by {diff:+.4f} > {min_improvement})"
            )
        else:
            decision_text = (
                f"REJECT (Candidate {primary_metric} {cand_score:.4f} did not exceed "
                f"champion {champ_score:.4f} by required min_improvement {min_improvement})"
            )

    comparison = {
        "primary_metric": primary_metric,
        "min_improvement": min_improvement,
        "candidate_score": cand_score,
        "champion_score": champ_score,
        "difference": diff,
        "decision": decision_text,
    }

    # 9. Handle Non-Promotion (Candidate rejected)
    if not should_promote:
        msg = (
            f"Candidate model version '{candidate_version}' ({primary_metric}: {cand_score:.4f}) "
            f"did not outperform champion ({primary_metric}: {champ_score:.4f}). "
            "Champion retained."
        )
        logger.info(msg)
        return _build_payload(
            status="rejected_no_improvement",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            promoted=False,
            candidate_version=str(candidate_version),
            active_model_version=champion_version,
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
            candidate_score=cand_score,
            champion_score=champ_score,
            metric_evaluated=primary_metric,
            evaluation_source=evaluation_source,
            comparison=comparison,
        )

    # 10. Handle Promotion
    # Step 10a: Update MLflow 'champion' alias to candidate_version
    try:
        logger.info(
            "Promoting candidate model version '%s' to alias 'champion' in MLflow...",
            candidate_version,
        )
        client.set_registered_model_alias(
            name=target_model_name,
            alias="champion",
            version=str(candidate_version),
        )
    except Exception as exc:
        msg = (
            f"Failed to assign 'champion' alias to candidate version '{candidate_version}' "
            f"in MLflow Model Registry: {exc}. Champion remains unchanged."
        )
        logger.error(msg)
        return _build_payload(
            status="failed",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            promoted=False,
            candidate_version=str(candidate_version),
            active_model_version=champion_version,
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
            candidate_score=cand_score,
            champion_score=champ_score,
            metric_evaluated=primary_metric,
            evaluation_source=evaluation_source,
            comparison=comparison,
        )

    # Step 10b: Update in-memory serving predictor using its supported public interface
    reload_failure_error: Optional[str] = None
    try:
        if hasattr(predictor, "set_model"):
            predictor.set_model(
                model=candidate_pipeline,
                model_name=target_model_name,
                model_alias="champion",
                model_version=str(candidate_version),
            )
            logger.info(
                "Serving predictor updated in-memory with new champion version '%s'.",
                candidate_version,
            )
        elif hasattr(predictor, "reload"):
            predictor.reload()
            logger.info("Serving predictor reloaded successfully.")
    except Exception as exc:
        reload_failure_error = str(exc)
        logger.warning(
            "Serving predictor reload failed after MLflow promotion: %s",
            exc,
        )

    # Note: MLflow promotion succeeded; if predictor reload failed, accurately report partial failure
    if reload_failure_error is not None:
        msg = (
            f"Candidate model version '{candidate_version}' was promoted to 'champion' in the "
            f"MLflow registry ({primary_metric}: {cand_score:.4f} vs {champ_score}), but serving "
            f"predictor reload failed: {reload_failure_error}. Trigger POST /reload-model to update serving."
        )
        logger.warning(msg)
        return _build_payload(
            status="partial_promotion_reload_failed",
            message=msg,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            promoted=True,
            candidate_version=str(candidate_version),
            active_model_version=str(candidate_version),
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
            candidate_score=cand_score,
            champion_score=champ_score,
            metric_evaluated=primary_metric,
            evaluation_source=evaluation_source,
            comparison=comparison,
        )

    champ_score_str = f"{champ_score:.4f}" if champ_score is not None else "None"
    msg = (
        f"Candidate model version '{candidate_version}' successfully promoted to champion "
        f"({primary_metric}: {cand_score:.4f} vs {champ_score_str})."
    )
    logger.info(msg)
    return _build_payload(
        status="promoted",
        message=msg,
        drift_detected=drift_detected,
        drift_percentage=drift_percentage,
        promoted=True,
        candidate_version=str(candidate_version),
        active_model_version=str(candidate_version),
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        candidate_score=cand_score,
        champion_score=champ_score,
        metric_evaluated=primary_metric,
        evaluation_source=evaluation_source,
        comparison=comparison,
    )


def _record_retraining_metrics(
    payload: Dict[str, Any], tracker: Optional[Any] = None
) -> None:
    """Record retraining outcome to metrics_tracker safely without raising exceptions."""
    try:
        if tracker is None:
            from monitoring.metrics import metrics_tracker as default_tracker

            active_tracker = default_tracker
        else:
            active_tracker = tracker

        status = payload.get("status", "failed")
        promoted = payload.get("promoted", False)
        # Only mark as successful promotion if status is strictly "promoted"
        is_successful_promotion = (status == "promoted") and bool(promoted)

        # Retraining was successful if it reached an expected terminal outcome
        # (promoted, rejected_no_improvement, skipped_no_drift, skipped_inconclusive)
        success = status in (
            "promoted",
            "rejected_no_improvement",
            "skipped_no_drift",
            "skipped_inconclusive",
        )

        champ_score = payload.get("champion_score")
        cand_score = payload.get("candidate_score")
        cand_version = (
            str(payload.get("candidate_version"))
            if is_successful_promotion and payload.get("candidate_version") is not None
            else None
        )
        metric_name = payload.get("metric_evaluated", "f1_score")

        details = {
            "status": status,
            "message": payload.get("message"),
            "evaluation_source": payload.get("evaluation_source"),
            "drift_detected": payload.get("drift_detected"),
            "drift_percentage": payload.get("drift_percentage"),
        }

        active_tracker.record_retraining_event(
            success=success,
            promoted=is_successful_promotion,
            champion_score=champ_score,
            candidate_score=cand_score,
            new_model_version=cand_version,
            evaluation_metric=metric_name,
            details=details,
        )
    except Exception as exc:
        logger.error("Failed to record retraining event to metrics store: %s", exc)


def orchestrate_retraining(
    drift_report: Optional[Union[DriftReport, Dict[str, Any]]] = None,
    new_training_data_path: Optional[Union[str, Path]] = None,
    evaluation_data_path: Optional[Union[str, Path]] = None,
    primary_metric: str = "f1_score",
    min_improvement: float = 0.0,
    force_retrain: bool = False,
    enable_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    model_name: Optional[str] = None,
    experiment_name: Optional[str] = "home-loan-experiments",
    record_metrics: bool = True,
    metrics_tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute end-to-end retraining gatekeeper, candidate training, evaluation, promotion, and metrics recording.

    Wraps the core retraining pipeline, executes champion promotion, and records the
    outcome to the central MetricsStore.
    """
    payload = _orchestrate_retraining_impl(
        drift_report=drift_report,
        new_training_data_path=new_training_data_path,
        evaluation_data_path=evaluation_data_path,
        primary_metric=primary_metric,
        min_improvement=min_improvement,
        force_retrain=force_retrain,
        enable_mlflow=enable_mlflow,
        tracking_uri=tracking_uri,
        model_name=model_name,
        experiment_name=experiment_name,
    )

    if record_metrics:
        _record_retraining_metrics(payload, tracker=metrics_tracker)

    return payload
