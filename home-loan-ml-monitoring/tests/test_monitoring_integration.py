"""Integration tests for statistical drift detection and automated retraining monitoring.

Verifies that drift detection and retraining orchestration record accurate, non-duplicated
lifecycle telemetry into the central MetricsStore, handle edge cases (inconclusive evaluations,
partial promotions, persistence failures), and never fabricate monitoring scores.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from monitoring.metrics import MetricsStore
from src.drift_detection import (
    DriftReport,
    FeatureDriftResult,
    detect_drift,
    run_drift_detection,
)
from src.preprocessing import ALL_FEATURES
from src.retrain import orchestrate_retraining


class DummyPipeline:
    """Mock Scikit-learn Pipeline with predictable predictions."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions


@pytest.fixture
def temp_metrics_store(tmp_path):
    """Provide an isolated, file-backed MetricsStore instance in a temporary directory."""
    metrics_file = tmp_path / "reports" / "metrics.json"
    store = MetricsStore(filepath=metrics_file, auto_save=True)
    return store


@pytest.fixture
def sample_drift_data():
    """Create sample reference and production DataFrames with 11 features."""
    np.random.seed(42)
    n = 60
    ref_data = {
        "ApplicantIncome": np.random.normal(5000, 1000, n),
        "CoapplicantIncome": np.random.normal(1500, 500, n),
        "LoanAmount": np.random.normal(150, 30, n),
        "Loan_Amount_Term": np.array([360.0] * n),
        "Credit_History": np.array([1.0] * n),
        "Gender": ["Male"] * (n - 10) + ["Female"] * 10,
        "Married": ["Yes"] * (n - 15) + ["No"] * 15,
        "Dependents": ["0"] * 30 + ["1"] * 20 + ["2"] * 10,
        "Education": ["Graduate"] * (n - 10) + ["Not Graduate"] * 10,
        "Self_Employed": ["No"] * (n - 8) + ["Yes"] * 8,
        "Property_Area": ["Urban"] * 20 + ["Semiurban"] * 20 + ["Rural"] * 20,
    }
    prod_data = dict(ref_data)
    # Drift ApplicantIncome and Property_Area
    prod_data["ApplicantIncome"] = np.random.normal(12000, 3000, n)
    prod_data["Property_Area"] = ["Rural"] * 40 + ["Urban"] * 10 + ["Semiurban"] * 10

    ref_df = pd.DataFrame(ref_data)
    prod_df = pd.DataFrame(prod_data)
    return ref_df, prod_df


@pytest.fixture
def sample_features_and_target():
    """Provide minimal feature DataFrame and target Series for retraining tests."""
    X = pd.DataFrame(
        {
            "ApplicantIncome": [5000.0, 6000.0, 3000.0, 8000.0],
            "CoapplicantIncome": [0.0, 1500.0, 2000.0, 0.0],
            "LoanAmount": [120.0, 150.0, 100.0, 200.0],
            "Loan_Amount_Term": [360.0, 360.0, 360.0, 180.0],
            "Credit_History": [1.0, 1.0, 0.0, 1.0],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Married": ["Yes", "No", "Yes", "No"],
            "Dependents": ["0", "1", "2", "0"],
            "Education": ["Graduate", "Not Graduate", "Graduate", "Graduate"],
            "Self_Employed": ["No", "Yes", "No", "No"],
            "Property_Area": ["Urban", "Rural", "Semiurban", "Urban"],
        }
    )
    y = pd.Series(["Y", "N", "N", "Y"])
    return X, y


def test_completed_drift_evaluation_records_one_event(temp_metrics_store, sample_drift_data):
    """Verify that a completed drift evaluation records exactly one event with correct metrics."""
    ref_df, prod_df = sample_drift_data
    store = temp_metrics_store

    report = run_drift_detection(
        reference_data=ref_df,
        production_data=prod_df,
        metrics_tracker=store,
        record_metrics=True,
    )

    assert isinstance(report, DriftReport)
    assert store.drift_events_count == 1
    assert store.last_drift_score == report.drift_percentage
    assert store.last_drift_status == report.status
    assert set(store.last_drifted_features) == set(report.drifted_features)
    assert store.last_drift_timestamp is not None


def test_inconclusive_evaluation_represented_correctly(temp_metrics_store):
    """Verify that an inconclusive evaluation is recorded accurately and not marked as NO DRIFT."""
    store = temp_metrics_store
    # Create single-row datasets that cause all feature tests to be skipped
    single_row_ref = pd.DataFrame([{col: 1.0 if col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"] else "Urban" for col in ALL_FEATURES}])
    single_row_prod = pd.DataFrame([{col: 1.0 if col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"] else "Urban" for col in ALL_FEATURES}])

    report = run_drift_detection(
        reference_data=single_row_ref,
        production_data=single_row_prod,
        metrics_tracker=store,
        record_metrics=True,
    )

    assert report.status == "INCONCLUSIVE"
    assert store.drift_events_count == 1
    assert store.last_drift_status == "INCONCLUSIVE"
    assert store.last_drift_status != "NO DRIFT DETECTED"


def test_retraining_attempt_with_no_drift_recorded(temp_metrics_store):
    """Verify that a retraining invocation with no drift is recorded as skipped."""
    store = temp_metrics_store
    drift_report = {
        "is_drift_detected": False,
        "drift_percentage": 0.0,
        "status": "NO DRIFT DETECTED",
    }

    result = orchestrate_retraining(
        drift_report=drift_report,
        force_retrain=False,
        metrics_tracker=store,
        record_metrics=True,
    )

    assert result["status"] == "skipped_no_drift"
    assert store.retraining_attempts == 1
    assert store.successful_promotions == 0
    assert len(store.retraining_history) == 1

    record = store.retraining_history[0]
    assert record["status"] == "skipped_no_drift"
    assert record["success"] is True
    assert record["promoted"] is False
    assert record["details"]["status"] == "skipped_no_drift"


@patch("src.retrain.predictor")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.run_training")
@patch("src.retrain.MlflowClient")
def test_rejected_candidate_recorded_without_changing_champion(
    mock_client_cls,
    mock_run_training,
    mock_load_champ,
    mock_predictor,
    temp_metrics_store,
    sample_features_and_target,
):
    """Verify that a rejected candidate is recorded with metrics without updating champion version."""
    store = temp_metrics_store
    store.current_model_version = "champion-v1"
    X, y = sample_features_and_target

    # Candidate and champion produce identical scores (f1=1.0) -> rejected under min_improvement=0.01
    champion_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    mock_load_champ.return_value = (champion_pipe, "champion-v1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 1.0},
        "mlflow": {"model_version": "candidate-v2"},
        "X_val": X,
        "y_val": y,
    }

    result = orchestrate_retraining(
        force_retrain=True,
        primary_metric="f1_score",
        min_improvement=0.01,
        metrics_tracker=store,
        record_metrics=True,
    )

    assert result["status"] == "rejected_no_improvement"
    assert store.retraining_attempts == 1
    assert store.successful_promotions == 0
    assert store.champion_performance == 1.0
    assert store.candidate_performance == 1.0
    # Current version remains champion-v1
    assert store.current_model_version == "champion-v1"

    record = store.retraining_history[0]
    assert record["status"] == "rejected_no_improvement"
    assert record["promoted"] is False
    assert record["details"]["status"] == "rejected_no_improvement"


@patch("src.retrain.predictor")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.run_training")
@patch("src.retrain.MlflowClient")
def test_successful_promotion_recorded(
    mock_client_cls,
    mock_run_training,
    mock_load_champ,
    mock_predictor,
    temp_metrics_store,
    sample_features_and_target,
):
    """Verify that a successful model promotion updates counts, scores, and active version."""
    store = temp_metrics_store
    store.current_model_version = "champion-v1"
    X, y = sample_features_and_target

    client_instance = mock_client_cls.return_value

    champion_pipe = DummyPipeline(["Y", "Y", "N", "N"])  # Worse predictions
    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])  # Perfect predictions
    mock_load_champ.return_value = (champion_pipe, "champion-v1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 1.0},
        "mlflow": {"model_version": "candidate-v2"},
        "X_val": X,
        "y_val": y,
    }

    result = orchestrate_retraining(
        force_retrain=True,
        primary_metric="f1_score",
        min_improvement=0.01,
        metrics_tracker=store,
        record_metrics=True,
    )

    assert result["status"] == "promoted"
    assert store.retraining_attempts == 1
    assert store.successful_promotions == 1
    assert store.champion_performance is not None
    assert store.candidate_performance is not None
    assert store.candidate_performance > store.champion_performance
    assert store.current_model_version == "candidate-v2"

    record = store.retraining_history[0]
    assert record["status"] == "promoted"
    assert record["promoted"] is True
    assert record["active_model_version"] == "candidate-v2"


@patch("src.retrain.MlflowClient")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.run_training", side_effect=RuntimeError("Training data corrupted"))
def test_failed_retraining_attempt_recorded(
    mock_run_training, mock_load_champ, mock_client_cls, temp_metrics_store
):
    """Verify that a failed retraining execution records failure and does not promote."""
    store = temp_metrics_store
    mock_load_champ.return_value = (DummyPipeline(["Y"]), "champion-v1")

    result = orchestrate_retraining(
        force_retrain=True,
        metrics_tracker=store,
        record_metrics=True,
    )

    assert result["status"] == "failed"
    assert store.retraining_attempts == 1
    assert store.successful_promotions == 0

    record = store.retraining_history[0]
    assert record["status"] == "failed"
    assert record["success"] is False
    assert record["promoted"] is False
    assert record["details"]["status"] == "failed"


@patch("src.retrain.predictor")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.run_training")
@patch("src.retrain.MlflowClient")
def test_partial_promotion_reload_failure_recorded_correctly(
    mock_client_cls,
    mock_run_training,
    mock_load_champ,
    mock_predictor,
    temp_metrics_store,
    sample_features_and_target,
):
    """Verify that partial promotion (alias set but serving reload failed) is not marked as successful promotion."""
    store = temp_metrics_store
    store.current_model_version = "champion-v1"
    X, y = sample_features_and_target

    # Predictor reload raises exception
    mock_predictor.set_model.side_effect = RuntimeError("Inference socket disconnected")

    champion_pipe = DummyPipeline(["Y", "Y", "N", "N"])
    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    mock_load_champ.return_value = (champion_pipe, "champion-v1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 1.0},
        "mlflow": {"model_version": "candidate-v2"},
        "X_val": X,
        "y_val": y,
    }

    result = orchestrate_retraining(
        force_retrain=True,
        primary_metric="f1_score",
        min_improvement=0.01,
        metrics_tracker=store,
        record_metrics=True,
    )

    assert result["status"] == "partial_promotion_reload_failed"
    assert store.retraining_attempts == 1
    # Must NOT increment successful_promotions
    assert store.successful_promotions == 0
    # Must NOT update current_model_version in monitoring store
    assert store.current_model_version == "champion-v1"

    record = store.retraining_history[0]
    assert record["status"] == "partial_promotion_reload_failed"
    assert record["promoted"] is False
    assert record["success"] is False
    assert record["details"]["status"] == "partial_promotion_reload_failed"


def test_monitoring_persistence_failure_does_not_break_workflows(sample_drift_data, tmp_path):
    """Verify that filesystem/persistence failures in MetricsStore do not crash drift detection or retraining."""
    ref_df, prod_df = sample_drift_data

    broken_file = tmp_path / "broken.json"
    broken_store = MetricsStore(filepath=broken_file, auto_save=False)
    broken_store.save_to_json = MagicMock(side_effect=PermissionError("Permission denied on metrics.json"))
    broken_store.record_drift_event = MagicMock(side_effect=PermissionError("Simulated write failure"))
    broken_store.record_retraining_event = MagicMock(side_effect=PermissionError("Simulated write failure"))

    # Drift detection must still succeed
    report = run_drift_detection(
        reference_data=ref_df,
        production_data=prod_df,
        metrics_tracker=broken_store,
        record_metrics=True,
    )
    assert isinstance(report, DriftReport)

    # Retraining orchestration must still succeed and return proper payload
    result = orchestrate_retraining(
        drift_report={"is_drift_detected": False, "drift_percentage": 0.0, "status": "NO DRIFT DETECTED"},
        force_retrain=False,
        metrics_tracker=broken_store,
        record_metrics=True,
    )
    assert result["status"] == "skipped_no_drift"


def test_no_duplicate_events_recorded(temp_metrics_store, sample_drift_data):
    """Verify that invoking drift evaluation and retraining records each event exactly once."""
    ref_df, prod_df = sample_drift_data
    store = temp_metrics_store

    # 1. Run drift detection once
    run_drift_detection(
        reference_data=ref_df,
        production_data=prod_df,
        metrics_tracker=store,
        record_metrics=True,
    )
    assert store.drift_events_count == 1

    # 2. Run retraining once
    orchestrate_retraining(
        drift_report={"is_drift_detected": False, "drift_percentage": 0.0, "status": "NO DRIFT DETECTED"},
        force_retrain=False,
        metrics_tracker=store,
        record_metrics=True,
    )
    assert store.retraining_attempts == 1
    assert len(store.retraining_history) == 1


def test_retraining_event_status_field_and_fallback(temp_metrics_store):
    """Verify that record_retraining_event stores explicit status and applies correct fallbacks."""
    store = temp_metrics_store

    # 1. Explicit status in details
    store.record_retraining_event(
        success=True,
        promoted=False,
        details={"status": "rejected_no_improvement"},
    )
    assert store.retraining_history[0]["status"] == "rejected_no_improvement"

    # 2. Legacy caller: promoted=True, no details
    store.record_retraining_event(
        success=True,
        promoted=True,
    )
    assert store.retraining_history[1]["status"] == "promoted"

    # 3. Legacy caller: success=True, promoted=False, no details
    store.record_retraining_event(
        success=True,
        promoted=False,
    )
    assert store.retraining_history[2]["status"] == "success"

    # 4. Legacy caller: success=False, promoted=False, no details
    store.record_retraining_event(
        success=False,
        promoted=False,
    )
    assert store.retraining_history[3]["status"] == "failed"

