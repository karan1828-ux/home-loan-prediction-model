"""Unit tests for automated model retraining and conditional model promotion."""

import math
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.drift_detection import DriftReport, FeatureDriftResult
from src.retrain import (
    _build_payload,
    _load_champion_from_mlflow,
    _validate_retraining_inputs,
    orchestrate_retraining,
)


@pytest.fixture
def mock_drift_report_drift_detected():
    """Create a mock DriftReport with confirmed drift."""
    return DriftReport(
        feature_results={
            "ApplicantIncome": FeatureDriftResult(
                feature_name="ApplicantIncome",
                feature_type="numerical",
                test_name="Kolmogorov-Smirnov",
                statistic=0.45,
                p_value=0.001,
                drift_detected=True,
                is_valid=True,
                status="DRIFT_DETECTED",
            ),
            "Property_Area": FeatureDriftResult(
                feature_name="Property_Area",
                feature_type="categorical",
                test_name="Chi-Square",
                statistic=15.2,
                p_value=0.002,
                drift_detected=True,
                is_valid=True,
                status="DRIFT_DETECTED",
            ),
        },
        p_value_threshold=0.05,
        drift_share_threshold=0.20,
    )


@pytest.fixture
def mock_drift_report_no_drift():
    """Create a mock DriftReport with no drift."""
    return DriftReport(
        feature_results={
            "ApplicantIncome": FeatureDriftResult(
                feature_name="ApplicantIncome",
                feature_type="numerical",
                test_name="Kolmogorov-Smirnov",
                statistic=0.05,
                p_value=0.85,
                drift_detected=False,
                is_valid=True,
                status="NO_DRIFT",
            ),
        },
        p_value_threshold=0.05,
        drift_share_threshold=0.20,
    )


@pytest.fixture
def mock_drift_report_inconclusive():
    """Create a mock DriftReport with 0 valid features (inconclusive)."""
    return DriftReport(
        feature_results={
            "ApplicantIncome": FeatureDriftResult(
                feature_name="ApplicantIncome",
                feature_type="numerical",
                test_name="Kolmogorov-Smirnov",
                statistic=None,
                p_value=None,
                drift_detected=False,
                is_valid=False,
                status="SKIPPED_INSUFFICIENT_DATA",
            ),
        },
        p_value_threshold=0.05,
        drift_share_threshold=0.20,
    )


@pytest.fixture
def sample_features_and_target():
    """Generate dummy evaluation features and labels."""
    X = pd.DataFrame(
        {
            "ApplicantIncome": [5000.0, 6000.0, 7000.0, 8000.0],
            "CoapplicantIncome": [1500.0, 0.0, 2000.0, 1000.0],
            "LoanAmount": [120.0, 140.0, 160.0, 180.0],
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


class DummyPipeline:
    """Mock Scikit-learn Pipeline with predictable predictions."""

    def __init__(self, predictions):
        self.predictions = predictions
        self.recorded_X = None

    def predict(self, X):
        self.recorded_X = X.copy()
        return self.predictions


def test_build_payload_structure():
    """Test _build_payload returns the exact 14-field dictionary specification."""
    payload = _build_payload(status="test_status", message="test message")
    assert isinstance(payload, dict)
    expected_keys = {
        "status",
        "drift_detected",
        "drift_percentage",
        "promoted",
        "candidate_version",
        "active_model_version",
        "candidate_metrics",
        "champion_metrics",
        "candidate_score",
        "champion_score",
        "metric_evaluated",
        "evaluation_source",
        "message",
        "comparison",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["status"] == "test_status"
    assert payload["promoted"] is False
    assert payload["candidate_version"] is None


def test_no_drift_skip(mock_drift_report_no_drift):
    """Test retraining is skipped when drift is not detected."""
    res = orchestrate_retraining(drift_report=mock_drift_report_no_drift)
    assert res["status"] == "skipped_no_drift"
    assert res["promoted"] is False
    assert res["drift_detected"] is False
    assert "not exceeded" in res["message"]


def test_inconclusive_drift_skip(mock_drift_report_inconclusive):
    """Test retraining is skipped when drift evaluation is inconclusive."""
    res = orchestrate_retraining(drift_report=mock_drift_report_inconclusive)
    assert res["status"] == "skipped_inconclusive"
    assert res["promoted"] is False
    assert "INCONCLUSIVE" in res["message"]


def test_missing_drift_skip():
    """Test retraining is skipped when drift report is missing and force_retrain=False."""
    res = orchestrate_retraining(drift_report=None, force_retrain=False)
    assert res["status"] == "skipped_no_drift"
    assert res["promoted"] is False
    assert "Missing drift report" in res["message"]


def test_unsupported_drift_report_type_raises_type_error():
    """Test TypeError is raised when unsupported drift_report type is provided."""
    with pytest.raises(TypeError, match="Unsupported drift_report type"):
        orchestrate_retraining(drift_report="invalid_string_report")  # type: ignore

    with pytest.raises(TypeError, match="Unsupported drift_report type"):
        orchestrate_retraining(drift_report=[1, 2, 3])  # type: ignore


@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_force_retraining_without_drift(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_drift_report_no_drift,
    sample_features_and_target,
):
    """Test force_retrain=True executes retraining even if drift report reports no drift."""
    X, y = sample_features_and_target
    mock_load_champ.return_value = (None, None)

    mock_run_training.return_value = {
        "pipeline": DummyPipeline(["Y", "N", "N", "Y"]),
        "metrics": {"f1_score": 0.85, "accuracy": 0.85, "precision": 0.85, "recall": 0.85},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(
        drift_report=mock_drift_report_no_drift,
        force_retrain=True,
    )

    mock_run_training.assert_called_once()
    assert res["status"] == "promoted"
    assert res["promoted"] is True


@patch("src.retrain.predictor")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_candidate_better_than_champion(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_predictor,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test candidate is promoted when its score strictly exceeds champion + min_improvement."""
    X, y = sample_features_and_target
    client_instance = mock_client_cls.return_value

    champion_pipe = DummyPipeline(["Y", "Y", "N", "N"])  # Worse predictions
    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])  # Perfect predictions
    mock_load_champ.return_value = (champion_pipe, "1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 1.0, "accuracy": 1.0, "precision": 1.0, "recall": 1.0},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(
        drift_report=mock_drift_report_drift_detected,
        primary_metric="f1_score",
        min_improvement=0.05,
    )

    assert res["status"] == "promoted"
    assert res["promoted"] is True
    assert res["candidate_version"] == "2"
    assert res["candidate_score"] > res["champion_score"]

    # Verify MLflow alias promotion call
    client_instance.set_registered_model_alias.assert_called_once_with(
        name="HomeLoanClassifier",
        alias="champion",
        version="2",
    )
    # Verify predictor update call
    mock_predictor.set_model.assert_called_once_with(
        model=candidate_pipe,
        model_name="HomeLoanClassifier",
        model_alias="champion",
        model_version="2",
    )


@patch("src.retrain.predictor")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_candidate_equal_to_champion_rejected(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_predictor,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test candidate is rejected when score is equal to champion (strict improvement required)."""
    X, y = sample_features_and_target
    client_instance = mock_client_cls.return_value

    # Both yield identical predictions
    champion_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    mock_load_champ.return_value = (champion_pipe, "1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 1.0, "accuracy": 1.0, "precision": 1.0, "recall": 1.0},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(
        drift_report=mock_drift_report_drift_detected,
        primary_metric="f1_score",
        min_improvement=0.0,
    )

    assert res["status"] == "rejected_no_improvement"
    assert res["promoted"] is False
    assert res["active_model_version"] == "1"

    # Alias must not be changed
    client_instance.set_registered_model_alias.assert_not_called()
    mock_predictor.set_model.assert_not_called()


@patch("src.retrain.predictor")
@patch("src.retrain.evaluate_model")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_strict_improvement_boundary_exact_difference_rejected(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_evaluate_model,
    mock_predictor,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test when diff == min_improvement, candidate is rejected (strictly greater required)."""
    X, y = sample_features_and_target
    client_instance = mock_client_cls.return_value

    champion_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    mock_load_champ.return_value = (champion_pipe, "1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 0.85},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    # Simulate candidate = 0.85, champion = 0.80 -> diff = 0.05
    mock_evaluate_model.side_effect = [
        {"f1_score": 0.85},  # candidate
        {"f1_score": 0.80},  # champion
    ]

    res = orchestrate_retraining(
        drift_report=mock_drift_report_drift_detected,
        primary_metric="f1_score",
        min_improvement=0.05,  # diff == min_improvement
    )

    assert res["status"] == "rejected_no_improvement"
    assert res["promoted"] is False
    assert res["comparison"]["difference"] == 0.05
    assert client_instance.set_registered_model_alias.call_count == 0
    mock_predictor.set_model.assert_not_called()


@patch("src.retrain.predictor")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_candidate_worse_than_champion_rejected(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_predictor,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test candidate is rejected when score is worse than champion."""
    X, y = sample_features_and_target
    client_instance = mock_client_cls.return_value

    champion_pipe = DummyPipeline(["Y", "N", "N", "Y"])  # Perfect predictions
    candidate_pipe = DummyPipeline(["N", "Y", "Y", "N"])  # Opposite predictions
    mock_load_champ.return_value = (champion_pipe, "1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(
        drift_report=mock_drift_report_drift_detected,
        primary_metric="f1_score",
    )

    assert res["status"] == "rejected_no_improvement"
    assert res["promoted"] is False
    assert res["active_model_version"] == "1"
    client_instance.set_registered_model_alias.assert_not_called()


@patch("src.retrain.predictor")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_initial_promotion_when_no_champion_exists(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_predictor,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test candidate is promoted as the first champion when no prior champion exists in MLflow."""
    X, y = sample_features_and_target
    client_instance = mock_client_cls.return_value

    # No champion in registry
    mock_load_champ.return_value = (None, None)

    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 0.85, "accuracy": 0.85, "precision": 0.85, "recall": 0.85},
        "mlflow": {"model_version": "1"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(
        drift_report=mock_drift_report_drift_detected,
    )

    assert res["status"] == "promoted"
    assert res["promoted"] is True
    assert res["champion_score"] is None
    assert "First champion registration" in res["comparison"]["decision"]
    client_instance.set_registered_model_alias.assert_called_once_with(
        name="HomeLoanClassifier",
        alias="champion",
        version="1",
    )


@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_both_models_evaluated_on_identical_evaluation_rows(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Verify candidate and champion are evaluated on the exact same DataFrame rows."""
    X, y = sample_features_and_target

    champion_pipe = DummyPipeline(["Y", "Y", "N", "N"])
    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    mock_load_champ.return_value = (champion_pipe, "1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 1.0, "accuracy": 1.0, "precision": 1.0, "recall": 1.0},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    _ = orchestrate_retraining(drift_report=mock_drift_report_drift_detected)

    assert candidate_pipe.recorded_X is not None
    assert champion_pipe.recorded_X is not None
    pd.testing.assert_frame_equal(candidate_pipe.recorded_X, champion_pipe.recorded_X)


@patch("src.retrain.load_training_data")
@patch("src.retrain.get_features_and_target")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_dedicated_evaluation_dataset_usage(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_get_feat_target,
    mock_load_data,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test dedicated evaluation dataset path is used when provided."""
    X, y = sample_features_and_target
    mock_load_data.return_value = pd.DataFrame({"dummy": [1, 2]})
    mock_get_feat_target.return_value = (X, y)
    mock_load_champ.return_value = (None, None)

    mock_run_training.return_value = {
        "pipeline": DummyPipeline(["Y", "N", "N", "Y"]),
        "metrics": {"f1_score": 0.85, "accuracy": 0.85, "precision": 0.85, "recall": 0.85},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(
        drift_report=mock_drift_report_drift_detected,
        evaluation_data_path="data/raw/eval.csv",
    )

    mock_load_data.assert_called_once_with("data/raw/eval.csv")
    assert res["evaluation_source"] == "dedicated_file"


@patch("src.retrain.load_training_data")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_dedicated_evaluation_dataset_empty_fails(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_load_data,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test empty dedicated evaluation dataset returns failure and halts promotion."""
    X, y = sample_features_and_target
    mock_load_data.return_value = pd.DataFrame()  # Empty dataframe
    mock_load_champ.return_value = (DummyPipeline(["Y"]), "1")

    mock_run_training.return_value = {
        "pipeline": DummyPipeline(["Y", "N"]),
        "metrics": {"f1_score": 0.85},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(
        drift_report=mock_drift_report_drift_detected,
        evaluation_data_path="data/raw/empty_eval.csv",
    )

    assert res["status"] == "failed"
    assert res["promoted"] is False
    assert "is empty" in res["message"]


@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_validation_split_fallback(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test validation split is used as fallback when evaluation_data_path is None."""
    X, y = sample_features_and_target
    mock_load_champ.return_value = (None, None)

    mock_run_training.return_value = {
        "pipeline": DummyPipeline(["Y", "N", "N", "Y"]),
        "metrics": {"f1_score": 0.85, "accuracy": 0.85, "precision": 0.85, "recall": 0.85},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(
        drift_report=mock_drift_report_drift_detected,
        evaluation_data_path=None,
    )

    assert res["evaluation_source"] == "validation_split_fallback"


@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_candidate_training_failure(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_drift_report_drift_detected,
):
    """Test graceful handling when candidate training raises an exception."""
    mock_load_champ.return_value = (DummyPipeline(["Y"]), "1")
    mock_run_training.side_effect = RuntimeError("Disk full / training failure")

    res = orchestrate_retraining(drift_report=mock_drift_report_drift_detected)

    assert res["status"] == "failed"
    assert res["promoted"] is False
    assert res["active_model_version"] == "1"
    assert "Candidate model training failed" in res["message"]


@patch("src.retrain.predictor")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_missing_candidate_model_version_aborts_promotion(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_predictor,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test when run_training output lacks model_version, promotion aborts and champion is preserved."""
    X, y = sample_features_and_target
    client_instance = mock_client_cls.return_value
    mock_load_champ.return_value = (DummyPipeline(["Y"]), "1")

    mock_run_training.return_value = {
        "pipeline": DummyPipeline(["Y", "N"]),
        "metrics": {"f1_score": 0.85},
        "mlflow": {"model_version": None},  # Missing version
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(drift_report=mock_drift_report_drift_detected)

    assert res["status"] == "failed"
    assert res["promoted"] is False
    assert res["active_model_version"] == "1"
    assert "missing or invalid" in res["message"]
    client_instance.set_registered_model_alias.assert_not_called()
    mock_predictor.set_model.assert_not_called()


@patch("src.retrain.evaluate_model")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_evaluation_failure(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_evaluate_model,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test graceful failure when evaluation routine encounters an error."""
    X, y = sample_features_and_target
    mock_load_champ.return_value = (DummyPipeline(["Y"]), "1")

    mock_run_training.return_value = {
        "pipeline": DummyPipeline(["Y"]),
        "metrics": {"f1_score": 0.8},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }
    mock_evaluate_model.side_effect = ValueError("Labels mismatch")

    res = orchestrate_retraining(drift_report=mock_drift_report_drift_detected)

    assert res["status"] == "failed"
    assert res["promoted"] is False
    assert "Model evaluation failed" in res["message"]


def test_mlflow_unavailable_fails_safely(mock_drift_report_drift_detected):
    """Test retraining returns failed status when MLflow is disabled (no offline promotion)."""
    res = orchestrate_retraining(
        drift_report=mock_drift_report_drift_detected,
        enable_mlflow=False,
    )
    assert res["status"] == "failed"
    assert res["promoted"] is False
    assert "MLflow must be enabled" in res["message"]


@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_champion_loading_failure(
    mock_client_cls,
    mock_load_champ,
    mock_drift_report_drift_detected,
):
    """Test unexpected champion loading failure does not assume no champion."""
    mock_load_champ.side_effect = RuntimeError("Corrupted model artifact or network failure")

    res = orchestrate_retraining(drift_report=mock_drift_report_drift_detected)

    assert res["status"] == "failed"
    assert res["promoted"] is False
    assert "Champion model loading failed" in res["message"]


@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_alias_promotion_failure(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test failure during alias reassignment does not leave the system in an inconsistent state."""
    X, y = sample_features_and_target
    client_instance = mock_client_cls.return_value
    client_instance.set_registered_model_alias.side_effect = RuntimeError("Registry write failure")

    champion_pipe = DummyPipeline(["Y", "Y", "N", "N"])
    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    mock_load_champ.return_value = (champion_pipe, "1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 1.0, "accuracy": 1.0, "precision": 1.0, "recall": 1.0},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(drift_report=mock_drift_report_drift_detected)

    assert res["status"] == "failed"
    assert res["promoted"] is False
    assert "Failed to assign 'champion' alias" in res["message"]


@patch("src.retrain.predictor")
@patch("src.retrain.run_training")
@patch("src.retrain._load_champion_from_mlflow")
@patch("src.retrain.MlflowClient")
def test_serving_model_reload_failure_after_promotion(
    mock_client_cls,
    mock_load_champ,
    mock_run_training,
    mock_predictor,
    mock_drift_report_drift_detected,
    sample_features_and_target,
):
    """Test partial promotion is accurately reported when predictor reload fails after MLflow promotion."""
    X, y = sample_features_and_target
    mock_predictor.set_model.side_effect = RuntimeError("In-memory memory allocation error")

    champion_pipe = DummyPipeline(["Y", "Y", "N", "N"])
    candidate_pipe = DummyPipeline(["Y", "N", "N", "Y"])
    mock_load_champ.return_value = (champion_pipe, "1")

    mock_run_training.return_value = {
        "pipeline": candidate_pipe,
        "metrics": {"f1_score": 1.0, "accuracy": 1.0, "precision": 1.0, "recall": 1.0},
        "mlflow": {"model_version": "2"},
        "X_val": X,
        "y_val": y,
    }

    res = orchestrate_retraining(drift_report=mock_drift_report_drift_detected)

    # MLflow alias was set, but reload failed
    assert res["status"] == "partial_promotion_reload_failed"
    assert res["promoted"] is True
    assert res["candidate_version"] == "2"
    assert "serving predictor reload failed" in res["message"]


def test_invalid_metric_name(mock_drift_report_drift_detected):
    """Test ValueError is raised when unsupported primary_metric is specified."""
    with pytest.raises(ValueError, match="Unsupported primary_metric"):
        orchestrate_retraining(
            drift_report=mock_drift_report_drift_detected,
            primary_metric="unsupported_loss",
        )


def test_invalid_min_improvement(mock_drift_report_drift_detected):
    """Test ValueError or TypeError is raised for invalid min_improvement values."""
    with pytest.raises(ValueError, match="min_improvement"):
        orchestrate_retraining(
            drift_report=mock_drift_report_drift_detected,
            min_improvement=-0.05,
        )

    with pytest.raises(ValueError, match="min_improvement"):
        orchestrate_retraining(
            drift_report=mock_drift_report_drift_detected,
            min_improvement=float("nan"),
        )

    with pytest.raises(TypeError, match="min_improvement"):
        orchestrate_retraining(
            drift_report=mock_drift_report_drift_detected,
            min_improvement=True,  # type: ignore
        )


def test_load_champion_nonexistent_returns_none():
    """Test _load_champion_from_mlflow returns (None, None) when model alias not found."""
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = Exception("RESOURCE_DOES_NOT_EXIST: Not found")

    model, version = _load_champion_from_mlflow(mock_client, "HomeLoanClassifier", "champion")
    assert model is None
    assert version is None
