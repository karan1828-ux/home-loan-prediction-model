"""Unit and integration tests for FastAPI serving endpoints."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.predict import ModelNotLoadedError, predictor
from monitoring.metrics import metrics_tracker


class DummyClassifierWithProba:
    """Mock ML model implementing both predict and predict_proba."""

    def __init__(self, pred_class=1, proba=(0.2, 0.8), classes=(0, 1)):
        self.pred_class = pred_class
        self.proba = proba
        self.classes_ = np.array(classes)

    def predict(self, X):
        return np.array([self.pred_class] * len(X))

    def predict_proba(self, X):
        return np.array([self.proba] * len(X))


class DummyClassifierWithoutProba:
    """Mock ML model implementing predict without probability estimation."""

    def __init__(self, pred_class="Y"):
        self.pred_class = pred_class

    def predict(self, X):
        return [self.pred_class] * len(X)


@pytest.fixture
def client():
    """Create FastAPI test client with mocked startup model loading."""
    with patch.object(predictor, "load_model", return_value=None):
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture(autouse=True)
def setup_teardown():
    """Reset predictor and metrics tracker around each test."""
    metrics_tracker.reset()
    yield
    metrics_tracker.reset()


SAMPLE_APPLICATION_PAYLOAD = {
    "Loan_ID": "LP001002",
    "Gender": "Male",
    "Married": "No",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5849.0,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 128.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Urban",
}


def test_root_endpoint(client):
    """Test GET / returns basic application information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "description" in data
    assert data["status"] == "online"
    assert "endpoints" in data
    assert "/predict" in data["endpoints"].values() or "POST /predict" in data["endpoints"].values()


def test_health_endpoint_healthy(client):
    """Test GET /health when model is loaded."""
    mock_model = DummyClassifierWithProba()
    predictor.set_model(mock_model, model_name="HomeLoanClassifier", model_alias="champion")

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "HomeLoanClassifier"
    assert data["model_alias"] == "champion"
    assert "uptime_seconds" in data
    assert "timestamp" in data


def test_health_endpoint_degraded(client):
    """Test GET /health when model is unloaded."""
    predictor.unload_model()

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


def test_predict_approved(client):
    """Test POST /predict returns approval status and probability."""
    mock_model = DummyClassifierWithProba(pred_class=1, proba=(0.15, 0.85), classes=(0, 1))
    predictor.set_model(mock_model)

    response = client.post("/predict", json=SAMPLE_APPLICATION_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["loan_status"] == "Y"
    assert data["prediction"] == 1
    assert data["probability"] == 0.85
    assert data["loan_id"] == "LP001002"
    assert data["model_name"] is not None


def test_predict_rejected(client):
    """Test POST /predict returns rejection status and lower approval probability."""
    mock_model = DummyClassifierWithProba(pred_class=0, proba=(0.82, 0.18), classes=(0, 1))
    predictor.set_model(mock_model)

    payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    payload["Credit_History"] = 0.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["loan_status"] == "N"
    assert data["prediction"] == 0
    assert data["probability"] == 0.18


def test_predict_without_probability_support(client):
    """Test POST /predict returns None for probability if unsupported by model."""
    mock_model = DummyClassifierWithoutProba(pred_class="Y")
    predictor.set_model(mock_model)

    response = client.post("/predict", json=SAMPLE_APPLICATION_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["loan_status"] == "Y"
    assert data["probability"] is None


def test_predict_with_snake_case_aliases(client):
    """Test POST /predict accepts snake_case aliases for input fields."""
    mock_model = DummyClassifierWithProba(pred_class=1, proba=(0.1, 0.9))
    predictor.set_model(mock_model)

    alias_payload = {
        "loan_id": "LP009999",
        "gender": "Female",
        "married": "Yes",
        "dependents": "1",
        "education": "Graduate",
        "self_employed": "Yes",
        "applicant_income": 4500.0,
        "coapplicant_income": 1200.0,
        "loan_amount": 115.0,
        "loan_amount_term": 360.0,
        "credit_history": 1.0,
        "property_area": "Semiurban",
    }

    response = client.post("/predict", json=alias_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["loan_status"] == "Y"
    assert data["loan_id"] == "LP009999"


def test_predict_model_not_loaded_503(client):
    """Test POST /predict returns 503 when model cannot be loaded."""
    predictor.unload_model()

    with patch.object(predictor, "load_model", side_effect=ModelNotLoadedError("MLflow connection failed")):
        response = client.post("/predict", json=SAMPLE_APPLICATION_PAYLOAD)
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()


def test_predict_validation_missing_required_income(client):
    """Test POST /predict returns 422 when required ApplicantIncome is omitted."""
    invalid_payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    del invalid_payload["ApplicantIncome"]

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_validation_negative_income(client):
    """Test POST /predict returns 422 when ApplicantIncome is negative."""
    invalid_payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    invalid_payload["ApplicantIncome"] = -100.0

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_validation_negative_loan_amount(client):
    """Test POST /predict returns 422 when LoanAmount is negative."""
    invalid_payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    invalid_payload["LoanAmount"] = -50.0

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_validation_invalid_term(client):
    """Test POST /predict returns 422 when Loan_Amount_Term is zero or negative."""
    invalid_payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    invalid_payload["Loan_Amount_Term"] = 0.0

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """Test GET /metrics returns accurate operational metrics."""
    mock_model = DummyClassifierWithProba(pred_class=1, proba=(0.2, 0.8))
    predictor.set_model(mock_model)

    # Initial metrics
    res1 = client.get("/metrics")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["total_predictions"] == 0
    assert data1["approved_predictions"] == 0

    # Perform 1 approved prediction
    client.post("/predict", json=SAMPLE_APPLICATION_PAYLOAD)

    res2 = client.get("/metrics")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["total_predictions"] == 1
    assert data2["successful_predictions"] == 1
    assert data2["approved_predictions"] == 1
    assert data2["rejected_predictions"] == 0
    assert data2["approval_rate"] == 1.0
    assert data2["average_latency_ms"] >= 0.0


def test_reload_model_success(client):
    """Test POST /reload-model reloads model successfully."""
    with patch.object(predictor, "reload", return_value={"model_loaded": True}):
        predictor.set_model(DummyClassifierWithProba(), model_version="v2")
        response = client.post("/reload-model")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "reloaded" in data["message"].lower()
        assert "reloaded_at" in data


def test_reload_model_failure_503(client):
    """Test POST /reload-model returns 503 when reload fails."""
    with patch.object(predictor, "reload", side_effect=ModelNotLoadedError("Registry unreachable")):
        response = client.post("/reload-model")
        assert response.status_code == 503
        assert "failed" in response.json()["detail"].lower()


def test_batch_prediction_endpoint(client):
    """Test POST /predict/batch scores multiple applications."""
    mock_model = DummyClassifierWithProba(pred_class=1, proba=(0.1, 0.9))
    predictor.set_model(mock_model)

    batch_payload = {
        "applications": [
            SAMPLE_APPLICATION_PAYLOAD,
            {**SAMPLE_APPLICATION_PAYLOAD, "Loan_ID": "LP001003", "ApplicantIncome": 3000.0},
        ]
    }

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["loan_id"] == "LP001002"
    assert data[1]["loan_id"] == "LP001003"
    assert data[0]["loan_status"] == "Y"
