"""Unit tests for MLflow experiment tracking and model registry integration."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import ID_COLUMN, TARGET_COLUMN
from src.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    build_pipeline,
    log_training_run,
)
from src.preprocessing import create_preprocessor
from src.train import run_training, train_model


def _create_synthetic_training_dataset(n_samples: int = 50) -> pd.DataFrame:
    """Helper to create a synthetic dataset matching the raw training schema."""
    np.random.seed(42)
    n_pos = int(n_samples * 0.7)
    n_neg = n_samples - n_pos
    y = ["Y"] * n_pos + ["N"] * n_neg

    data = {
        ID_COLUMN: [f"LP00100{i}" for i in range(n_samples)],
        "Gender": np.random.choice(["Male", "Female"], size=n_samples),
        "Married": np.random.choice(["Yes", "No"], size=n_samples),
        "Dependents": np.random.choice(["0", "1", "2", "3+"], size=n_samples),
        "Education": np.random.choice(["Graduate", "Not Graduate"], size=n_samples),
        "Self_Employed": np.random.choice(["Yes", "No"], size=n_samples),
        "ApplicantIncome": np.random.uniform(2000, 10000, size=n_samples),
        "CoapplicantIncome": np.random.uniform(0, 5000, size=n_samples),
        "LoanAmount": np.random.uniform(50, 400, size=n_samples),
        "Loan_Amount_Term": np.random.choice([180.0, 360.0, 480.0], size=n_samples),
        "Credit_History": np.random.choice([0.0, 1.0], size=n_samples),
        "Property_Area": np.random.choice(["Urban", "Rural", "Semiurban"], size=n_samples),
        TARGET_COLUMN: y,
    }
    return pd.DataFrame(data)


class TestMLflowIntegration(unittest.TestCase):
    """Test suite for pipeline assembly, MLflow tracking, and model registration."""

    def setUp(self):
        """Set up trained components for testing."""
        df = _create_synthetic_training_dataset(n_samples=50)
        self.X_train = df.iloc[:40].drop(columns=[ID_COLUMN, TARGET_COLUMN])
        self.y_train = df.iloc[:40][TARGET_COLUMN]
        self.X_val = df.iloc[40:].drop(columns=[ID_COLUMN, TARGET_COLUMN])
        self.y_val = df.iloc[40:][TARGET_COLUMN]

        training_artifacts = train_model(self.X_train, self.y_train, random_state=42)
        self.model = training_artifacts["model"]
        self.preprocessor = training_artifacts["preprocessor"]

    def test_build_pipeline_structure(self):
        """Test building an end-to-end Pipeline without refitting components."""
        pipeline = build_pipeline(preprocessor=self.preprocessor, model=self.model)

        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.steps[0][0], "preprocessor")
        self.assertEqual(pipeline.steps[1][0], "classifier")
        self.assertIs(pipeline.named_steps["preprocessor"], self.preprocessor)
        self.assertIs(pipeline.named_steps["classifier"], self.model)

    def test_pipeline_predict_on_raw_dataframe(self):
        """Verify that the Pipeline accepts raw feature DataFrames directly."""
        pipeline = build_pipeline(preprocessor=self.preprocessor, model=self.model)

        # Raw features DataFrame (not transformed)
        predictions = pipeline.predict(self.X_val)

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_val))
        for p in predictions:
            self.assertIn(p, ["Y", "N"])

    def test_pipeline_predict_proba(self):
        """Verify that predict_proba works on the Pipeline."""
        pipeline = build_pipeline(preprocessor=self.preprocessor, model=self.model)

        probabilities = pipeline.predict_proba(self.X_val)

        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (len(self.X_val), 2))
        # Probabilities must sum to 1 per row
        row_sums = probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(self.X_val)))

    def test_build_pipeline_unfitted_raises_error(self):
        """Test that attempting to build a pipeline with unfitted estimators raises ValueError."""
        unfitted_preprocessor = create_preprocessor()
        unfitted_model = DecisionTreeClassifier(random_state=42)

        # Unfitted preprocessor
        with self.assertRaises(ValueError) as ctx:
            build_pipeline(unfitted_preprocessor, self.model)
        self.assertIn("Preprocessor must be fitted", str(ctx.exception))

        # Unfitted model
        with self.assertRaises(ValueError) as ctx_m:
            build_pipeline(self.preprocessor, unfitted_model)
        self.assertIn("Classifier model must be fitted", str(ctx_m.exception))

    def test_build_pipeline_invalid_types_raise_error(self):
        """Test that invalid types raise TypeError."""
        with self.assertRaises(TypeError):
            build_pipeline("not_a_preprocessor", self.model)  # type: ignore

        with self.assertRaises(TypeError):
            build_pipeline(self.preprocessor, "not_a_model")  # type: ignore

    @patch("src.mlflow_tracking.MlflowClient")
    @patch("src.mlflow_tracking.mlflow")
    def test_log_training_run_mocked(self, mock_mlflow, mock_client_cls):
        """Test MLflow parameter, metric, artifact logging and model registration."""
        # Configure mock objects
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_run.info.experiment_id = "exp_456"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/test_run_123/model"
        mock_model_info.registered_model_version = "1"
        mock_mlflow.sklearn.log_model.return_value = mock_model_info

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        pipeline = build_pipeline(preprocessor=self.preprocessor, model=self.model)
        metrics = {
            "accuracy": 0.85,
            "precision": 0.88,
            "recall": 0.82,
            "f1_score": 0.85,
            "confusion_matrix": np.array([[10, 2], [3, 15]]),
        }
        params = {
            "random_state": 42,
            "test_size": 0.20,
            "model_type": "DecisionTreeClassifier",
        }

        # Enable MLFLOW_AVAILABLE for the test
        with patch("src.mlflow_tracking.MLFLOW_AVAILABLE", True):
            result = log_training_run(
                pipeline=pipeline,
                metrics=metrics,
                params=params,
                tracking_uri="http://localhost:5000",
                experiment_name="home-loan-experiments",
                model_name="HomeLoanClassifier",
                model_alias="champion",
                register_model=True,
                classification_report_text="Sample report",
            )

        # 1. Verify tracking URI and experiment
        mock_mlflow.set_tracking_uri.assert_called_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_with("home-loan-experiments")

        # 2. Verify parameter logging
        mock_mlflow.log_params.assert_called_once()
        logged_params = mock_mlflow.log_params.call_args[0][0]
        self.assertEqual(logged_params["random_state"], 42)
        self.assertEqual(logged_params["test_size"], 0.20)

        # 3. Verify metric logging
        mock_mlflow.log_metrics.assert_called_once()
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        self.assertEqual(logged_metrics["accuracy"], 0.85)
        self.assertEqual(logged_metrics["f1_score"], 0.85)

        # 4. Verify model logging
        mock_mlflow.sklearn.log_model.assert_called_once_with(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="HomeLoanClassifier",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

        # 5. Verify model alias assignment to 'champion'
        mock_client.set_registered_model_alias.assert_called_once_with(
            name="HomeLoanClassifier",
            alias="champion",
            version="1",
        )

        # 6. Verify returned metadata
        self.assertEqual(result["run_id"], "test_run_123")
        self.assertEqual(result["model_alias"], "champion")
        self.assertEqual(result["model_name"], "HomeLoanClassifier")
        self.assertEqual(result["model_version"], "1")

    @patch("src.mlflow_tracking.MlflowClient")
    @patch("src.mlflow_tracking.mlflow")
    def test_run_training_with_mlflow_enabled(self, mock_mlflow, mock_client_cls):
        """Test that run_training orchestrates MLflow logging when enable_mlflow=True."""
        import tempfile
        from pathlib import Path

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_orchestrated"
        mock_run.info.experiment_id = "exp_orchestrated"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/test_run_orchestrated/model"
        mock_model_info.registered_model_version = "2"
        mock_mlflow.sklearn.log_model.return_value = mock_model_info

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        df = _create_synthetic_training_dataset(n_samples=50)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "train.csv"
            df.to_csv(csv_path, index=False)

            with patch("src.mlflow_tracking.MLFLOW_AVAILABLE", True):
                results = run_training(
                    filepath=csv_path,
                    test_size=0.20,
                    random_state=42,
                    enable_mlflow=True,
                )

            # Check that results contains standard outputs plus MLflow metadata
            self.assertIn("model", results)
            self.assertIn("preprocessor", results)
            self.assertIn("pipeline", results)
            self.assertIn("metrics", results)
            self.assertIn("mlflow", results)
            self.assertEqual(results["mlflow"]["run_id"], "test_run_orchestrated")
            self.assertEqual(results["mlflow"]["model_alias"], "champion")

    @patch("src.mlflow_tracking.MlflowClient")
    @patch("src.mlflow_tracking.mlflow")
    def test_log_training_run_explicit_none_alias(self, mock_mlflow, mock_client_cls):
        """Test that passing model_alias=None registers the model without setting any alias."""
        mock_run = MagicMock()
        mock_run.info.run_id = "run_no_alias"
        mock_run.info.experiment_id = "exp_no_alias"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/run_no_alias/model"
        mock_model_info.registered_model_version = "3"
        mock_mlflow.sklearn.log_model.return_value = mock_model_info

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        pipeline = build_pipeline(preprocessor=self.preprocessor, model=self.model)

        with patch("src.mlflow_tracking.MLFLOW_AVAILABLE", True):
            result = log_training_run(
                pipeline=pipeline,
                metrics={"accuracy": 0.8},
                params={"random_state": 42},
                model_alias=None,
                register_model=True,
            )

        # Model should still be logged and registered
        mock_mlflow.sklearn.log_model.assert_called_once()
        # But set_registered_model_alias must NOT be called
        mock_client.set_registered_model_alias.assert_not_called()
        self.assertIsNone(result["model_alias"])
        self.assertEqual(result["model_version"], "3")

    @patch("src.mlflow_tracking.MlflowClient")
    @patch("src.mlflow_tracking.mlflow")
    def test_log_training_run_custom_candidate_alias(self, mock_mlflow, mock_client_cls):
        """Test that passing a custom alias (e.g. 'candidate') sets only that alias."""
        mock_run = MagicMock()
        mock_run.info.run_id = "run_candidate"
        mock_run.info.experiment_id = "exp_candidate"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/run_candidate/model"
        mock_model_info.registered_model_version = "4"
        mock_mlflow.sklearn.log_model.return_value = mock_model_info

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        pipeline = build_pipeline(preprocessor=self.preprocessor, model=self.model)

        with patch("src.mlflow_tracking.MLFLOW_AVAILABLE", True):
            result = log_training_run(
                pipeline=pipeline,
                metrics={"accuracy": 0.82},
                params={"random_state": 42},
                model_alias="candidate",
                register_model=True,
            )

        # set_registered_model_alias must be called specifically with 'candidate'
        mock_client.set_registered_model_alias.assert_called_once_with(
            name="HomeLoanClassifier",
            alias="candidate",
            version="4",
        )
        self.assertEqual(result["model_alias"], "candidate")

    @patch("src.mlflow_tracking.MlflowClient")
    @patch("src.mlflow_tracking.mlflow")
    def test_run_training_with_explicit_none_alias(self, mock_mlflow, mock_client_cls):
        """Test run_training with model_alias=None registers model without assigning alias."""
        import tempfile
        from pathlib import Path

        mock_run = MagicMock()
        mock_run.info.run_id = "run_orch_no_alias"
        mock_run.info.experiment_id = "exp_orch_no_alias"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/run_orch_no_alias/model"
        mock_model_info.registered_model_version = "5"
        mock_mlflow.sklearn.log_model.return_value = mock_model_info

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        df = _create_synthetic_training_dataset(n_samples=50)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "train.csv"
            df.to_csv(csv_path, index=False)

            with patch("src.mlflow_tracking.MLFLOW_AVAILABLE", True):
                results = run_training(
                    filepath=csv_path,
                    test_size=0.20,
                    random_state=42,
                    enable_mlflow=True,
                    model_alias=None,
                )

            mock_client.set_registered_model_alias.assert_not_called()
            self.assertIsNone(results["mlflow"]["model_alias"])


if __name__ == "__main__":
    unittest.main()

