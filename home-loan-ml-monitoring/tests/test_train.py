"""Unit tests for the model training module."""

import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import ID_COLUMN, TARGET_COLUMN
from src.train import run_training, train_model


def _create_synthetic_training_dataset(n_samples: int = 50) -> pd.DataFrame:
    """Helper to create a synthetic dataset matching the full raw training schema."""
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


class TestTrain(unittest.TestCase):
    """Test suite for model training routines."""

    def test_train_model_success(self):
        """Test that train_model returns a trained model and fitted preprocessor."""
        df = _create_synthetic_training_dataset(n_samples=40)
        X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        result = train_model(X, y, random_state=42)

        self.assertIn("model", result)
        self.assertIn("preprocessor", result)
        self.assertIsInstance(result["model"], DecisionTreeClassifier)
        self.assertTrue(hasattr(result["preprocessor"], "transformers_"))

        # Verify classes
        self.assertListEqual(sorted(list(result["model"].classes_)), ["N", "Y"])

    def test_train_model_with_missing_values(self):
        """Test that train_model successfully trains even when input contains NaNs."""
        df = _create_synthetic_training_dataset(n_samples=40)
        X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
        y = df[TARGET_COLUMN]

        # Inject NaNs in both numeric and categorical columns
        X.loc[0, "ApplicantIncome"] = np.nan
        X.loc[1, "LoanAmount"] = np.nan
        X.loc[2, "Credit_History"] = np.nan
        X.loc[3, "Gender"] = np.nan
        X.loc[4, "Dependents"] = np.nan

        # Should train without error due to internal imputation
        result = train_model(X, y, random_state=42)

        self.assertIsInstance(result["model"], DecisionTreeClassifier)
        self.assertIsNotNone(result["preprocessor"])

    def test_preprocessing_fitted_only_on_training_data(self):
        """Verify that preprocessor statistics are derived strictly from X_train."""
        df = _create_synthetic_training_dataset(n_samples=40)
        X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
        y = df[TARGET_COLUMN]

        # Set specific known values for LoanAmount in X_train
        X_train = X.iloc[:30].copy()
        y_train = y.iloc[:30].copy()
        X_train["LoanAmount"] = 150.0  # Constant median = 150.0

        result = train_model(X_train, y_train, random_state=42)
        preprocessor = result["preprocessor"]

        # Find numerical imputer
        num_transformer = [t for t in preprocessor.transformers_ if t[0] == "num"][0][1]
        imputer = num_transformer.named_steps["imputer"]

        # LoanAmount is at index 2 of NUMERICAL_FEATURES
        self.assertEqual(imputer.statistics_[2], 150.0)

    def test_successful_validation_prediction(self):
        """Test predicting on a validation set using the trained model and preprocessor."""
        df = _create_synthetic_training_dataset(n_samples=40)
        X_train = df.iloc[:30].drop(columns=[ID_COLUMN, TARGET_COLUMN])
        y_train = df.iloc[:30][TARGET_COLUMN]

        X_val = df.iloc[30:].drop(columns=[ID_COLUMN, TARGET_COLUMN])
        y_val = df.iloc[30:][TARGET_COLUMN]

        result = train_model(X_train, y_train, random_state=42)
        model = result["model"]
        preprocessor = result["preprocessor"]

        from src.preprocessing import transform_features

        X_val_transformed = transform_features(preprocessor, X_val)
        predictions = model.predict(X_val_transformed)

        self.assertEqual(len(predictions), len(y_val))
        for pred in predictions:
            self.assertIn(pred, ["Y", "N"])

    def test_stratified_train_validation_split(self):
        """Test that run_training preserves class proportions during split."""
        df = _create_synthetic_training_dataset(n_samples=50)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "synthetic_train.csv"
            df.to_csv(csv_path, index=False)

            results = run_training(filepath=csv_path, test_size=0.20, random_state=42)

            y_train = results["y_train"]
            y_val = results["y_val"]

            # 50 total: 40 train, 10 val
            self.assertEqual(len(y_train), 40)
            self.assertEqual(len(y_val), 10)

            # Class ratio in original: 35/50 = 70% 'Y', 15/50 = 30% 'N'
            # Train should have 28 'Y' (70%) and 12 'N' (30%)
            # Val should have 7 'Y' (70%) and 3 'N' (30%)
            self.assertEqual((y_train == "Y").sum(), 28)
            self.assertEqual((y_val == "Y").sum(), 7)

    def test_run_training_returns_complete_artifacts(self):
        """Test that run_training returns all required keys and metrics."""
        df = _create_synthetic_training_dataset(n_samples=40)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "train.csv"
            df.to_csv(csv_path, index=False)

            results = run_training(filepath=csv_path, test_size=0.25, random_state=42)

            expected_keys = [
                "model",
                "preprocessor",
                "pipeline",
                "metrics",
                "X_train",
                "X_val",
                "y_train",
                "y_val",
                "y_pred",
            ]
            for key in expected_keys:
                self.assertIn(key, results)

            metrics = results["metrics"]
            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1_score", metrics)
            self.assertIn("confusion_matrix", metrics)

    def test_train_model_invalid_inputs(self):
        """Test that invalid training inputs raise TypeError or ValueError."""
        with self.assertRaises(TypeError):
            train_model("not_a_dataframe", pd.Series(["Y", "N"]))  # type: ignore

        with self.assertRaises(ValueError):
            train_model(pd.DataFrame(), pd.Series([], dtype=object))

        df = _create_synthetic_training_dataset(n_samples=10)
        X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
        with self.assertRaises(ValueError):
            train_model(X, pd.Series(["Y", "N"]))  # Length mismatch


if __name__ == "__main__":
    unittest.main()
