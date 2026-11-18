"""Unit tests for the data_loader module."""

import tempfile
from pathlib import Path
import unittest
import warnings

import numpy as np
import pandas as pd

from src.data_loader import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    RAW_PREDICTION_COLUMNS,
    RAW_TRAINING_COLUMNS,
    TARGET_COLUMN,
    check_duplicate_rows,
    check_missing_values,
    get_features_and_target,
    get_model_features,
    load_prediction_data,
    load_training_data,
    validate_prediction_data,
    validate_training_data,
)


def _create_sample_training_df(n_rows: int = 5) -> pd.DataFrame:
    """Helper to create a small valid training DataFrame."""
    data = {
        "Loan_ID": [f"LP00100{i}" for i in range(n_rows)],
        "Gender": ["Male", "Female", "Male", "Male", "Female"][:n_rows],
        "Married": ["No", "Yes", "Yes", "No", "Yes"][:n_rows],
        "Dependents": ["0", "1", "2", "0", "0"][:n_rows],
        "Education": [
            "Graduate",
            "Graduate",
            "Not Graduate",
            "Graduate",
            "Graduate",
        ][:n_rows],
        "Self_Employed": ["No", "No", "Yes", "No", "No"][:n_rows],
        "ApplicantIncome": [5000, 3000, 4000, 6000, 2500][:n_rows],
        "CoapplicantIncome": [0.0, 1500.0, 2000.0, 0.0, 1200.0][:n_rows],
        "LoanAmount": [120.0, 110.0, 150.0, 200.0, 90.0][:n_rows],
        "Loan_Amount_Term": [360.0, 360.0, 360.0, 180.0, 360.0][:n_rows],
        "Credit_History": [1.0, 1.0, 0.0, 1.0, 1.0][:n_rows],
        "Property_Area": ["Urban", "Rural", "Semiurban", "Urban", "Semiurban"][
            :n_rows
        ],
        "Loan_Status": ["Y", "N", "N", "Y", "Y"][:n_rows],
    }
    return pd.DataFrame(data)


def _create_sample_prediction_df(n_rows: int = 4) -> pd.DataFrame:
    """Helper to create a small valid prediction DataFrame (no Loan_Status)."""
    df = _create_sample_training_df(n_rows=n_rows)
    return df.drop(columns=[TARGET_COLUMN])


class TestDataLoader(unittest.TestCase):
    """Test suite for data loading and validation routines."""

    def test_successful_training_data_loading(self):
        """Test loading a valid training CSV file successfully."""
        df_sample = _create_sample_training_df(n_rows=5)
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "train.csv"
            df_sample.to_csv(csv_path, index=False)

            loaded_df = load_training_data(filepath=csv_path)

            self.assertEqual(len(loaded_df), 5)
            self.assertEqual(len(loaded_df.columns), 13)
            self.assertIn(TARGET_COLUMN, loaded_df.columns)
            self.assertIn(ID_COLUMN, loaded_df.columns)
            self.assertListEqual(
                sorted(loaded_df.columns.tolist()),
                sorted(RAW_TRAINING_COLUMNS),
            )

    def test_successful_prediction_data_loading(self):
        """Test loading a valid prediction CSV file successfully."""
        df_sample = _create_sample_prediction_df(n_rows=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test.csv"
            df_sample.to_csv(csv_path, index=False)

            loaded_df = load_prediction_data(filepath=csv_path)

            self.assertEqual(len(loaded_df), 4)
            self.assertEqual(len(loaded_df.columns), 12)
            self.assertNotIn(TARGET_COLUMN, loaded_df.columns)
            self.assertIn(ID_COLUMN, loaded_df.columns)
            self.assertListEqual(
                sorted(loaded_df.columns.tolist()),
                sorted(RAW_PREDICTION_COLUMNS),
            )

    def test_missing_file_handling(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        non_existent_train = Path("non_existent_directory") / "train.csv"
        with self.assertRaises(FileNotFoundError) as ctx_train:
            load_training_data(filepath=non_existent_train)
        self.assertIn("Training dataset file not found", str(ctx_train.exception))

        non_existent_pred = Path("non_existent_directory") / "test.csv"
        with self.assertRaises(FileNotFoundError) as ctx_pred:
            load_prediction_data(filepath=non_existent_pred)
        self.assertIn("Prediction dataset file not found", str(ctx_pred.exception))

    def test_missing_required_column_handling(self):
        """Test that ValueError is raised if any expected column is missing."""
        df_sample = _create_sample_training_df()

        # Drop Credit_History from training data
        corrupted_training = df_sample.drop(columns=["Credit_History"])
        with self.assertRaises(ValueError) as ctx:
            validate_training_data(corrupted_training)
        self.assertIn("Credit_History", str(ctx.exception))

        # Drop Target column Loan_Status from training validation
        corrupted_training_no_target = df_sample.drop(columns=[TARGET_COLUMN])
        with self.assertRaises(ValueError) as ctx_no_target:
            validate_training_data(corrupted_training_no_target)
        self.assertIn(TARGET_COLUMN, str(ctx_no_target.exception))

        # Drop Education from prediction validation
        pred_sample = _create_sample_prediction_df()
        corrupted_prediction = pred_sample.drop(columns=["Education"])
        with self.assertRaises(ValueError) as ctx_pred:
            validate_prediction_data(corrupted_prediction)
        self.assertIn("Education", str(ctx_pred.exception))

    def test_correct_feature_and_target_separation(self):
        """Test separating X (features) and y (target) cleanly."""
        df_sample = _create_sample_training_df(n_rows=5)
        features, target = get_features_and_target(df_sample)

        # Target checks
        self.assertIsInstance(target, pd.Series)
        self.assertEqual(len(target), 5)
        self.assertEqual(target.name, TARGET_COLUMN)
        self.assertListEqual(target.tolist(), ["Y", "N", "N", "Y", "Y"])

        # Features checks
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 5)
        self.assertNotIn(TARGET_COLUMN, features.columns)
        self.assertNotIn(ID_COLUMN, features.columns)
        self.assertListEqual(sorted(features.columns.tolist()), sorted(FEATURE_COLUMNS))

        # Ensure original DataFrame is not mutated
        self.assertIn(TARGET_COLUMN, df_sample.columns)
        self.assertIn(ID_COLUMN, df_sample.columns)
        self.assertEqual(len(df_sample.columns), 13)

    def test_loan_id_excluded_from_model_features(self):
        """Test that Loan_ID is excluded from model features but kept in original DF."""
        df_sample = _create_sample_training_df(n_rows=5)

        # When extracting features and target
        features, _ = get_features_and_target(df_sample, drop_id=True)
        self.assertNotIn(ID_COLUMN, features.columns)

        # When extracting model features directly
        model_feats = get_model_features(df_sample)
        self.assertNotIn(ID_COLUMN, model_feats.columns)
        self.assertNotIn(TARGET_COLUMN, model_feats.columns)
        self.assertEqual(len(model_feats.columns), 11)

        # Verify original df still has Loan_ID
        self.assertIn(ID_COLUMN, df_sample.columns)
        self.assertEqual(len(df_sample), 5)

    def test_missing_values_reported_without_deleting_records(self):
        """Test that missing values are accurately reported and no rows are dropped."""
        df_sample = _create_sample_training_df(n_rows=5)
        # Inject known missing values
        df_sample.loc[0, "Gender"] = np.nan
        df_sample.loc[1, "LoanAmount"] = np.nan
        df_sample.loc[2, "LoanAmount"] = np.nan

        initial_row_count = len(df_sample)
        missing_report = check_missing_values(df_sample)

        self.assertEqual(missing_report["Gender"], 1)
        self.assertEqual(missing_report["LoanAmount"], 2)
        self.assertEqual(missing_report["Education"], 0)

        # Verify that checking missing values did not delete or modify rows
        self.assertEqual(len(df_sample), initial_row_count)
        self.assertTrue(pd.isna(df_sample.loc[0, "Gender"]))
        self.assertTrue(pd.isna(df_sample.loc[1, "LoanAmount"]))

    def test_prediction_dataset_does_not_require_loan_status(self):
        """Test that prediction dataset validation succeeds without Loan_Status."""
        df_pred = _create_sample_prediction_df(n_rows=3)
        self.assertNotIn(TARGET_COLUMN, df_pred.columns)

        # Validation should succeed without error
        result = validate_prediction_data(df_pred)
        self.assertTrue(result)

        # get_features_and_target should fail when target is missing (ensuring test data cannot be mislabeled)
        with self.assertRaises(ValueError) as ctx:
            get_features_and_target(df_pred)
        self.assertIn("Target column 'Loan_Status' not found", str(ctx.exception))

    def test_duplicate_rows_reporting(self):
        """Test reporting duplicate rows without deleting any rows."""
        df_sample = _create_sample_training_df(n_rows=3)
        # Append a duplicate of the first row
        df_with_duplicate = pd.concat(
            [df_sample, df_sample.iloc[[0]]], ignore_index=True
        )

        self.assertEqual(len(df_with_duplicate), 4)
        num_duplicates = check_duplicate_rows(df_with_duplicate)
        self.assertEqual(num_duplicates, 1)

        # Ensure no rows were deleted
        self.assertEqual(len(df_with_duplicate), 4)

    def test_empty_dataframe_validation(self):
        """Test that empty DataFrames raise ValueError."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            validate_training_data(empty_df)
        with self.assertRaises(ValueError):
            validate_prediction_data(empty_df)

    def test_invalid_type_raises_type_error(self):
        """Test that passing non-DataFrames raises TypeError."""
        with self.assertRaises(TypeError):
            validate_training_data("not_a_df")  # type: ignore
        with self.assertRaises(TypeError):
            validate_prediction_data(12345)  # type: ignore
        with self.assertRaises(TypeError):
            get_features_and_target(["list"])  # type: ignore


if __name__ == "__main__":
    unittest.main()

