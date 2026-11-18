"""Unit tests for the data preprocessing module."""

import unittest
import numpy as np
import pandas as pd

from src.data_loader import ID_COLUMN, TARGET_COLUMN
from src.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    SKLEARN_AVAILABLE,
    create_preprocessor,
    fit_preprocessor,
    fit_transform_features,
    get_transformed_feature_names,
    transform_features,
    validate_and_clean_features,
)


def _create_synthetic_features(n_rows: int = 6) -> pd.DataFrame:
    """Helper to create a synthetic DataFrame with all 11 feature columns."""
    data = {
        "ApplicantIncome": [5000.0, 3000.0, 4500.0, 6000.0, 2500.0, 7000.0][:n_rows],
        "CoapplicantIncome": [0.0, 1500.0, 2000.0, 0.0, 1200.0, 2500.0][:n_rows],
        "LoanAmount": [120.0, 110.0, 150.0, 200.0, 90.0, 180.0][:n_rows],
        "Loan_Amount_Term": [360.0, 360.0, 360.0, 180.0, 360.0, 360.0][:n_rows],
        "Credit_History": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0][:n_rows],
        "Gender": ["Male", "Female", "Male", "Male", "Female", "Male"][:n_rows],
        "Married": ["No", "Yes", "Yes", "No", "Yes", "Yes"][:n_rows],
        "Dependents": ["0", "1", "2", "3+", "0", "1"][:n_rows],
        "Education": [
            "Graduate",
            "Graduate",
            "Not Graduate",
            "Graduate",
            "Graduate",
            "Not Graduate",
        ][:n_rows],
        "Self_Employed": ["No", "No", "Yes", "No", "No", "Yes"][:n_rows],
        "Property_Area": [
            "Urban",
            "Rural",
            "Semiurban",
            "Urban",
            "Semiurban",
            "Rural",
        ][:n_rows],
    }
    return pd.DataFrame(data)


class TestPreprocessing(unittest.TestCase):
    """Test suite for data preprocessing and transformation pipelines."""

    def test_feature_column_definitions(self):
        """Test feature definitions and consistency."""
        self.assertEqual(len(NUMERICAL_FEATURES), 5)
        self.assertEqual(len(CATEGORICAL_FEATURES), 6)
        self.assertEqual(len(ALL_FEATURES), 11)
        self.assertIn("Dependents", CATEGORICAL_FEATURES)
        self.assertNotIn(ID_COLUMN, ALL_FEATURES)
        self.assertNotIn(TARGET_COLUMN, ALL_FEATURES)

    def test_missing_required_columns_raise_error(self):
        """Test that missing required feature columns raise ValueError."""
        df = _create_synthetic_features()
        corrupted = df.drop(columns=["Credit_History"])
        with self.assertRaises(ValueError) as ctx:
            validate_and_clean_features(corrupted)
        self.assertIn("Credit_History", str(ctx.exception))

    def test_empty_dataframe_raises_error(self):
        """Test that passing an empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            validate_and_clean_features(empty_df)

    def test_invalid_type_raises_type_error(self):
        """Test that non-DataFrame inputs raise TypeError."""
        with self.assertRaises(TypeError):
            validate_and_clean_features("not_a_df")  # type: ignore

    def test_loan_id_and_status_excluded_by_validator(self):
        """Test that Loan_ID and Loan_Status are strictly stripped from features."""
        df = _create_synthetic_features()
        df[ID_COLUMN] = [f"LP00100{i}" for i in range(len(df))]
        df[TARGET_COLUMN] = ["Y", "N", "Y", "Y", "N", "Y"]

        cleaned = validate_and_clean_features(df)
        self.assertNotIn(ID_COLUMN, cleaned.columns)
        self.assertNotIn(TARGET_COLUMN, cleaned.columns)
        self.assertEqual(len(cleaned.columns), 11)
        self.assertListEqual(list(cleaned.columns), ALL_FEATURES)

        # Verify original df was not mutated
        self.assertIn(ID_COLUMN, df.columns)
        self.assertIn(TARGET_COLUMN, df.columns)

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_create_preprocessor(self):
        """Test unfitted ColumnTransformer creation and structure."""
        preprocessor = create_preprocessor()
        self.assertIsNotNone(preprocessor)
        transformer_names = [t[0] for t in preprocessor.transformers]
        self.assertIn("num", transformer_names)
        self.assertIn("cat", transformer_names)

        # Ensure ID and Target are not among transformer columns
        for _, _, cols in preprocessor.transformers:
            self.assertNotIn(ID_COLUMN, cols)
            self.assertNotIn(TARGET_COLUMN, cols)

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_correct_numerical_and_categorical_column_handling(self):
        """Test that transformers map correctly to numerical and categorical features."""
        preprocessor = create_preprocessor()
        transformers_dict = {t[0]: t for t in preprocessor.transformers}

        num_trans = transformers_dict["num"]
        cat_trans = transformers_dict["cat"]

        self.assertListEqual(num_trans[2], NUMERICAL_FEATURES)
        self.assertListEqual(cat_trans[2], CATEGORICAL_FEATURES)

        # Check imputation strategies
        num_imputer = num_trans[1].named_steps["imputer"]
        cat_imputer = cat_trans[1].named_steps["imputer"]
        self.assertEqual(num_imputer.strategy, "median")
        self.assertEqual(cat_imputer.strategy, "most_frequent")

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_missing_numerical_values_imputed(self):
        """Test that missing numerical values are imputed with median."""
        df_train = _create_synthetic_features()
        df_train.loc[0, "LoanAmount"] = np.nan
        df_train.loc[1, "Credit_History"] = np.nan

        preprocessor = fit_preprocessor(df_train)
        transformed = transform_features(preprocessor, df_train)

        # Transformed output should have no NaNs
        self.assertFalse(np.isnan(transformed).any())

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_missing_categorical_values_imputed(self):
        """Test that missing categorical values are imputed with most frequent."""
        df_train = _create_synthetic_features()
        df_train.loc[0, "Gender"] = np.nan
        df_train.loc[2, "Dependents"] = np.nan

        preprocessor = fit_preprocessor(df_train)
        transformed = transform_features(preprocessor, df_train)

        # Transformed output should have no NaNs
        self.assertFalse(np.isnan(transformed).any())

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_successful_fit_and_transformation(self):
        """Test fit_preprocessor, transform_features, and fit_transform_features."""
        df_train = _create_synthetic_features(n_rows=5)

        # Fit preprocessor
        preprocessor = fit_preprocessor(df_train)
        transformed_1 = transform_features(preprocessor, df_train)

        # Fit & transform
        _, transformed_2 = fit_transform_features(df_train)

        self.assertIsInstance(transformed_1, np.ndarray)
        self.assertEqual(transformed_1.shape[0], 5)
        self.assertEqual(transformed_1.ndim, 2)
        np.testing.assert_array_almost_equal(transformed_1, transformed_2)

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_unseen_categories_in_validation_data(self):
        """Test that unseen categorical values do not cause errors (one-hot encodes to zeros)."""
        df_train = _create_synthetic_features(n_rows=4)
        preprocessor = fit_preprocessor(df_train)

        df_val = _create_synthetic_features(n_rows=2)
        # Inject unseen category
        df_val.loc[0, "Property_Area"] = "Island"
        df_val.loc[1, "Education"] = "Doctorate"

        # Should transform without raising an exception
        transformed = transform_features(preprocessor, df_val)
        self.assertEqual(transformed.shape[0], 2)
        self.assertFalse(np.isnan(transformed).any())

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_consistent_transformed_feature_dimensions(self):
        """Test that train and test datasets produce identical column counts after transformation."""
        df_train = _create_synthetic_features(n_rows=5)
        df_test = _create_synthetic_features(n_rows=3)

        preprocessor, train_transformed = fit_transform_features(df_train)
        test_transformed = transform_features(preprocessor, df_test)

        self.assertEqual(train_transformed.shape[1], test_transformed.shape[1])
        self.assertEqual(test_transformed.shape[0], 3)

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_input_dataframe_remains_unchanged(self):
        """Test that input DataFrame is preserved without mutations."""
        df_train = _create_synthetic_features()
        df_copy = df_train.copy(deep=True)

        preprocessor = fit_preprocessor(df_train)
        _ = transform_features(preprocessor, df_train)
        _, _ = fit_transform_features(df_train)

        pd.testing.assert_frame_equal(df_train, df_copy)

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_unfitted_preprocessor_raises_error(self):
        """Test that transforming with an unfitted preprocessor raises ValueError."""
        preprocessor = create_preprocessor()
        df = _create_synthetic_features()
        with self.assertRaises(ValueError):
            transform_features(preprocessor, df)

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is not installed in the environment"
    )
    def test_loan_id_and_status_excluded_during_fit_and_transform(self):
        """Test that preprocessor works when Loan_ID and Loan_Status are present in input."""
        df = _create_synthetic_features()
        df[ID_COLUMN] = [f"LP00100{i}" for i in range(len(df))]
        df[TARGET_COLUMN] = ["Y", "N", "Y", "Y", "N", "Y"]

        preprocessor, transformed = fit_transform_features(df)
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.shape[0], len(df))

        # Check feature names out do not include Loan_ID or Loan_Status
        out_names = get_transformed_feature_names(preprocessor)
        for name in out_names:
            self.assertNotIn(ID_COLUMN, name)
            self.assertNotIn(TARGET_COLUMN, name)


if __name__ == "__main__":
    unittest.main()
