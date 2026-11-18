"""Unit tests for statistical drift detection routines."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.drift_detection import (
    DEFAULT_DRIFT_SHARE_THRESHOLD,
    DEFAULT_P_VALUE_THRESHOLD,
    DriftReport,
    FeatureDriftResult,
    compute_categorical_drift,
    compute_numerical_drift,
    detect_drift,
    load_production_data,
    load_reference_data,
    save_drift_report,
)
from src.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


@pytest.fixture
def synthetic_datasets():
    """Generate paired reference and production datasets deterministically."""
    rng = np.random.RandomState(42)
    n = 250

    ref = pd.DataFrame(
        {
            "ApplicantIncome": rng.normal(5000, 1000, n),
            "CoapplicantIncome": rng.normal(1500, 400, n),
            "LoanAmount": rng.normal(130, 30, n),
            "Loan_Amount_Term": rng.choice([360.0, 180.0], size=n, p=[0.8, 0.2]),
            "Credit_History": rng.choice([1.0, 0.0], size=n, p=[0.85, 0.15]),
            "Gender": rng.choice(["Male", "Female"], size=n, p=[0.8, 0.2]),
            "Married": rng.choice(["Yes", "No"], size=n, p=[0.65, 0.35]),
            "Dependents": rng.choice(["0", "1", "2", "3+"], size=n, p=[0.5, 0.2, 0.2, 0.1]),
            "Education": rng.choice(["Graduate", "Not Graduate"], size=n, p=[0.75, 0.25]),
            "Self_Employed": rng.choice(["No", "Yes"], size=n, p=[0.85, 0.15]),
            "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.4, 0.35, 0.25]),
        }
    )

    # Production dataset without drift (sampled from same distributions with different seed)
    rng_no_drift = np.random.RandomState(99)
    prod_no_drift = pd.DataFrame(
        {
            "ApplicantIncome": rng_no_drift.normal(5000, 1000, n),
            "CoapplicantIncome": rng_no_drift.normal(1500, 400, n),
            "LoanAmount": rng_no_drift.normal(130, 30, n),
            "Loan_Amount_Term": rng_no_drift.choice([360.0, 180.0], size=n, p=[0.8, 0.2]),
            "Credit_History": rng_no_drift.choice([1.0, 0.0], size=n, p=[0.85, 0.15]),
            "Gender": rng_no_drift.choice(["Male", "Female"], size=n, p=[0.8, 0.2]),
            "Married": rng_no_drift.choice(["Yes", "No"], size=n, p=[0.65, 0.35]),
            "Dependents": rng_no_drift.choice(["0", "1", "2", "3+"], size=n, p=[0.5, 0.2, 0.2, 0.1]),
            "Education": rng_no_drift.choice(["Graduate", "Not Graduate"], size=n, p=[0.75, 0.25]),
            "Self_Employed": rng_no_drift.choice(["No", "Yes"], size=n, p=[0.85, 0.15]),
            "Property_Area": rng_no_drift.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.4, 0.35, 0.25]),
        }
    )

    # Production dataset with heavy drift in 3 features:
    # 1. ApplicantIncome (numerical mean shift)
    # 2. Credit_History (numerical distribution shift)
    # 3. Property_Area (categorical distribution change)
    prod_drifted = prod_no_drift.copy()
    prod_drifted["ApplicantIncome"] = rng.normal(12000, 1500, n)
    prod_drifted["Credit_History"] = rng.choice([1.0, 0.0], size=n, p=[0.20, 0.80])
    prod_drifted["Property_Area"] = rng.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.85, 0.10, 0.05])

    return ref, prod_no_drift, prod_drifted


def test_ks_numerical_no_drift():
    """Test KS test flags no drift for two samples from identical distribution."""
    rng = np.random.RandomState(42)
    ref = pd.Series(rng.normal(5000, 800, 300))
    prod = pd.Series(rng.normal(5000, 800, 300))

    result = compute_numerical_drift(
        ref, prod, feature_name="ApplicantIncome", p_value_threshold=0.05
    )
    assert isinstance(result, FeatureDriftResult)
    assert result.feature_name == "ApplicantIncome"
    assert result.feature_type == "numerical"
    assert result.test_name == "Kolmogorov-Smirnov"
    assert result.is_valid is True
    assert result.p_value is not None
    assert result.p_value >= 0.05
    assert result.drift_detected is False
    assert result.status == "NO_DRIFT"


def test_ks_numerical_with_drift():
    """Test KS test correctly detects distributional drift for shifted continuous feature."""
    rng = np.random.RandomState(42)
    ref = pd.Series(rng.normal(5000, 800, 300))
    prod = pd.Series(rng.normal(12000, 1500, 300))

    result = compute_numerical_drift(
        ref, prod, feature_name="ApplicantIncome", p_value_threshold=0.05
    )
    assert result.is_valid is True
    assert result.p_value is not None
    assert result.p_value < 0.05
    assert result.drift_detected is True
    assert result.status == "DRIFT_DETECTED"


def test_chi2_categorical_no_drift():
    """Test Chi-Square test flags no drift for stable categorical proportions."""
    ref = pd.Series(["Urban"] * 100 + ["Rural"] * 50 + ["Semiurban"] * 100)
    prod = pd.Series(["Urban"] * 98 + ["Rural"] * 52 + ["Semiurban"] * 100)

    result = compute_categorical_drift(
        ref, prod, feature_name="Property_Area", p_value_threshold=0.05
    )
    assert isinstance(result, FeatureDriftResult)
    assert result.feature_name == "Property_Area"
    assert result.feature_type == "categorical"
    assert result.test_name == "Chi-Square"
    assert result.is_valid is True
    assert result.p_value is not None
    assert result.p_value >= 0.05
    assert result.drift_detected is False
    assert result.status == "NO_DRIFT"


def test_chi2_categorical_with_drift():
    """Test Chi-Square test detects drift when proportions invert."""
    ref = pd.Series(["Urban"] * 150 + ["Rural"] * 50)
    prod = pd.Series(["Urban"] * 20 + ["Rural"] * 180)

    result = compute_categorical_drift(
        ref, prod, feature_name="Property_Area", p_value_threshold=0.05
    )
    assert result.is_valid is True
    assert result.p_value is not None
    assert result.p_value < 0.05
    assert result.drift_detected is True
    assert result.status == "DRIFT_DETECTED"


def test_missing_and_invalid_values_handled_gracefully():
    """Test that NaNs, None, and unparseable strings are safely handled."""
    # Numerical: mixture of numbers, NaNs, None, and invalid string tokens
    ref_num = pd.Series([100.0, np.nan, 200.0, 300.0, None, "invalid_val"] * 50)
    prod_num = pd.Series([105.0, 195.0, np.nan, 310.0, 150.0, "unknown"] * 50)

    num_result = compute_numerical_drift(ref_num, prod_num, feature_name="LoanAmount")
    assert num_result.is_valid is True
    assert num_result.p_value is not None
    assert not np.isnan(num_result.p_value)

    # Categorical: mixture of strings, None, and np.nan
    ref_cat = pd.Series(["Male", None, "Female", "Male", np.nan] * 50)
    prod_cat = pd.Series(["Male", "Female", np.nan, "Male", "Female"] * 50)
    cat_result = compute_categorical_drift(ref_cat, prod_cat, feature_name="Gender")
    assert cat_result.is_valid is True
    assert cat_result.p_value is not None
    assert not np.isnan(cat_result.p_value)


def test_insufficient_samples_marks_test_skipped():
    """Test that empty or insufficient samples are explicitly marked as skipped."""
    # Numerical with insufficient samples (< 2 valid values)
    ref_empty = pd.Series([100.0])
    prod_empty = pd.Series([100.0])

    num_res = compute_numerical_drift(ref_empty, prod_empty, feature_name="LoanAmount")
    assert num_res.is_valid is False
    assert num_res.drift_detected is False
    assert num_res.statistic is None
    assert num_res.p_value is None
    assert num_res.status == "SKIPPED_INSUFFICIENT_DATA"

    # Categorical with insufficient observations
    ref_cat_empty = pd.Series(["Male"])
    prod_cat_empty = pd.Series(["Male"])

    cat_res = compute_categorical_drift(ref_cat_empty, prod_cat_empty, feature_name="Gender")
    assert cat_res.is_valid is False
    assert cat_res.drift_detected is False
    assert cat_res.statistic is None
    assert cat_res.p_value is None
    assert cat_res.status in ("SKIPPED_INSUFFICIENT_DATA", "SKIPPED_CONSTANT_COLUMN")


def test_constant_columns_handling():
    """Test constant feature distributions for numerical and categorical types."""
    # Identical constant numerical: variance is zero in both, no drift
    ref_const = pd.Series([1.0] * 100)
    prod_const = pd.Series([1.0] * 100)
    num_res_identical = compute_numerical_drift(ref_const, prod_const, feature_name="Term")
    assert num_res_identical.is_valid is True
    assert num_res_identical.drift_detected is False
    assert num_res_identical.p_value == 1.0

    # Shifted constant numerical: point mass changed completely
    prod_shifted_const = pd.Series([2.0] * 100)
    num_res_shifted = compute_numerical_drift(ref_const, prod_shifted_const, feature_name="Term")
    assert num_res_shifted.is_valid is True
    assert num_res_shifted.drift_detected is True
    assert num_res_shifted.p_value < 0.05

    # Constant categorical: only 1 category across both datasets cannot form a 2x2 table
    ref_cat_const = pd.Series(["Graduate"] * 100)
    prod_cat_const = pd.Series(["Graduate"] * 100)
    cat_res_const = compute_categorical_drift(ref_cat_const, prod_cat_const, feature_name="Education")
    assert cat_res_const.is_valid is False
    assert cat_res_const.drift_detected is False
    assert cat_res_const.status == "SKIPPED_CONSTANT_COLUMN"


def test_expected_frequencies_below_1_skipped():
    """Test categorical distribution where cell expected frequencies are < 1.0."""
    ref_small = pd.Series(["A"] * 10 + ["B"] * 1)
    prod_small = pd.Series(["A"] * 10 + ["C"] * 1)

    cat_res = compute_categorical_drift(ref_small, prod_small, feature_name="SmallCat")
    assert cat_res.is_valid is False
    assert cat_res.drift_detected is False
    assert cat_res.statistic is None
    assert cat_res.p_value is None
    assert cat_res.status == "SKIPPED_UNSUITABLE_FREQUENCIES"
    assert "Expected frequencies unsuitable" in cat_res.message


def test_more_than_20_percent_expected_below_5_skipped():
    """Test Cochran rule: skip test when more than 20% of expected cell frequencies are < 5.0."""
    # 2x3 contingency table where min expected is >= 1.0, but 2 of 6 cells (33.3% > 20%) are < 5.0
    # ref: A=20, B=20, C=4 (sum=44); prod: A=20, B=20, C=4 (sum=44).
    # expected for C is (44 * 8) / 88 = 4.0 >= 1.0.
    ref = pd.Series(["A"] * 20 + ["B"] * 20 + ["C"] * 4)
    prod = pd.Series(["A"] * 20 + ["B"] * 20 + ["C"] * 4)

    cat_res = compute_categorical_drift(ref, prod, feature_name="CategoryTest")
    assert cat_res.is_valid is False
    assert cat_res.drift_detected is False
    assert cat_res.status == "SKIPPED_UNSUITABLE_FREQUENCIES"
    assert "below 5.0" in cat_res.message


def test_new_unseen_categories():
    """Test categorical comparison when a new category appears only in production data."""
    ref = pd.Series(["Urban"] * 100 + ["Rural"] * 100)
    prod = pd.Series(["Urban"] * 80 + ["Rural"] * 80 + ["Semiurban"] * 40)

    result = compute_categorical_drift(ref, prod, feature_name="Property_Area")
    assert result.is_valid is True
    assert result.drift_detected is True
    assert result.p_value is not None
    assert result.p_value < 0.05


def test_detect_drift_no_drift_dataset(synthetic_datasets):
    """Test detect_drift on stable baseline vs production returns NO DRIFT DETECTED."""
    ref, prod_no_drift, _ = synthetic_datasets
    report = detect_drift(ref, prod_no_drift)

    assert isinstance(report, DriftReport)
    assert report.total_features == 11
    assert report.valid_count == 11
    assert report.drift_percentage < 20.0
    assert report.is_drift_detected is False
    assert report.status == "NO DRIFT DETECTED"


def test_detect_drift_drifted_dataset(synthetic_datasets):
    """Test detect_drift on drifted dataset flags overall drift and identifies drifted features."""
    ref, _, prod_drifted = synthetic_datasets
    report = detect_drift(ref, prod_drifted)

    assert report.total_features == 11
    assert "ApplicantIncome" in report.drifted_features
    assert "Credit_History" in report.drifted_features
    assert "Property_Area" in report.drifted_features
    assert report.drift_count >= 3
    # 3 / 11 = 27.27% >= 20.0%
    assert report.drift_percentage >= 20.0
    assert report.is_drift_detected is True
    assert report.status == "DRIFT DETECTED"


def test_all_features_skipped_status_inconclusive():
    """Test that when all feature tests are skipped, overall status is INCONCLUSIVE."""
    # Data with only 1 sample per column (insufficient for KS and Chi-Square)
    ref = pd.DataFrame({"ApplicantIncome": [5000.0], "Gender": ["Male"]})
    prod = pd.DataFrame({"ApplicantIncome": [5000.0], "Gender": ["Male"]})

    report = detect_drift(
        ref, prod, numerical_features=["ApplicantIncome"], categorical_features=["Gender"]
    )
    assert report.total_features == 2
    assert report.valid_count == 0
    assert len(report.skipped_features) == 2
    assert report.is_drift_detected is False
    assert report.status == "INCONCLUSIVE"

    # Verify report explicitly documents inconclusive status
    text = report.generate_report()
    assert "Status: INCONCLUSIVE" in text
    assert "All feature tests were skipped" in text


def test_overall_drift_share_calculated_only_on_valid_features():
    """Test that skipped/invalid feature tests do not dilute the drift calculation."""
    # Create dataset with 3 valid features and 1 skipped constant feature
    ref = pd.DataFrame(
        {
            "ApplicantIncome": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0] * 20,
            "LoanAmount": [100.0, 150.0, 200.0, 250.0, 300.0] * 20,
            "Property_Area": ["Urban"] * 50 + ["Rural"] * 50,
            "Education": ["Graduate"] * 100,  # Constant: will be skipped for Chi-Square
        }
    )
    # Drift ApplicantIncome only
    prod = pd.DataFrame(
        {
            "ApplicantIncome": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0] * 20,
            "LoanAmount": [100.0, 150.0, 200.0, 250.0, 300.0] * 20,
            "Property_Area": ["Urban"] * 50 + ["Rural"] * 50,
            "Education": ["Graduate"] * 100,
        }
    )

    report = detect_drift(
        ref,
        prod,
        numerical_features=["ApplicantIncome", "LoanAmount"],
        categorical_features=["Property_Area", "Education"],
    )
    assert report.total_features == 4
    assert report.valid_count == 3
    assert len(report.skipped_features) == 1
    assert "Education" in report.skipped_features
    assert report.drift_count == 1
    # 1 drifted out of 3 valid = 33.33% (NOT 1 / 4 = 25.0%)
    assert report.drift_percentage == pytest.approx(33.33, abs=0.01)


def test_valid_fraction_and_percentage_thresholds(synthetic_datasets):
    """Test both fraction (0.80) and percentage (80.0) formats for drift_share_threshold."""
    ref, _, prod_drifted = synthetic_datasets

    # Fraction threshold
    report_frac = detect_drift(ref, prod_drifted, drift_share_threshold=0.80)
    assert report_frac.is_drift_detected is False
    assert report_frac.status == "NO DRIFT DETECTED"

    # Percentage threshold (80.0%)
    report_pct = detect_drift(ref, prod_drifted, drift_share_threshold=80.0)
    assert report_pct.is_drift_detected is False
    assert report_pct.status == "NO DRIFT DETECTED"

    # Low percentage threshold (10.0%)
    report_low_pct = detect_drift(ref, prod_drifted, drift_share_threshold=10.0)
    assert report_low_pct.is_drift_detected is True
    assert report_low_pct.status == "DRIFT DETECTED"


def test_invalid_p_value_thresholds_raise_error(synthetic_datasets):
    """Test invalid significance thresholds raise appropriate ValueError or TypeError."""
    ref, prod, _ = synthetic_datasets

    invalid_p_values = [0.0, 1.0, -0.05, 1.5, float("nan"), float("inf")]
    for p_val in invalid_p_values:
        with pytest.raises(ValueError, match="p_value_threshold"):
            detect_drift(ref, prod, p_value_threshold=p_val)

    with pytest.raises(TypeError, match="p_value_threshold"):
        detect_drift(ref, prod, p_value_threshold="0.05")  # type: ignore

    with pytest.raises(TypeError, match="p_value_threshold"):
        detect_drift(ref, prod, p_value_threshold=True)  # type: ignore


def test_invalid_drift_share_thresholds_raise_error(synthetic_datasets):
    """Test invalid drift share thresholds raise appropriate ValueError or TypeError."""
    ref, prod, _ = synthetic_datasets

    invalid_shares = [-0.1, 105.0, -50.0, float("nan"), float("inf")]
    for share in invalid_shares:
        with pytest.raises(ValueError, match="drift_share_threshold"):
            detect_drift(ref, prod, drift_share_threshold=share)

    with pytest.raises(TypeError, match="drift_share_threshold"):
        detect_drift(ref, prod, drift_share_threshold="20%")  # type: ignore

    with pytest.raises(TypeError, match="drift_share_threshold"):
        detect_drift(ref, prod, drift_share_threshold=False)  # type: ignore


def test_input_dataframe_immutability(synthetic_datasets):
    """Test that detect_drift does not mutate input DataFrames."""
    ref, _, prod_drifted = synthetic_datasets
    ref_copy = ref.copy()
    prod_copy = prod_drifted.copy()

    _ = detect_drift(ref, prod_drifted)

    pd.testing.assert_frame_equal(ref, ref_copy)
    pd.testing.assert_frame_equal(prod_drifted, prod_copy)


def test_report_formatting_and_serialization(synthetic_datasets, tmp_path):
    """Test text report generation, JSON serialization, and file saving."""
    ref, _, prod_drifted = synthetic_datasets
    report = detect_drift(ref, prod_drifted)

    # 1. Text report formatting
    text_report = report.generate_report()
    assert "Drift Detection Report" in text_report
    assert "ApplicantIncome: Drift Detected" in text_report
    assert "Credit_History: Drift Detected" in text_report
    assert "Property_Area: Drift Detected" in text_report
    assert f"Drifted Features: {report.drift_count}" in text_report
    assert f"Total Features: {report.total_features}" in text_report
    assert "Drift Percentage:" in text_report
    assert "Status: DRIFT DETECTED" in text_report

    # 2. to_dict() serialization
    report_dict = report.to_dict()
    assert isinstance(report_dict, dict)
    assert report_dict["status"] == "DRIFT DETECTED"
    assert report_dict["is_drift_detected"] is True
    assert report_dict["drift_count"] >= 3
    assert "ApplicantIncome" in report_dict["drifted_features"]
    assert "feature_results" in report_dict

    # 3. File saving (JSON and text)
    txt_path = tmp_path / "drift_report.txt"
    json_path = tmp_path / "drift_report.json"

    save_drift_report(report, txt_path)
    assert txt_path.exists()
    assert "Drift Detection Report" in txt_path.read_text(encoding="utf-8")

    save_drift_report(report, json_path)
    assert json_path.exists()
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["status"] == "DRIFT DETECTED"


def test_credit_history_treated_as_numerical():
    """Verify Credit_History is classified as numerical and evaluated with KS test."""
    assert "Credit_History" in NUMERICAL_FEATURES
    assert "Credit_History" not in CATEGORICAL_FEATURES

    rng = np.random.RandomState(42)
    ref = pd.DataFrame({"Credit_History": rng.choice([1.0, 0.0], size=100, p=[0.9, 0.1])})
    prod = pd.DataFrame({"Credit_History": rng.choice([1.0, 0.0], size=100, p=[0.1, 0.9])})

    report = detect_drift(ref, prod)
    res = report.feature_results["Credit_History"]
    assert res.feature_type == "numerical"
    assert res.test_name == "Kolmogorov-Smirnov"
    assert res.is_valid is True
    assert res.drift_detected is True


def test_missing_reference_file_raises_error():
    """Test load_reference_data raises FileNotFoundError when path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_reference_data("non_existent_reference_file_xyz.csv")


def test_missing_production_file_raises_error():
    """Test load_production_data raises FileNotFoundError when path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_production_data("non_existent_production_file_xyz.csv")


def test_no_common_features_raises_value_error():
    """Test detect_drift raises ValueError when no common columns exist."""
    df1 = pd.DataFrame({"ColA": [1, 2, 3]})
    df2 = pd.DataFrame({"ColB": [4, 5, 6]})
    with pytest.raises(ValueError, match="No common feature columns found"):
        detect_drift(df1, df2)
