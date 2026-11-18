"""Statistical tests and data drift detection module.

This module monitors distribution drift between a reference baseline dataset
and incoming production inference data using SciPy statistical tests:
- Numerical features: Two-sample Kolmogorov-Smirnov (KS) test
- Categorical features: Pearson's Chi-Square test of independence

Feature definitions are aligned with src/preprocessing.py:
- NUMERICAL_FEATURES (5): ApplicantIncome, CoapplicantIncome, LoanAmount,
  Loan_Amount_Term, Credit_History
- CATEGORICAL_FEATURES (6): Gender, Married, Dependents, Education,
  Self_Employed, Property_Area
"""

from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.config import PROJECT_ROOT, RAW_DATA_DIR, REFERENCE_DATA_DIR
from src.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

logger = logging.getLogger(__name__)

# Default reference and production dataset paths
DEFAULT_REFERENCE_PATH: Path = REFERENCE_DATA_DIR / "reference.csv"
DEFAULT_PRODUCTION_PATH: Path = RAW_DATA_DIR / "test.csv"
DEFAULT_TRAIN_PATH: Path = RAW_DATA_DIR / "train.csv"

# Academic default thresholds
DEFAULT_P_VALUE_THRESHOLD: float = 0.05
DEFAULT_DRIFT_SHARE_THRESHOLD: float = 0.20  # 20% of valid monitored features


def _validate_p_value_threshold(p_value_threshold: float) -> float:
    """Validate that p_value_threshold is a valid probability strictly in (0, 1)."""
    if isinstance(p_value_threshold, bool) or not isinstance(p_value_threshold, (int, float)):
        raise TypeError(
            f"p_value_threshold must be a real number, got {type(p_value_threshold).__name__}."
        )
    if math.isnan(p_value_threshold) or math.isinf(p_value_threshold):
        raise ValueError(
            f"p_value_threshold cannot be NaN or infinite, got {p_value_threshold}."
        )
    val = float(p_value_threshold)
    if not (0.0 < val < 1.0):
        raise ValueError(
            f"p_value_threshold must be strictly greater than 0 and less than 1, got {p_value_threshold}."
        )
    return val


def _validate_thresholds(
    p_value_threshold: float,
    drift_share_threshold: float,
) -> Tuple[float, float]:
    """Validate and normalize significance and drift-share thresholds.

    Parameters
    ----------
    p_value_threshold : float
        Statistical significance level alpha (must be strictly in (0, 1)).
    drift_share_threshold : float
        Threshold for dataset drift fraction or percentage (must be in [0, 100]).

    Returns
    -------
    tuple[float, float]
        Validated p_value_threshold and normalized drift_share_threshold (as fraction in [0, 1]).
    """
    p_val = _validate_p_value_threshold(p_value_threshold)

    if isinstance(drift_share_threshold, bool) or not isinstance(drift_share_threshold, (int, float)):
        raise TypeError(
            f"drift_share_threshold must be a real number, got {type(drift_share_threshold).__name__}."
        )
    if math.isnan(drift_share_threshold) or math.isinf(drift_share_threshold):
        raise ValueError(
            f"drift_share_threshold cannot be NaN or infinite, got {drift_share_threshold}."
        )
    share = float(drift_share_threshold)
    if not (0.0 <= share <= 100.0):
        raise ValueError(
            f"drift_share_threshold must be between 0 and 100 inclusive, got {drift_share_threshold}."
        )

    # Normalize: values in [0, 1] are treated as fractions; values in (1, 100] are treated as percentages
    normalized_share = share if share <= 1.0 else share / 100.0
    return p_val, normalized_share


@dataclass
class FeatureDriftResult:
    """Stores drift test results for an individual feature."""

    feature_name: str
    feature_type: str  # 'numerical' or 'categorical'
    test_name: str  # 'Kolmogorov-Smirnov' or 'Chi-Square'
    statistic: Optional[float]
    p_value: Optional[float]
    drift_detected: bool
    is_valid: bool = True
    status: str = "NO_DRIFT"  # 'DRIFT_DETECTED', 'NO_DRIFT', 'SKIPPED_INSUFFICIENT_DATA', 'SKIPPED_UNSUITABLE_FREQUENCIES', 'SKIPPED_CONSTANT_COLUMN'
    message: Optional[str] = None
    threshold: float = DEFAULT_P_VALUE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4) if self.statistic is not None else None,
            "p_value": round(self.p_value, 5) if self.p_value is not None else None,
            "drift_detected": self.drift_detected,
            "is_valid": self.is_valid,
            "status": self.status,
            "message": self.message,
            "threshold": self.threshold,
        }


@dataclass
class DriftReport:
    """Encapsulates the complete dataset drift detection summary and report."""

    feature_results: Dict[str, FeatureDriftResult] = field(default_factory=dict)
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD

    @property
    def total_features(self) -> int:
        """Total number of monitored features evaluated."""
        return len(self.feature_results)

    @property
    def valid_features(self) -> List[str]:
        """List of feature names where statistical tests were executed validly."""
        return [
            name
            for name, res in self.feature_results.items()
            if res.is_valid
        ]

    @property
    def valid_count(self) -> int:
        """Number of features with valid statistical test execution."""
        return len(self.valid_features)

    @property
    def drifted_features(self) -> List[str]:
        """List of feature names where statistically significant drift was detected."""
        return [
            name
            for name, res in self.feature_results.items()
            if res.is_valid and res.drift_detected
        ]

    @property
    def drift_count(self) -> int:
        """Number of features exhibiting statistically significant drift."""
        return len(self.drifted_features)

    @property
    def skipped_features(self) -> List[str]:
        """List of feature names where statistical tests could not be reliably executed."""
        return [
            name
            for name, res in self.feature_results.items()
            if not res.is_valid
        ]

    @property
    def drift_percentage(self) -> float:
        """Percentage of VALID monitored features exhibiting drift (0.0 to 100.0).

        Note: Excludes skipped or invalid tests from the denominator to avoid
        diluting drift signals or making false claims of stability.
        """
        if self.valid_count == 0:
            return 0.0
        return round((self.drift_count / self.valid_count) * 100.0, 2)

    @property
    def is_drift_detected(self) -> bool:
        """Whether overall dataset drift exceeds the configured share threshold.

        Threshold can be specified as a fraction (e.g. 0.20) or percentage (e.g. 20.0).
        Evaluated strictly over valid feature tests.
        Returns False when valid_count == 0.
        Note: When valid_count == 0, check self.status == 'INCONCLUSIVE' to distinguish
        the absence of drift from the complete absence of valid feature tests.
        """
        if self.valid_count == 0:
            return False

        effective_threshold = (
            self.drift_share_threshold
            if self.drift_share_threshold <= 1.0
            else self.drift_share_threshold / 100.0
        )
        return (self.drift_count / self.valid_count) >= effective_threshold

    @property
    def status(self) -> str:
        """Human-readable overall status string.

        Returns:
            'INCONCLUSIVE': If valid_count == 0 (no features could be evaluated).
            'DRIFT DETECTED': If is_drift_detected is True.
            'NO DRIFT DETECTED': If valid tests exist and drift share is below threshold.
        """
        if self.valid_count == 0:
            return "INCONCLUSIVE"
        return "DRIFT DETECTED" if self.is_drift_detected else "NO DRIFT DETECTED"

    def generate_report(self) -> str:
        """Generate human-readable text report formatted according to project specification.

        Format:
        Drift Detection Report

        ApplicantIncome: Drift Detected
        LoanAmount: No Drift
        ...

        Drifted Features: 3
        Total Features: 11
        Valid Features: 11
        Drift Percentage: 27.3%

        Status: DRIFT DETECTED
        """
        lines = ["Drift Detection Report", ""]

        for feature_name, res in self.feature_results.items():
            if not res.is_valid:
                status_text = f"Skipped ({res.message or 'Insufficient data'})"
            elif res.drift_detected:
                status_text = "Drift Detected"
            else:
                status_text = "No Drift"
            lines.append(f"{feature_name}: {status_text}")

        lines.append("")
        lines.append(f"Drifted Features: {self.drift_count}")
        lines.append(f"Total Features: {self.total_features}")
        if self.skipped_features:
            lines.append(f"Valid Features: {self.valid_count}")
        pct_str = (
            f"{int(self.drift_percentage)}%"
            if self.drift_percentage.is_integer()
            else f"{self.drift_percentage:.1f}%"
        )
        lines.append(f"Drift Percentage: {pct_str}")
        lines.append("")
        lines.append(f"Status: {self.status}")
        if self.valid_count == 0:
            lines.append("Warning: All feature tests were skipped. No features could be evaluated.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to structured dictionary for metrics tracking and APIs."""
        return {
            "status": self.status,
            "is_drift_detected": self.is_drift_detected,
            "drift_percentage": self.drift_percentage,
            "drift_count": self.drift_count,
            "valid_count": self.valid_count,
            "total_features": self.total_features,
            "drifted_features": self.drifted_features,
            "skipped_features": self.skipped_features,
            "p_value_threshold": self.p_value_threshold,
            "drift_share_threshold": self.drift_share_threshold,
            "feature_results": {
                name: res.to_dict() for name, res in self.feature_results.items()
            },
        }


def compute_numerical_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str = "numerical_feature",
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
) -> FeatureDriftResult:
    """Perform Two-sample Kolmogorov-Smirnov (KS) test on numerical feature distributions.

    Parameters
    ----------
    reference : pd.Series
        Baseline reference feature values.
    current : pd.Series
        Incoming production feature values.
    feature_name : str
        Name of the feature being tested.
    p_value_threshold : float, default=0.05
        Significance level below which drift is flagged (must be strictly in (0, 1)).

    Returns
    -------
    FeatureDriftResult
        Structured result containing test statistic, p-value, validity, and drift boolean.
    """
    valid_p_thresh = _validate_p_value_threshold(p_value_threshold)

    ref_clean = pd.to_numeric(reference, errors="coerce").dropna()
    curr_clean = pd.to_numeric(current, errors="coerce").dropna()

    # Minimum sample requirement: KS 2-sample test requires at least 2 observations
    if len(ref_clean) < 2 or len(curr_clean) < 2:
        msg = (
            f"Insufficient valid samples (ref: {len(ref_clean)}, "
            f"curr: {len(curr_clean)}; minimum 2 required)."
        )
        logger.warning("Feature '%s': %s", feature_name, msg)
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            test_name="Kolmogorov-Smirnov",
            statistic=None,
            p_value=None,
            drift_detected=False,
            is_valid=False,
            status="SKIPPED_INSUFFICIENT_DATA",
            message=msg,
            threshold=valid_p_thresh,
        )

    try:
        res = ks_2samp(ref_clean, curr_clean)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
        drift = bool(p_val < valid_p_thresh)

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            test_name="Kolmogorov-Smirnov",
            statistic=stat,
            p_value=p_val,
            drift_detected=drift,
            is_valid=True,
            status="DRIFT_DETECTED" if drift else "NO_DRIFT",
            message=None,
            threshold=valid_p_thresh,
        )
    except Exception as exc:
        logger.warning(
            "Kolmogorov-Smirnov test failed for feature '%s': %s",
            feature_name,
            exc,
        )
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            test_name="Kolmogorov-Smirnov",
            statistic=None,
            p_value=None,
            drift_detected=False,
            is_valid=False,
            status="ERROR",
            message=str(exc),
            threshold=valid_p_thresh,
        )


def compute_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str = "categorical_feature",
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
) -> FeatureDriftResult:
    """Perform Chi-Square test of independence on categorical feature distributions.

    Constructs a 2-way contingency table across the union of observed categories.
    Validates expected cell frequencies according to Cochran's criteria (min expected >= 1
    and no more than 20% of cells < 5) to prevent unreliable asymptotic approximations.

    Parameters
    ----------
    reference : pd.Series
        Baseline reference feature values.
    current : pd.Series
        Incoming production feature values.
    feature_name : str
        Name of the feature being tested.
    p_value_threshold : float, default=0.05
        Significance level below which drift is flagged (must be strictly in (0, 1)).

    Returns
    -------
    FeatureDriftResult
        Structured result containing test statistic, p-value, validity, and drift boolean.
    """
    valid_p_thresh = _validate_p_value_threshold(p_value_threshold)

    ref_clean = reference.dropna().astype(str)
    curr_clean = current.dropna().astype(str)

    # Minimum sample check
    if len(ref_clean) < 2 or len(curr_clean) < 2:
        msg = (
            f"Insufficient valid observations (ref: {len(ref_clean)}, "
            f"curr: {len(curr_clean)}; minimum 2 required)."
        )
        logger.warning("Feature '%s': %s", feature_name, msg)
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square",
            statistic=None,
            p_value=None,
            drift_detected=False,
            is_valid=False,
            status="SKIPPED_INSUFFICIENT_DATA",
            message=msg,
            threshold=valid_p_thresh,
        )

    ref_counts = ref_clean.value_counts()
    curr_counts = curr_clean.value_counts()
    all_categories = sorted(set(ref_counts.index).union(set(curr_counts.index)))

    # A 2xK contingency table requires at least 2 distinct categories overall
    if len(all_categories) < 2:
        msg = (
            "Only one distinct category observed across both datasets; "
            "Chi-Square test requires at least 2 categories."
        )
        logger.warning("Feature '%s': %s", feature_name, msg)
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square",
            statistic=None,
            p_value=None,
            drift_detected=False,
            is_valid=False,
            status="SKIPPED_CONSTANT_COLUMN",
            message=msg,
            threshold=valid_p_thresh,
        )

    ref_freqs = [ref_counts.get(cat, 0) for cat in all_categories]
    curr_freqs = [curr_counts.get(cat, 0) for cat in all_categories]

    contingency_table = np.array([ref_freqs, curr_freqs])

    # Check suitability of expected frequencies
    row_sums = contingency_table.sum(axis=1)
    col_sums = contingency_table.sum(axis=0)
    total_count = contingency_table.sum()

    if total_count == 0 or np.any(row_sums == 0) or np.any(col_sums == 0):
        msg = "Contingency table has empty margins; test cannot be computed."
        logger.warning("Feature '%s': %s", feature_name, msg)
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square",
            statistic=None,
            p_value=None,
            drift_detected=False,
            is_valid=False,
            status="SKIPPED_INSUFFICIENT_DATA",
            message=msg,
            threshold=valid_p_thresh,
        )

    expected = np.outer(row_sums, col_sums) / total_count
    min_exp = float(np.min(expected))
    pct_below_5 = float(np.mean(expected < 5.0) * 100.0)

    # Cochran's criteria: no expected cell below 1, and no more than 20% of cells below 5
    if min_exp < 1.0 or pct_below_5 > 20.0:
        msg = (
            f"Expected frequencies unsuitable for Chi-Square test: "
            f"minimum expected frequency {min_exp:.2f} (< 1.0) or "
            f"{pct_below_5:.1f}% of cells below 5.0 (> 20.0% threshold)."
        )
        logger.warning("Feature '%s': %s", feature_name, msg)
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square",
            statistic=None,
            p_value=None,
            drift_detected=False,
            is_valid=False,
            status="SKIPPED_UNSUITABLE_FREQUENCIES",
            message=msg,
            threshold=valid_p_thresh,
        )

    try:
        chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
        stat = float(chi2_stat)
        p_val = float(p_val)
        drift = bool(p_val < valid_p_thresh)

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square",
            statistic=stat,
            p_value=p_val,
            drift_detected=drift,
            is_valid=True,
            status="DRIFT_DETECTED" if drift else "NO_DRIFT",
            message=None,
            threshold=valid_p_thresh,
        )
    except Exception as exc:
        logger.warning(
            "Chi-Square test failed for feature '%s': %s", feature_name, exc
        )
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square",
            statistic=None,
            p_value=None,
            drift_detected=False,
            is_valid=False,
            status="ERROR",
            message=str(exc),
            threshold=valid_p_thresh,
        )


def load_reference_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load baseline reference data from CSV without mutating source files.

    Parameters
    ----------
    filepath : str | Path | None
        Path to reference dataset. If None, checks data/reference/reference.csv,
        then falls back to data/raw/train.csv.

    Returns
    -------
    pd.DataFrame
        Loaded reference dataset.

    Raises
    ------
    FileNotFoundError
        If the specified or default file does not exist.
    """
    if filepath is not None:
        target_path = Path(filepath)
        if target_path.exists():
            return pd.read_csv(target_path)
        if target_path == DEFAULT_REFERENCE_PATH and DEFAULT_TRAIN_PATH.exists():
            return pd.read_csv(DEFAULT_TRAIN_PATH)
        raise FileNotFoundError(
            f"Reference dataset not found at '{target_path}'."
        )

    # Check default candidates in priority order
    for candidate in [DEFAULT_REFERENCE_PATH, DEFAULT_TRAIN_PATH]:
        if candidate.exists():
            return pd.read_csv(candidate)

    raise FileNotFoundError(
        f"Reference dataset not found at '{DEFAULT_REFERENCE_PATH}' or '{DEFAULT_TRAIN_PATH}'. "
        "Please provide a reference DataFrame or place baseline CSV in data/reference/."
    )


def load_production_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load incoming production-like data from CSV without mutating source files.

    Parameters
    ----------
    filepath : str | Path | None
        Path to production dataset. Defaults to data/raw/test.csv.

    Returns
    -------
    pd.DataFrame
        Loaded production dataset.

    Raises
    ------
    FileNotFoundError
        If the specified or default file does not exist.
    """
    target_path = Path(filepath) if filepath else DEFAULT_PRODUCTION_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Production dataset not found at '{target_path}'. "
            "Please provide a production DataFrame or place CSV in data/raw/test.csv."
        )
    return pd.read_csv(target_path)


def detect_drift(
    reference_data: Union[pd.DataFrame, str, Path],
    production_data: Union[pd.DataFrame, str, Path],
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD,
    record_metrics: bool = False,
    metrics_tracker: Optional[Any] = None,
) -> DriftReport:
    """Execute end-to-end data drift detection comparing reference vs production datasets.

    Non-destructively evaluates numerical features via two-sample Kolmogorov-Smirnov test
    and categorical features via Pearson's Chi-Square test of independence.

    Parameters
    ----------
    reference_data : pd.DataFrame | str | Path
        Reference baseline dataset (DataFrame or CSV path).
    production_data : pd.DataFrame | str | Path
        Incoming production-like dataset (DataFrame or CSV path).
    numerical_features : list[str] | None
        Numerical feature columns to test. If None, uses NUMERICAL_FEATURES from src.preprocessing.
    categorical_features : list[str] | None
        Categorical feature columns to test. If None, uses CATEGORICAL_FEATURES from src.preprocessing.
    p_value_threshold : float, default=0.05
        Statistical significance p-value threshold for feature-level drift (strictly in (0, 1)).
    drift_share_threshold : float, default=0.20
        Fraction [0, 1] or percentage [0, 100] of drifted valid features required
        to flag overall dataset drift (default=0.20 or 20%).
    record_metrics : bool, default=False
        Whether to record drift evaluation outcome to the metrics store.
    metrics_tracker : Any | None, default=None
        Custom MetricsStore instance to record event into (uses global singleton if None).

    Returns
    -------
    DriftReport
        Comprehensive drift report containing individual feature test results,
        drift percentage, valid count, and overall status.
    """
    valid_p_val, valid_norm_share = _validate_thresholds(
        p_value_threshold, drift_share_threshold
    )

    ref_df = (
        load_reference_data(reference_data)
        if isinstance(reference_data, (str, Path))
        else reference_data.copy()
    )
    prod_df = (
        load_production_data(production_data)
        if isinstance(production_data, (str, Path))
        else production_data.copy()
    )

    # Determine numerical and categorical feature sets
    if numerical_features is None and categorical_features is None:
        common_cols = [c for c in ref_df.columns if c in prod_df.columns]
        num_cols = [c for c in NUMERICAL_FEATURES if c in common_cols]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in common_cols]

        # Fallback to inference if canonical columns are not present
        if not num_cols and not cat_cols:
            for col in common_cols:
                if col in ("Loan_ID", "Loan_Status", "loan_id", "loan_status"):
                    continue
                if pd.api.types.is_numeric_dtype(ref_df[col]) and ref_df[col].nunique() > 5:
                    num_cols.append(col)
                else:
                    cat_cols.append(col)
    else:
        num_cols = [
            c
            for c in (numerical_features or [])
            if c in ref_df.columns and c in prod_df.columns
        ]
        cat_cols = [
            c
            for c in (categorical_features or [])
            if c in ref_df.columns and c in prod_df.columns
        ]

    if not num_cols and not cat_cols:
        raise ValueError(
            "No common feature columns found between reference and production datasets."
        )

    feature_results: Dict[str, FeatureDriftResult] = {}

    # 1. Run Kolmogorov-Smirnov test for numerical features
    for col in num_cols:
        res = compute_numerical_drift(
            reference=ref_df[col],
            current=prod_df[col],
            feature_name=col,
            p_value_threshold=valid_p_val,
        )
        feature_results[col] = res

    # 2. Run Chi-Square test for categorical features
    for col in cat_cols:
        res = compute_categorical_drift(
            reference=ref_df[col],
            current=prod_df[col],
            feature_name=col,
            p_value_threshold=valid_p_val,
        )
        feature_results[col] = res

    report = DriftReport(
        feature_results=feature_results,
        p_value_threshold=valid_p_val,
        drift_share_threshold=valid_norm_share,
    )

    if record_metrics:
        try:
            if metrics_tracker is None:
                from monitoring.metrics import metrics_tracker as default_tracker

                tracker = default_tracker
            else:
                tracker = metrics_tracker

            tracker.record_drift_event(
                is_drift_detected=report.is_drift_detected,
                drift_percentage=report.drift_percentage,
                drifted_features=report.drifted_features,
                status=report.status,
            )
        except Exception as exc:
            logger.error("Failed to record drift event to metrics store: %s", exc)

    return report


def run_drift_detection(
    reference_data: Optional[Union[pd.DataFrame, str, Path]] = None,
    production_data: Optional[Union[pd.DataFrame, str, Path]] = None,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD,
    save_report_path: Optional[Union[str, Path]] = None,
    record_metrics: bool = True,
    metrics_tracker: Optional[Any] = None,
) -> DriftReport:
    """Execute workflow-level drift detection and record metrics to central store.

    Resolves default reference (data/reference/reference.csv or data/raw/train.csv)
    and production (data/raw/test.csv) datasets, executes detect_drift(), optionally
    saves the report artifact to disk, and records the drift event to metrics_tracker.
    """
    if reference_data is not None:
        ref = reference_data
    elif DEFAULT_REFERENCE_PATH.exists():
        ref = DEFAULT_REFERENCE_PATH
    else:
        ref = DEFAULT_TRAIN_PATH
    prod = production_data if production_data is not None else DEFAULT_PRODUCTION_PATH

    report = detect_drift(
        reference_data=ref,
        production_data=prod,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        p_value_threshold=p_value_threshold,
        drift_share_threshold=drift_share_threshold,
        record_metrics=record_metrics,
        metrics_tracker=metrics_tracker,
    )

    if save_report_path:
        save_drift_report(report, save_report_path)

    return report


def save_drift_report(
    report: Union[DriftReport, Dict[str, Any]],
    filepath: Union[str, Path],
) -> Path:
    """Save drift report to text or JSON file.

    Parameters
    ----------
    report : DriftReport | dict
        Drift report object or serialized dictionary.
    filepath : str | Path
        Destination path. If ends with .json, saves JSON; else saves text report.

    Returns
    -------
    Path
        Path to the saved report file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        data = report.to_dict() if isinstance(report, DriftReport) else report
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        text = (
            report.generate_report()
            if isinstance(report, DriftReport)
            else str(report)
        )
        path.write_text(text, encoding="utf-8")
    return path


# Convenience aliases
test_numerical_drift = compute_numerical_drift
test_categorical_drift = compute_categorical_drift
