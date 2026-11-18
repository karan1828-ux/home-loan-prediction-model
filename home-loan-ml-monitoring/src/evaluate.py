"""Model evaluation module for the Home Loan Prediction project.

This module provides reusable routines to compute classification metrics
(accuracy, precision, recall, F1-score, and confusion matrix) with support
for non-numeric string labels ('Y'/'N') and explicit zero-division handling.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _determine_pos_label(
    y_true: Union[pd.Series, np.ndarray, List[Any]],
    y_pred: Union[pd.Series, np.ndarray, List[Any]],
    pos_label: Optional[Union[str, int]] = None,
) -> Union[str, int]:
    """Infer or validate the positive class label for binary classification.

    Defaults to 'Y' if present in the target classes, or 1 for numeric targets.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray | list
        Ground truth target values.
    y_pred : pd.Series | np.ndarray | list
        Predicted target values.
    pos_label : str | int | None, default=None
        User-specified positive label, if provided.

    Returns
    -------
    str | int
        Determined positive label.
    """
    if pos_label is not None:
        return pos_label

    unique_vals = set(pd.Series(y_true).dropna().unique()) | set(
        pd.Series(y_pred).dropna().unique()
    )

    if "Y" in unique_vals:
        return "Y"
    if 1 in unique_vals:
        return 1
    if "1" in unique_vals:
        return "1"

    # Fallback to sorted unique elements
    sorted_vals = sorted(list(unique_vals), key=lambda x: str(x))
    return sorted_vals[-1] if sorted_vals else "Y"


def evaluate_model(
    y_true: Union[pd.Series, np.ndarray, List[Any]],
    y_pred: Union[pd.Series, np.ndarray, List[Any]],
    pos_label: Optional[Union[str, int]] = None,
    labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics for model predictions.

    Computes:
    - accuracy
    - precision (binary averaging, zero_division=0)
    - recall (binary averaging, zero_division=0)
    - f1_score (binary averaging, zero_division=0)
    - confusion_matrix (2x2 numpy array)

    Parameters
    ----------
    y_true : pd.Series | np.ndarray | list
        Actual ground truth target labels.
    y_pred : pd.Series | np.ndarray | list
        Predicted target labels from the model.
    pos_label : str | int | None, default=None
        Positive class label. If None, infers 'Y' for string targets or 1 for numeric.
    labels : list | None, default=None
        Ordered list of class labels for confusion matrix construction.
        If None and target contains 'Y'/'N', defaults to ['N', 'Y'].

    Returns
    -------
    dict[str, Any]
        Dictionary of machine-readable metrics:
        {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1_score': float,
            'confusion_matrix': np.ndarray,
        }

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or are empty.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} samples, "
            f"but y_pred has {len(y_pred_arr)} samples."
        )

    # Determine positive class label
    effective_pos_label = _determine_pos_label(y_true_arr, y_pred_arr, pos_label)

    # Determine confusion matrix label order
    if labels is None:
        unique_vals = set(y_true_arr) | set(y_pred_arr)
        if unique_vals.issubset({"N", "Y"}):
            labels = ["N", "Y"]
        elif unique_vals.issubset({0, 1}):
            labels = [0, 1]
        else:
            labels = sorted(list(unique_vals), key=lambda x: str(x))

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(
        precision_score(
            y_true_arr,
            y_pred_arr,
            pos_label=effective_pos_label,
            average="binary",
            zero_division=0,
        )
    )
    rec = float(
        recall_score(
            y_true_arr,
            y_pred_arr,
            pos_label=effective_pos_label,
            average="binary",
            zero_division=0,
        )
    )
    f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            pos_label=effective_pos_label,
            average="binary",
            zero_division=0,
        )
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
    }


def get_classification_report(
    y_true: Union[pd.Series, np.ndarray, List[Any]],
    y_pred: Union[pd.Series, np.ndarray, List[Any]],
) -> str:
    """Generate a formatted textual classification report.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray | list
        Actual ground truth target labels.
    y_pred : pd.Series | np.ndarray | list
        Predicted target labels.

    Returns
    -------
    str
        Textual classification report showing per-class precision, recall, and F1.
    """
    return classification_report(y_true, y_pred, zero_division=0)
