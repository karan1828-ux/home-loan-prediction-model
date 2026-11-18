"""Unit tests for the model evaluation module."""

import unittest
import numpy as np
import pandas as pd

from src.evaluate import evaluate_model, get_classification_report


class TestEvaluate(unittest.TestCase):
    """Test suite for classification metrics evaluation."""

    def test_evaluate_model_perfect_predictions(self):
        """Test evaluation metrics when predictions are 100% accurate."""
        y_true = pd.Series(["Y", "N", "Y", "N", "Y", "N"])
        y_pred = np.array(["Y", "N", "Y", "N", "Y", "N"])

        metrics = evaluate_model(y_true, y_pred)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

        # In ['N', 'Y'] order: TN=3, FP=0, FN=0, TP=3
        expected_cm = np.array([[3, 0], [0, 3]])
        np.testing.assert_array_equal(metrics["confusion_matrix"], expected_cm)

    def test_evaluate_model_imperfect_predictions(self):
        """Test evaluation metrics with known error rates."""
        # TN=2, FP=1, FN=1, TP=2
        # y_true: N, N, N, Y, Y, Y
        # y_pred: N, N, Y, N, Y, Y
        y_true = ["N", "N", "N", "Y", "Y", "Y"]
        y_pred = ["N", "N", "Y", "N", "Y", "Y"]

        metrics = evaluate_model(y_true, y_pred)

        # Total=6, Correct=4 -> Accuracy = 4/6 = 0.6667
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6, places=4)
        # Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3 = 0.6667
        self.assertAlmostEqual(metrics["precision"], 2 / 3, places=4)
        # Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3 = 0.6667
        self.assertAlmostEqual(metrics["recall"], 2 / 3, places=4)
        # F1 = 2 * (P * R) / (P + R) = 2/3 = 0.6667
        self.assertAlmostEqual(metrics["f1_score"], 2 / 3, places=4)

        expected_cm = np.array([[2, 1], [1, 2]])
        np.testing.assert_array_equal(metrics["confusion_matrix"], expected_cm)

    def test_confusion_matrix_correctness(self):
        """Test confusion matrix structure and label alignment."""
        y_true = ["N", "N", "Y", "Y"]
        y_pred = ["N", "Y", "N", "Y"]

        metrics = evaluate_model(y_true, y_pred)
        cm = metrics["confusion_matrix"]

        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.shape, (2, 2))
        # Row 0: Actual N -> TN=1, FP=1
        # Row 1: Actual Y -> FN=1, TP=1
        self.assertEqual(cm[0, 0], 1)
        self.assertEqual(cm[0, 1], 1)
        self.assertEqual(cm[1, 0], 1)
        self.assertEqual(cm[1, 1], 1)

    def test_zero_division_handling(self):
        """Test handling of cases where precision or recall has zero divisor."""
        # No positive instances predicted: TP=0, FP=0 -> Precision has 0 division
        y_true = ["N", "N", "Y"]
        y_pred = ["N", "N", "N"]

        metrics = evaluate_model(y_true, y_pred)

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1_score"], 0.0)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3, places=4)

    def test_numeric_labels_evaluation(self):
        """Test that evaluate_model handles integer 0/1 labels properly."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]

        metrics = evaluate_model(y_true, y_pred)

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        # TP=2, FP=1 -> Precision = 2/3
        self.assertAlmostEqual(metrics["precision"], 2 / 3, places=4)
        # Recall = 2/2 = 1.0
        self.assertEqual(metrics["recall"], 1.0)
        self.assertIsInstance(metrics["confusion_matrix"], np.ndarray)

    def test_length_mismatch_raises_error(self):
        """Test that mismatched input lengths raise ValueError."""
        y_true = ["Y", "N", "Y"]
        y_pred = ["Y", "N"]

        with self.assertRaises(ValueError) as ctx:
            evaluate_model(y_true, y_pred)
        self.assertIn("Length mismatch", str(ctx.exception))

    def test_empty_inputs_raise_error(self):
        """Test that empty inputs raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            evaluate_model([], [])
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_classification_report_generation(self):
        """Test that get_classification_report returns a valid formatted string."""
        y_true = ["N", "Y", "Y", "N"]
        y_pred = ["N", "Y", "N", "N"]

        report = get_classification_report(y_true, y_pred)
        self.assertIsInstance(report, str)
        self.assertIn("precision", report)
        self.assertIn("recall", report)
        self.assertIn("f1-score", report)


if __name__ == "__main__":
    unittest.main()

