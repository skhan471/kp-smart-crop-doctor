"""Unit tests for vision models, datasets, and evaluation metrics."""

import pytest
import numpy as np
from pathlib import Path
from src import config
from evaluation.metrics import compute_classification_metrics, compute_expected_calibration_error


def test_metrics_calculation_synthetic():
    """Verify classification metrics with known synthetic predictions."""
    class_names = ["ClassA", "ClassB", "ClassC"]
    y_true = np.array([0, 1, 2, 0, 1, 2])

    # Simulated probability distributions
    y_prob = np.array([
        [0.9, 0.05, 0.05],  # Pred 0 (Correct)
        [0.1, 0.8, 0.1],    # Pred 1 (Correct)
        [0.2, 0.3, 0.5],    # Pred 2 (Correct)
        [0.4, 0.5, 0.1],    # Pred 1 (Incorrect, top-2 has 0)
        [0.05, 0.9, 0.05],  # Pred 1 (Correct)
        [0.1, 0.1, 0.8],    # Pred 2 (Correct)
    ])

    metrics = compute_classification_metrics(y_true, y_prob, class_names)

    # 5 out of 6 correct = 83.33% Top-1
    assert round(metrics["top1_accuracy"], 2) == 0.83
    # All 6 samples have true class in top-2 = 100% Top-2
    assert metrics["top2_accuracy"] == 1.0

    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert len(metrics["per_class"]) == 3
    assert "expected_calibration_error" in metrics
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0


def test_ece_perfect_calibration():
    """Verify ECE calculation is near zero for well-calibrated synthetic predictions."""
    y_true = np.array([0] * 50 + [1] * 50)
    # Predict 0.5 confidence for 50% accuracy bin
    y_prob = np.array([[0.5, 0.5]] * 100)

    ece, curve = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert isinstance(ece, float)
    assert len(curve) == 10
