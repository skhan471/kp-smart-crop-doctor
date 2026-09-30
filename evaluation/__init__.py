"""Evaluation suites: classification metrics, top-2 accuracy, and RAG grounding fidelity."""

from evaluation.metrics import compute_classification_metrics, compute_expected_calibration_error

__all__ = [
    "compute_classification_metrics",
    "compute_expected_calibration_error",
]
