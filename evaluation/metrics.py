"""Comprehensive evaluation metrics suite for classification and confidence calibration.

Computes Top-1 accuracy, Top-2 accuracy, Macro/Per-class Precision/Recall/F1,
Confusion Matrix, and Expected Calibration Error (ECE).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute complete classification metrics including Top-1, Top-2, Macro and Per-Class stats.

    Args:
        y_true: 1D array of ground truth integer class labels (shape: [N]).
        y_prob: 2D array of predicted softmax probabilities (shape: [N, num_classes]).
        class_names: Ordered list of class label strings.

    Returns:
        Dictionary containing overall and per-class metrics.
    """
    y_pred = np.argmax(y_prob, axis=1)
    num_classes = len(class_names)

    # Top-1 Accuracy
    top1_acc = float(accuracy_score(y_true, y_pred))

    # Top-2 Accuracy
    all_classes = np.arange(num_classes)
    if num_classes >= 2:
        top2_acc = float(top_k_accuracy_score(y_true, y_prob, k=2, labels=all_classes))
    else:
        top2_acc = top1_acc

    # Macro Precision, Recall, F1
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Weighted Precision, Recall, F1
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-Class Metrics
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_classes, average=None, zero_division=0
    )

    per_class_metrics = {}
    for idx, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(per_class_p[idx]),
            "recall": float(per_class_r[idx]),
            "f1_score": float(per_class_f1[idx]),
            "support": int(support[idx]),
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    cm_normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-10)

    # Expected Calibration Error (ECE)
    ece, calibration_curve = compute_expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "top1_accuracy": top1_acc,
        "top2_accuracy": top2_acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "expected_calibration_error": float(ece),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "calibration_curve": calibration_curve,
    }


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute Expected Calibration Error (ECE) and reliability bin statistics.

    Args:
        y_true: Ground truth labels (shape: [N]).
        y_prob: Softmax predicted probabilities (shape: [N, num_classes]).
        n_bins: Number of confidence bins (default: 10).

    Returns:
        Tuple of (ece_value, bin_statistics_list).
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(y_true)
    bin_stats = []

    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if bin_idx == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_weight = bin_count / total_samples
            ece += bin_weight * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = float((bin_lower + bin_upper) / 2.0)

        bin_stats.append({
            "bin": bin_idx,
            "range": [float(bin_lower), float(bin_upper)],
            "count": bin_count,
            "accuracy": bin_acc,
            "confidence": bin_conf,
        })

    return round(float(ece), 4), bin_stats
