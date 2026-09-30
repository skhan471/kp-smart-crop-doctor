"""Comparative evaluation, latency benchmarking, and visualization generator.

Compares MobileNetV3-Large and EfficientNet-B0 on test set, generates confusion matrices,
training curves, reliability diagrams, and produces the final comparative report.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.utils import get_device, set_seed, get_logger
from src.vision.dataset import CropDataset, get_transforms
from src.vision.model import create_model, count_parameters, get_model_size_mb
from evaluation.metrics import compute_classification_metrics

logger = get_logger(__name__)


def benchmark_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device_str: str = "cpu",
    warmup: int = 50,
    iterations: int = 200,
) -> Dict[str, float]:
    """Measure single-image inference latency and throughput.

    Args:
        model: Evaluated PyTorch model.
        input_size: Tensor dimensions (batch_size=1, C=3, H=224, W=224).
        device_str: Computation device ('cuda' or 'cpu').
        warmup: Number of untimed warmup passes.
        iterations: Number of timed inference iterations.

    Returns:
        Dictionary with mean latency (ms), std (ms), p95 (ms), and throughput (FPS).
    """
    device = torch.device(device_str)
    model.eval()
    dummy_input = torch.randn(input_size, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Synchronize if GPU
    if device_str == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device_str == "cuda":
                torch.cuda.synchronize()
            duration = (time.perf_counter() - start) * 1000.0  # ms
            timings.append(duration)

    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    p95_ms = float(np.percentile(timings, 95))
    throughput_fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "device": device_str,
        "mean_latency_ms": round(mean_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "throughput_fps": round(throughput_fps, 1),
    }


def plot_training_history(
    mnet_history: Dict[str, List[float]],
    effnet_history: Dict[str, List[float]],
    output_path: Path,
):
    """Plot comparative Loss, Accuracy, and Validation Macro-F1 curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs_mnet = range(1, len(mnet_history["train_loss"]) + 1)
    epochs_eff = range(1, len(effnet_history["train_loss"]) + 1)

    # 1. Loss
    axes[0].plot(epochs_mnet, mnet_history["train_loss"], "b--", label="MobileNetV3 (Train)")
    axes[0].plot(epochs_mnet, mnet_history["val_loss"], "b-", label="MobileNetV3 (Val)")
    axes[0].plot(epochs_eff, effnet_history["train_loss"], "r--", label="EfficientNet-B0 (Train)")
    axes[0].plot(epochs_eff, effnet_history["val_loss"], "r-", label="EfficientNet-B0 (Val)")
    axes[0].set_title("Cross-Entropy Loss Across Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # 2. Accuracy
    axes[1].plot(epochs_mnet, mnet_history["train_accuracy"], "b--", label="MobileNetV3 (Train)")
    axes[1].plot(epochs_mnet, mnet_history["val_accuracy"], "b-", label="MobileNetV3 (Val)")
    axes[1].plot(epochs_eff, effnet_history["train_accuracy"], "r--", label="EfficientNet-B0 (Train)")
    axes[1].plot(epochs_eff, effnet_history["val_accuracy"], "r-", label="EfficientNet-B0 (Val)")
    axes[1].set_title("Accuracy Across Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # 3. Validation Macro-F1 (Selection Metric)
    axes[2].plot(epochs_mnet, mnet_history["val_macro_f1"], "b-o", label="MobileNetV3")
    axes[2].plot(epochs_eff, effnet_history["val_macro_f1"], "r-s", label="EfficientNet-B0")
    axes[2].set_title("Validation Macro-F1 (Model Selection Criterion)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1 Score")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved training history curves to {output_path}")


def plot_confusion_matrix(cm_normalized: List[List[float]], class_names: List[str], title: str, output_path: Path):
    """Plot and save normalized confusion matrix heatmap."""
    plt.figure(figsize=(11, 9))
    cm_arr = np.array(cm_normalized)

    # Shorten class names for readability on axis
    short_names = [c.replace("Corn_(maize)___", "Corn: ").replace("Potato___", "Potato: ").replace("Apple___", "Apple: ") for c in class_names]

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        cbar=True,
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")


def plot_calibration_diagram(mnet_curve: List[Dict], eff_curve: List[Dict], mnet_ece: float, eff_ece: float, output_path: Path):
    """Plot reliability diagram comparing confidence calibration between models."""
    plt.figure(figsize=(8, 6))

    # Perfect calibration reference diagonal
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # MobileNetV3 points
    m_confs = [b["confidence"] for b in mnet_curve if b["count"] > 0]
    m_accs = [b["accuracy"] for b in mnet_curve if b["count"] > 0]
    plt.plot(m_confs, m_accs, "b-o", label=f"MobileNetV3-Large (ECE = {mnet_ece:.4f})")

    # EfficientNet-B0 points
    e_confs = [b["confidence"] for b in eff_curve if b["count"] > 0]
    e_accs = [b["accuracy"] for b in eff_curve if b["count"] > 0]
    plt.plot(e_confs, e_accs, "r-s", label=f"EfficientNet-B0 (ECE = {eff_ece:.4f})")

    plt.title("Reliability Diagram (Confidence Calibration on Test Set)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved calibration diagram to {output_path}")
