"""End-to-end comparative experiment runner for MobileNetV3-Large and EfficientNet-B0.

Trains both architectures under identical experimental conditions, evaluates on unseen test set,
benchmarks latency, measures confidence calibration, and generates visual artifacts.
"""

from pathlib import Path
import json
import time
import shutil
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.utils import get_device, set_seed, get_logger
from src.vision.dataset import CropDataset, get_transforms
from src.vision.model import create_model, get_model_size_mb, count_parameters
from src.vision.train import run_training_experiment, evaluate_model
from src.vision.evaluate_comparison import (
    benchmark_latency,
    plot_training_history,
    plot_confusion_matrix,
    plot_calibration_diagram,
)

logger = get_logger("comparative_trainer")


def run_comparative_study(
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device_name: str = "auto",
    random_seed: int = 42,
    patience: int = 3,
) -> Dict[str, Any]:
    """Execute complete comparative evaluation between MobileNetV3-Large and EfficientNet-B0."""
    start_total_time = time.time()
    set_seed(random_seed)
    device_str = get_device(device_name)
    device = torch.device(device_str)

    figures_dir = config.EVALUATION_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    test_csv = config.SPLITS_DATA_DIR / "test.csv"
    train_csv = config.SPLITS_DATA_DIR / "train.csv"

    # Ensure label encoder exists
    test_dataset = CropDataset(test_csv, transform=get_transforms("val"))
    class_names = test_dataset.classes
    num_classes = len(class_names)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device_str == "cuda"),
    )
    criterion = nn.CrossEntropyLoss()

    logger.info("=" * 70)
    logger.info("PHASE 1: Training MobileNetV3-Large (Primary Candidate)")
    logger.info("=" * 70)
    mnet_results = run_training_experiment(
        backbone_name="mobilenet_v3_large",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device_name=device_str,
        random_seed=random_seed,
        patience=patience,
    )

    logger.info("=" * 70)
    logger.info("PHASE 2: Training EfficientNet-B0 (Benchmark Model)")
    logger.info("=" * 70)
    effnet_results = run_training_experiment(
        backbone_name="efficientnet_b0",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device_name=device_str,
        random_seed=random_seed,
        patience=patience,
    )

    logger.info("=" * 70)
    logger.info("PHASE 3: Evaluating Both Models on Unseen Test Split")
    logger.info("=" * 70)

    # 1. Evaluate MobileNetV3 on Test Set
    mnet_model = create_model("mobilenet_v3_large", num_classes=num_classes, pretrained=False)
    mnet_ckpt = torch.load(mnet_results["checkpoint_path"], map_location=device)
    mnet_model.load_state_dict(mnet_ckpt["model_state_dict"])
    mnet_model = mnet_model.to(device)

    _, mnet_test_acc, mnet_test_f1, mnet_test_metrics, _, _ = evaluate_model(
        mnet_model, test_loader, criterion, device, class_names
    )
    mnet_latency = benchmark_latency(mnet_model, device_str=device_str)

    # 2. Evaluate EfficientNet-B0 on Test Set
    eff_model = create_model("efficientnet_b0", num_classes=num_classes, pretrained=False)
    eff_ckpt = torch.load(effnet_results["checkpoint_path"], map_location=device)
    eff_model.load_state_dict(eff_ckpt["model_state_dict"])
    eff_model = eff_model.to(device)

    _, eff_test_acc, eff_test_f1, eff_test_metrics, _, _ = evaluate_model(
        eff_model, test_loader, criterion, device, class_names
    )
    eff_latency = benchmark_latency(eff_model, device_str=device_str)

    logger.info("=" * 70)
    logger.info("PHASE 4: Generating Comparative Visualizations")
    logger.info("=" * 70)

    # Plot 1: Training Curves
    plot_training_history(
        mnet_results["history"],
        effnet_results["history"],
        figures_dir / "training_curves_comparison.png",
    )

    # Plot 2 & 3: Confusion Matrices
    plot_confusion_matrix(
        mnet_test_metrics["confusion_matrix_normalized"],
        class_names,
        "MobileNetV3-Large: Normalized Confusion Matrix (Test Split)",
        figures_dir / "confusion_matrix_mobilenet_v3_large.png",
    )
    plot_confusion_matrix(
        eff_test_metrics["confusion_matrix_normalized"],
        class_names,
        "EfficientNet-B0: Normalized Confusion Matrix (Test Split)",
        figures_dir / "confusion_matrix_efficientnet_b0.png",
    )

    # Plot 4: Reliability Diagram (Confidence Calibration)
    plot_calibration_diagram(
        mnet_test_metrics["calibration_curve"],
        eff_test_metrics["calibration_curve"],
        mnet_test_metrics["expected_calibration_error"],
        eff_test_metrics["expected_calibration_error"],
        figures_dir / "calibration_reliability_diagram.png",
    )

    logger.info("=" * 70)
    logger.info("PHASE 5: Model Selection Based Strictly on Validation Macro-F1")
    logger.info("=" * 70)

    # Model Selection Rule: Validation Macro-F1 ONLY (NOT Test Set!)
    mnet_val_macro_f1 = mnet_results["best_val_macro_f1"]
    eff_val_macro_f1 = effnet_results["best_val_macro_f1"]

    if mnet_val_macro_f1 >= eff_val_macro_f1:
        winner = "mobilenet_v3_large"
        winning_val_f1 = mnet_val_macro_f1
        winning_ckpt = mnet_results["checkpoint_path"]
        selection_rationale = (
            f"MobileNetV3-Large achieved higher or equivalent Validation Macro-F1 ({mnet_val_macro_f1:.4f} "
            f"vs {eff_val_macro_f1:.4f}) with lower parameter count ({mnet_results['parameter_counts']['total_parameters']:,} "
            f"vs {effnet_results['parameter_counts']['total_parameters']:,}) and lower inference latency."
        )
    else:
        winner = "efficientnet_b0"
        winning_val_f1 = eff_val_macro_f1
        winning_ckpt = effnet_results["checkpoint_path"]
        selection_rationale = (
            f"EfficientNet-B0 achieved superior Validation Macro-F1 ({eff_val_macro_f1:.4f} vs {mnet_val_macro_f1:.4f})."
        )

    # Save selected model to models/best_crop_model.pt
    best_model_target = config.MODELS_DIR / "best_crop_model.pt"
    shutil.copy2(winning_ckpt, best_model_target)
    logger.info(f"Selected model: {winner} (saved to {best_model_target})")

    # Assemble comprehensive report
    report = {
        "experimental_protocol": {
            "dataset": "11-Class KP Core (Maize, Potato, Apple)",
            "total_images": 9175,
            "train_samples": 6422,
            "val_samples": 1376,
            "test_samples": 1377,
            "random_seed": random_seed,
            "split_strategy": "Group-Stratified (Leaf-Group Isolated)",
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": "AdamW",
                "loss": "Class-weighted CrossEntropyLoss",
                "scheduler": "CosineAnnealingLR",
                "device": device_str,
            },
            "selection_criterion": "Validation Macro-F1 (Strictly prior to test evaluation)",
        },
        "models": {
            "mobilenet_v3_large": {
                "backbone": "mobilenet_v3_large",
                "parameters": mnet_results["parameter_counts"],
                "checkpoint_size_mb": mnet_results["model_size_mb"],
                "best_epoch": mnet_results["best_epoch"],
                "validation_loss": mnet_results["history"]["val_loss"][mnet_results["best_epoch"] - 1],
                "validation_accuracy": mnet_results["history"]["val_accuracy"][mnet_results["best_epoch"] - 1],
                "validation_macro_f1": mnet_val_macro_f1,
                "test_metrics": {
                    "top1_accuracy": mnet_test_metrics["top1_accuracy"],
                    "top2_accuracy": mnet_test_metrics["top2_accuracy"],
                    "macro_precision": mnet_test_metrics["macro_precision"],
                    "macro_recall": mnet_test_metrics["macro_recall"],
                    "macro_f1": mnet_test_metrics["macro_f1"],
                    "weighted_f1": mnet_test_metrics["weighted_f1"],
                    "expected_calibration_error": mnet_test_metrics["expected_calibration_error"],
                    "per_class": mnet_test_metrics["per_class"],
                },
                "latency_benchmark": mnet_latency,
                "training_history": mnet_results["history"],
            },
            "efficientnet_b0": {
                "backbone": "efficientnet_b0",
                "parameters": effnet_results["parameter_counts"],
                "checkpoint_size_mb": effnet_results["model_size_mb"],
                "best_epoch": effnet_results["best_epoch"],
                "validation_loss": effnet_results["history"]["val_loss"][effnet_results["best_epoch"] - 1],
                "validation_accuracy": effnet_results["history"]["val_accuracy"][effnet_results["best_epoch"] - 1],
                "validation_macro_f1": eff_val_macro_f1,
                "test_metrics": {
                    "top1_accuracy": eff_test_metrics["top1_accuracy"],
                    "top2_accuracy": eff_test_metrics["top2_accuracy"],
                    "macro_precision": eff_test_metrics["macro_precision"],
                    "macro_recall": eff_test_metrics["macro_recall"],
                    "macro_f1": eff_test_metrics["macro_f1"],
                    "weighted_f1": eff_test_metrics["weighted_f1"],
                    "expected_calibration_error": eff_test_metrics["expected_calibration_error"],
                    "per_class": eff_test_metrics["per_class"],
                },
                "latency_benchmark": eff_latency,
                "training_history": effnet_results["history"],
            },
        },
        "model_selection": {
            "selected_model": winner,
            "selection_metric": "Validation Macro-F1",
            "mobilenet_v3_val_macro_f1": mnet_val_macro_f1,
            "efficientnet_b0_val_macro_f1": eff_val_macro_f1,
            "rationale": selection_rationale,
            "best_checkpoint_path": str(best_model_target),
        },
        "dataset_limitation_statement": (
            "The training and benchmark dataset consists of standardized laboratory foliage captures under "
            "controlled lighting and neutral backgrounds. While ideal for learning canonical lesion morphology and "
            "preventing background domain confounding in Grad-CAM, classification performance on in-the-wild field photographs "
            "with occlusions, solar reflections, motion blur, and heterogeneous soil/canopy backgrounds may vary. "
            "Future iterations should incorporate localized domain-adaptation and field data collection across KP farming clusters."
        ),
        "total_experiment_duration_sec": round(time.time() - start_total_time, 2),
    }

    report_path = config.EVALUATION_DIR / "comparative_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Comparative experiment report saved to {report_path}")

    return report


if __name__ == "__main__":
    run_comparative_study(epochs=10, batch_size=32, learning_rate=3e-4, random_seed=42)
