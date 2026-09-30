"""Comprehensive evaluation script for Grad-CAM explainability pipeline.

Evaluates:
1. Visual explainability across all 11 crop disease classes (including difficult classes:
   Maize Cercospora/Gray Leaf Spot and Potato healthy).
2. Verification that Grad-CAM corresponds to predicted class.
3. Model-randomization sanity check (Adebayo et al., NeurIPS 2018).
4. Generation and export of publication-grade figures in evaluation/figures/.
5. Export of structured quantitative evaluation report to evaluation/gradcam_evaluation_report.json.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torch

from src import config
from src.utils import get_logger
from src.vision.model import create_model
from src.explainability.gradcam import (
    GradCAM,
    predict_and_explain,
    model_randomization_sanity_check,
    load_label_encoder,
)

logger = get_logger(__name__)


def load_trained_model(
    checkpoint_path: Path = config.MODELS_DIR / "best_crop_model.pt",
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Load the trained MobileNetV3-Large model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading trained model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", config.NUM_CLASSES)
    backbone = checkpoint.get("backbone_name", config.DEFAULT_BACKBONE)

    model = create_model(backbone_name=backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded {backbone} (trained epoch {checkpoint.get('epoch')}) successfully.")
    return model


def select_representative_samples(
    df_test: pd.DataFrame,
    class_names: List[str],
    model: torch.nn.Module,
) -> Dict[str, Dict[str, Any]]:
    """Select representative test samples for each of the 11 classes.

    Ensures each selected sample has clean ground-truth label and evaluates top-1 prediction.
    """
    representative_samples = {}

    for c in class_names:
        sub = df_test[df_test["class_name"] == c]
        if sub.empty:
            continue

        selected_row = None
        # Prioritize samples where top-1 matches ground truth
        for _, row in sub.iterrows():
            res = predict_and_explain(model, row["file_path"], class_names=class_names)
            if res["predicted_class"] == c:
                selected_row = row
                break

        # Fallback to first sample if no perfect match in first sweep
        if selected_row is None:
            selected_row = sub.iloc[0]

        representative_samples[c] = {
            "file_path": selected_row["file_path"],
            "filename": selected_row["filename"],
            "ground_truth": c,
            "crop": selected_row.get("crop", c.split("___")[0]),
            "condition": selected_row.get("condition", c.split("___")[-1]),
        }

    return representative_samples


def generate_11_class_grid_figure(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate high-resolution 11-class visualization grid (Original, Heatmap, Overlay)."""
    n_classes = len(results)
    # 11 classes -> 11 rows, 3 columns: Original, Heatmap, Overlay
    fig, axes = plt.subplots(n_classes, 3, figsize=(14, 3.2 * n_classes), dpi=200)

    for i, res in enumerate(results):
        gt = res["ground_truth"]
        pred = res["predicted_class"]
        conf = res["confidence"] * 100.0
        top2 = res["top2_predictions"]
        is_correct = (gt == pred)

        # Simplify class name for display
        clean_gt = gt.replace("___", ": ").replace("_", " ")
        clean_pred = pred.replace("___", ": ").replace("_", " ")

        # Column 0: Original
        axes[i, 0].imshow(res["original_image"])
        axes[i, 0].set_title(f"Target: {clean_gt}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # Column 1: Heatmap
        axes[i, 1].imshow(res["gradcam_heatmap"])
        axes[i, 1].set_title(f"Grad-CAM Heatmap (features.16)", fontsize=10)
        axes[i, 1].axis("off")

        # Column 2: Overlay
        axes[i, 2].imshow(res["gradcam_overlay"])
        pred_status = "CORRECT" if is_correct else "MISCLASSIFIED"
        title_color = "#1b5e20" if is_correct else "#b71c1c"
        overlay_title = (
            f"Pred: {clean_pred}\n"
            f"Conf: {conf:.1f}% ({pred_status}) | Top-2: {top2[1]['confidence']*100:.1f}%"
        )
        axes[i, 2].set_title(overlay_title, fontsize=10, color=title_color, fontweight="semibold")
        axes[i, 2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 11-class Grad-CAM grid to: {output_path}")


def generate_sanity_check_figure(
    sanity_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate publication figure for Adebayo et al. model-randomization sanity checks.

    Shows for representative diseases:
    1. Original Image
    2. Trained Model Grad-CAM
    3. Randomized Model Grad-CAM
    4. Quantitative metrics (Pearson r, Spearman rho, MAE)
    """
    n_cases = len(sanity_results)
    fig, axes = plt.subplots(n_cases, 4, figsize=(16, 3.8 * n_cases), dpi=200)

    for i, item in enumerate(sanity_results):
        orig_img = item["original_image"]
        trained_cam = item["trained_cam"]
        rand_cam = item["randomized_cam"]
        metrics = item["similarity"]
        class_name = item["class_name"].replace("___", ": ").replace("_", " ")

        # 1. Original
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original Leaf\n{class_name}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # 2. Trained CAM
        axes[i, 1].imshow(trained_cam, cmap="jet")
        axes[i, 1].set_title("Trained Model Grad-CAM\n(Localized Lesion Features)", fontsize=10, color="#1b5e20")
        axes[i, 1].axis("off")

        # 3. Randomized CAM
        axes[i, 2].imshow(rand_cam, cmap="jet")
        axes[i, 2].set_title("Randomized Model Grad-CAM\n(Untrained Random Weights)", fontsize=10, color="#b71c1c")
        axes[i, 2].axis("off")

        # 4. Difference Map & Metric Summary
        diff_map = np.abs(trained_cam - rand_cam)
        im = axes[i, 3].imshow(diff_map, cmap="viridis")
        metric_text = (
            f"Absolute Difference (|T - R|)\n"
            f"Pearson r: {metrics['pearson_correlation']:.4f}\n"
            f"Spearman \u03c1: {metrics['spearman_rank_correlation']:.4f}\n"
            f"MAE: {metrics['mean_absolute_error']:.4f}\n"
            f"Sanity Check: {'PASSED' if item['passed'] else 'FAILED'}"
        )
        axes[i, 3].set_title(metric_text, fontsize=9.5, fontweight="semibold")
        axes[i, 3].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sanity check figure to: {output_path}")


def run_gradcam_evaluation() -> Dict[str, Any]:
    """Execute complete Milestone 4 Grad-CAM evaluation suite."""
    logger.info("Starting Grad-CAM evaluation pipeline...")

    # Load model and class names
    model = load_trained_model()
    class_names = load_label_encoder()

    # Load test split
    test_csv_path = config.SPLITS_DATA_DIR / "test.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test split not found at: {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    logger.info(f"Loaded test split with {len(df_test)} samples across {df_test['class_name'].nunique()} classes.")

    # 1. Select representative samples
    rep_samples = select_representative_samples(df_test, class_names, model)

    # 2. Evaluate Grad-CAM on all 11 classes
    results = []
    logger.info("Generating Grad-CAM explanations for all 11 classes...")
    for c in class_names:
        sample_info = rep_samples[c]
        res = predict_and_explain(
            model=model,
            image=sample_info["file_path"],
            class_names=class_names,
        )
        res["ground_truth"] = c
        res["filename"] = sample_info["filename"]
        res["sample_path"] = sample_info["file_path"]
        results.append(res)
        logger.info(
            f"Class [{c}] -> Predicted: [{res['predicted_class']}] "
            f"(Conf: {res['confidence']*100:.2f}%, Top-2: {res['top2_predictions'][1]['class_name']} "
            f"at {res['top2_predictions'][1]['confidence']*100:.2f}%)"
        )

    # 3. Model-Randomization Sanity Checks on representative cases
    # Specifically include difficult classes and varied pathologies:
    sanity_test_classes = [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Potato___Early_blight",
        "Apple___Apple_scab",
        "Potato___healthy",
    ]
    logger.info("Executing Adebayo et al. model-randomization sanity checks...")
    sanity_results = []
    for c in sanity_test_classes:
        sample_info = rep_samples[c]
        sanity = model_randomization_sanity_check(
            trained_model=model,
            image=sample_info["file_path"],
            random_seed=42,
        )
        _, pil_img = res["original_image"], Image.open(sample_info["file_path"]).convert("RGB").resize((224, 224))
        sanity["class_name"] = c
        sanity["filename"] = sample_info["filename"]
        sanity["original_image"] = pil_img
        sanity_results.append(sanity)

    # 4. Generate Figures
    figures_dir = config.EVALUATION_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    grid_fig_path = figures_dir / "gradcam_11_classes_grid.png"
    sanity_fig_path = figures_dir / "gradcam_sanity_check_randomization.png"

    generate_11_class_grid_figure(results, grid_fig_path)
    generate_sanity_check_figure(sanity_results, sanity_fig_path)

    # 5. Build structured JSON report
    report = {
        "milestone": "Milestone 4 - Explainable AI Pipeline using Grad-CAM",
        "model_architecture": "MobileNetV3-Large",
        "target_layer": config.GRADCAM_TARGET_LAYER,
        "classes_evaluated": len(results),
        "per_class_explanations": [
            {
                "ground_truth": r["ground_truth"],
                "predicted_class": r["predicted_class"],
                "is_correct": bool(r["ground_truth"] == r["predicted_class"]),
                "confidence": round(float(r["confidence"]), 4),
                "top2_predictions": [
                    {
                        "class_name": p["class_name"],
                        "confidence": round(float(p["confidence"]), 4),
                    }
                    for p in r["top2_predictions"]
                ],
                "filename": r["filename"],
                "cam_min": round(float(r["raw_cam"].min()), 4),
                "cam_max": round(float(r["raw_cam"].max()), 4),
                "cam_mean": round(float(r["raw_cam"].mean()), 4),
            }
            for r in results
        ],
        "sanity_check_results": [
            {
                "class_name": s["class_name"],
                "filename": s["filename"],
                "pearson_correlation": s["similarity"]["pearson_correlation"],
                "spearman_rank_correlation": s["similarity"]["spearman_rank_correlation"],
                "mean_absolute_error": s["similarity"]["mean_absolute_error"],
                "passed": bool(s["passed"]),
            }
            for s in sanity_results
        ],
        "all_sanity_checks_passed": all(s["passed"] for s in sanity_results),
        "figures": {
            "gradcam_11_classes_grid": str(grid_fig_path),
            "gradcam_sanity_check_randomization": str(sanity_fig_path),
        },
    }

    report_path = config.EVALUATION_DIR / "gradcam_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Grad-CAM evaluation report saved to: {report_path}")
    return report


if __name__ == "__main__":
    run_gradcam_evaluation()
