"""Explainable AI (XAI) package for KP Smart Crop Doctor.

Provides Grad-CAM visual interpretability, heatmap generation, overlay blending,
and model-randomization sanity checks.
"""

from src.explainability.gradcam import (
    GradCAM,
    generate_heatmap,
    generate_overlay,
    predict_and_explain,
    model_randomization_sanity_check,
    compute_heatmap_similarity,
    load_label_encoder,
    preprocess_image_for_model,
)

__all__ = [
    "GradCAM",
    "generate_heatmap",
    "generate_overlay",
    "predict_and_explain",
    "model_randomization_sanity_check",
    "compute_heatmap_similarity",
    "load_label_encoder",
    "preprocess_image_for_model",
]
