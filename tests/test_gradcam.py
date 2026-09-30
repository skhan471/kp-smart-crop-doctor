"""Unit and integration tests for Grad-CAM explainability pipeline.

Tests:
1. Hook registration and cleanup (memory safety).
2. CAM tensor dimensions and min-max value normalization [0.0, 1.0].
3. Colormap heatmap and overlay generation.
4. Reusable inference API contract (predict_and_explain).
5. Model-randomization sanity checks (Adebayo et al., NeurIPS 2018).
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

from src import config
from src.vision.model import create_model
from src.explainability.gradcam import (
    GradCAM,
    generate_heatmap,
    generate_overlay,
    predict_and_explain,
    compute_heatmap_similarity,
    model_randomization_sanity_check,
    preprocess_image_for_model,
)


@pytest.fixture
def dummy_model():
    """Instantiate a lightweight model for deterministic test assertions."""
    model = create_model("mobilenet_v3_large", num_classes=11, pretrained=False)
    model.eval()
    return model


@pytest.fixture
def synthetic_image():
    """Create a synthetic 256x256 RGB image."""
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_gradcam_hook_registration_and_cleanup(dummy_model):
    """Verify forward and backward hooks register and clean up cleanly."""
    gradcam = GradCAM(dummy_model, target_layer="features.16")
    assert len(gradcam.hooks) == 2

    # Verify context manager or explicit remove_hooks
    gradcam.remove_hooks()
    assert len(gradcam.hooks) == 0


def test_gradcam_output_shape_and_bounds(dummy_model):
    """Verify CAM output matches input spatial dimensions and lies strictly in [0.0, 1.0]."""
    gradcam = GradCAM(dummy_model, target_layer="features.16")
    dummy_input = torch.randn(1, 3, 224, 224)

    cam, target_cls, conf, logits = gradcam.generate_cam(dummy_input, target_class=2)
    gradcam.remove_hooks()

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0
    assert target_cls == 2
    assert 0.0 <= conf <= 1.0
    assert logits.shape == (1, 11)


def test_heatmap_and_overlay_generation(synthetic_image):
    """Verify heatmap and overlay blending produce valid RGB uint8 images."""
    dummy_cam = np.linspace(0.0, 1.0, 224 * 224).reshape((224, 224))

    heatmap = generate_heatmap(dummy_cam, colormap="jet")
    assert heatmap.shape == (224, 224, 3)
    assert heatmap.dtype == np.uint8

    overlay, _ = generate_overlay(synthetic_image, dummy_cam, alpha=0.45)
    assert overlay.shape == (256, 256, 3)
    assert overlay.dtype == np.uint8


def test_predict_and_explain_api_contract(dummy_model, synthetic_image):
    """Verify reusable inference API returns all required contract fields."""
    dummy_classes = [f"Class_{i}" for i in range(11)]

    result = predict_and_explain(
        model=dummy_model,
        image=synthetic_image,
        class_names=dummy_classes,
    )

    required_keys = [
        "predicted_class",
        "confidence",
        "top2_predictions",
        "all_predictions",
        "original_image",
        "gradcam_heatmap",
        "gradcam_overlay",
        "raw_cam",
        "target_class_idx",
        "target_class_name",
    ]

    for k in required_keys:
        assert k in result, f"Missing required key: {k}"

    assert isinstance(result["predicted_class"], str)
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top2_predictions"]) == 2
    assert result["top2_predictions"][0]["confidence"] >= result["top2_predictions"][1]["confidence"]
    assert result["gradcam_heatmap"].shape == (224, 224, 3)
    assert result["gradcam_overlay"].shape == (224, 224, 3)
    assert result["raw_cam"].shape == (224, 224)


def test_compute_heatmap_similarity():
    """Verify correlation and MAE calculations on identical and inverted heatmaps."""
    map_a = np.ones((50, 50))
    map_a[:25, :] = 0.0  # Step pattern

    # Perfect identical maps
    sim_identical = compute_heatmap_similarity(map_a, map_a)
    assert pytest.approx(sim_identical["pearson_correlation"], 0.01) == 1.0
    assert pytest.approx(sim_identical["mean_absolute_error"], 0.01) == 0.0

    # Inverted maps
    map_b = 1.0 - map_a
    sim_inverted = compute_heatmap_similarity(map_a, map_b)
    assert pytest.approx(sim_inverted["pearson_correlation"], 0.01) == -1.0


def test_model_randomization_sanity_check(dummy_model, synthetic_image):
    """Verify Adebayo sanity check executes and returns valid structure."""
    result = model_randomization_sanity_check(
        trained_model=dummy_model,
        image=synthetic_image,
        random_seed=42,
    )

    assert "trained_cam" in result
    assert "randomized_cam" in result
    assert "similarity" in result
    assert "passed" in result
    assert isinstance(result["passed"], (bool, np.bool_))
    assert "pearson_correlation" in result["similarity"]
    assert "spearman_rank_correlation" in result["similarity"]
