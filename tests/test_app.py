"""Automated UI and workflow smoke tests for KP Smart Crop Doctor Streamlit Dashboard.

Tests:
1. Demo samples catalog integrity and image loading.
2. Crop and disease class formatting in English and Urdu.
3. End-to-end dashboard pipeline execution:
   Image -> Classification -> Top-2 -> Grad-CAM -> Advisory -> Citations -> English/Urdu.
4. Follow-up Q&A groundedness and guardrail abstention.
5. Headless Streamlit App execution smoke test via streamlit.testing.v1.AppTest.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from src import config
from app import extract_crop_name, get_cached_vision_model, get_cached_advisory_generator
from src.app.demo_samples import DEMO_SAMPLES, get_demo_sample_keys, load_demo_image
from src.app.ui_components import format_class_name
from src.explainability.gradcam import predict_and_explain, generate_overlay


# =============================================================================
# 1. Demo Samples Catalog Tests
# =============================================================================

def test_demo_samples_integrity():
    """Verify all 11 demo samples have valid keys, metadata, and existing files."""
    keys = get_demo_sample_keys()
    assert len(keys) == 11, f"Expected 11 demo sample keys, got {len(keys)}"

    for k in keys:
        meta = DEMO_SAMPLES[k]
        assert "crop" in meta
        assert "disease_class" in meta
        assert "label_en" in meta
        assert "label_ur" in meta
        assert "relative_path" in meta

        img = load_demo_image(k)
        assert img is not None, f"Failed to load demo image for key '{k}'"
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.width > 0 and img.height > 0


# =============================================================================
# 2. Text and Crop Formatting Helpers
# =============================================================================

def test_extract_crop_name():
    """Verify crop name extraction across various class label formats."""
    assert extract_crop_name("Apple___Apple_scab") == "Apple"
    assert extract_crop_name("Potato___Late_blight") == "Potato"
    assert extract_crop_name("Corn_(maize)___Common_rust_") == "Corn_(maize)"
    assert extract_crop_name("Corn_(maize)___Northern_Leaf_Blight") == "Corn_(maize)"


def test_format_class_name_bilingual():
    """Verify human-readable formatting in English and Urdu."""
    en_title = format_class_name("Potato___Late_blight", language="en")
    assert "Potato" in en_title
    assert "Late Blight" in en_title

    ur_title = format_class_name("Potato___Late_blight", language="ur")
    assert "آلو" in ur_title
    assert "پچھیتی جھلساؤ" in ur_title

    en_apple = format_class_name("Apple___Apple_scab", language="en")
    assert "Apple" in en_apple
    assert "Apple Scab" in en_apple

    ur_apple = format_class_name("Apple___Apple_scab", language="ur")
    assert "سیب" in ur_apple
    assert "پپڑی روگ" in ur_apple


# =============================================================================
# 3. End-to-End Workflow Pipeline Integration Test
# =============================================================================

def test_end_to_end_dashboard_workflow():
    """Test the complete 9-step diagnostic and advisory workflow on a real sample."""
    # 1. Load sample leaf
    img = load_demo_image("potato_late_blight")
    assert img is not None

    # 2. Load cached model & generator
    model, class_names = get_cached_vision_model()
    generator = get_cached_advisory_generator()

    # 3. Run prediction & Grad-CAM
    res = predict_and_explain(model, img, class_names=class_names, alpha=0.45)
    assert res["predicted_class"] == "Potato___Late_blight"
    assert res["confidence"] > 0.90
    assert len(res["top2_predictions"]) == 2
    assert res["gradcam_overlay"].shape == (224, 224, 3)
    assert res["gradcam_heatmap"].shape == (224, 224, 3)

    # 4. Generate dynamic re-blending
    overlay_custom, _ = generate_overlay(np.array(res["original_image"]), res["raw_cam"], alpha=0.75)
    assert overlay_custom.shape == (224, 224, 3)

    # 5. Generate English Advisory
    crop = extract_crop_name(res["predicted_class"])
    adv_en = generator.generate_advisory(
        query=f"Diagnostic symptoms, cause, and management for {crop} {res['predicted_class']}",
        crop=crop,
        disease_class=res["predicted_class"],
        confidence=res["confidence"],
        language="en",
    )
    assert adv_en.status == "grounded"
    assert adv_en.language == "en"
    assert adv_en.direction == "ltr"
    assert len(adv_en.key_symptoms) > 0
    assert len(adv_en.recommended_management) > 0
    assert len(adv_en.source_citations) > 0
    assert any("KP-AGRI-POT-002" in c for c in adv_en.source_citations)

    # 6. Generate Urdu Advisory
    adv_ur = generator.generate_advisory(
        query=f"علامات اور تدارک",
        crop=crop,
        disease_class=res["predicted_class"],
        confidence=res["confidence"],
        language="ur",
    )
    assert adv_ur.status == "grounded"
    assert adv_ur.language == "ur"
    assert adv_ur.direction == "rtl"
    assert len(adv_ur.key_symptoms) > 0
    assert len(adv_ur.recommended_management) > 0
    # Citations must be preserved verbatim
    assert adv_ur.source_citations == adv_en.source_citations


# =============================================================================
# 4. Follow-Up Q&A and Guardrail Rejection Test
# =============================================================================

def test_followup_qa_and_guardrail():
    """Verify follow-up Q&A supports relevant questions and rejects out-of-scope crops."""
    generator = get_cached_advisory_generator()

    # Valid supported follow-up question
    qa_valid = generator.generate_advisory(
        query="What certified seed practices prevent late blight in Swat?",
        crop="Potato",
        disease_class="Potato___Late_blight",
        language="en",
    )
    assert qa_valid.status == "grounded"
    assert len(qa_valid.source_citations) > 0

    # Out-of-scope follow-up question (Wheat yellow rust)
    qa_invalid = generator.generate_advisory(
        query="What fungicide spray is recommended for wheat yellow rust?",
        crop=None,
        disease_class=None,
        language="en",
    )
    assert qa_invalid.status == "information_unavailable"
    assert qa_invalid.evidence_chunks_count == 0
    assert len(qa_invalid.source_citations) == 0
    assert "Wheat" in qa_invalid.refusal_reason


# =============================================================================
# 5. Headless Streamlit App Execution Smoke Test
# =============================================================================

def test_streamlit_app_headless_smoke():
    """Run app.py via Streamlit's AppTest simulator to verify zero exceptions on initial load."""
    from streamlit.testing.v1 import AppTest

    app_path = config.PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Verify no unhandled exception occurred during script run
    assert not at.exception, f"Streamlit app raised an exception: {at.exception}"
    # Verify title or header was rendered
    assert len(at.markdown) > 0
    # Verify sidebar elements rendered
    assert len(at.sidebar.radio) >= 1
