"""KP Smart Crop Doctor - Interactive Streamlit Web Application Dashboard.

An explainable, knowledge-grounded agricultural advisory system for Khyber Pakhtunkhwa (Apple, Maize, Potato).
Unifies:
1. Transfer Learning Classification (MobileNetV3-Large)
2. Explainable AI Visual Heatmaps (Grad-CAM on features.16)
3. Source-Grounded RAG Retrieval (FAISS vector store)
4. Structured Multilingual Agronomic Advisories (English & Urdu)
5. Constrained Agricultural Q&A with Guardrail Abstention
"""

import os
import sys
from pathlib import Path
from typing import Optional
from PIL import Image

# Ensure offline cache loading for fast local startup
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch

from src import config
from src.utils import get_logger
from src.explainability.evaluate_gradcam import load_trained_model
from src.explainability.gradcam import predict_and_explain, load_label_encoder
from src.rag.generator import GroundedAdvisoryGenerator, StructuredAdvisory
from src.app.demo_samples import DEMO_SAMPLES, get_demo_sample_keys, load_demo_image
from src.app.ui_components import (
    inject_custom_css,
    render_header,
    render_disclaimer,
    render_prediction_card,
    render_gradcam_view,
    render_structured_advisory,
    render_followup_qa,
)

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# 1. Caching & Model Initializers
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_cached_vision_model():
    """Load and cache the trained MobileNetV3-Large vision model on CPU."""
    logger.info("Initializing cached MobileNetV3-Large model...")
    model = load_trained_model(device=torch.device("cpu"))
    class_names = load_label_encoder()
    return model, class_names


@st.cache_resource(show_spinner=False)
def get_cached_advisory_generator():
    """Load and cache the GroundedAdvisoryGenerator with FAISS index."""
    logger.info("Initializing cached GroundedAdvisoryGenerator...")
    return GroundedAdvisoryGenerator()


def extract_crop_name(class_name: str) -> str:
    """Extract standard crop name from full class label (e.g. 'Corn_(maize)___Common_rust_')."""
    raw_crop = class_name.split("___")[0]
    if "Corn" in raw_crop or "maize" in raw_crop:
        return "Corn_(maize)"
    elif "Apple" in raw_crop:
        return "Apple"
    elif "Potato" in raw_crop:
        return "Potato"
    return raw_crop


# -----------------------------------------------------------------------------
# 2. Main Streamlit Application
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="KP Smart Crop Doctor",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -------------------------------------------------------------------------
    # Sidebar Controls
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🌐 زبان / Language")
        lang_choice = st.radio(
            "Select Interface Language:",
            options=["English", "اردو (Urdu)"],
            index=0,
            label_visibility="collapsed",
        )
        is_urdu = ("Urdu" in lang_choice or "اردو" in lang_choice)
        lang_code = "ur" if is_urdu else "en"

        st.markdown("---")
        st.markdown("### 📷 " + ("تصویر کا ماخذ" if is_urdu else "Input Source"))

        input_mode = st.radio(
            "Choose Image Source:" if not is_urdu else "تصویر منتخب کرنے کا طریقہ:",
            options=[
                "نمونہ تصویر (Demo Samples)" if is_urdu else "Select Sample Demo Leaf",
                "اپنی تصویر اپلوڈ کریں (Upload Image)" if is_urdu else "Upload Your Own Leaf Image",
            ],
            index=0,
        )

        selected_image: Optional[Image.Image] = None
        selected_sample_key: Optional[str] = None

        if "Demo" in input_mode or "نمونہ" in input_mode:
            sample_keys = get_demo_sample_keys()
            sample_labels = [
                (DEMO_SAMPLES[k]["label_ur"] if is_urdu else DEMO_SAMPLES[k]["label_en"])
                for k in sample_keys
            ]
            chosen_idx = st.selectbox(
                "منتخب نمونہ پتا:" if is_urdu else "Choose Sample Disease Leaf:",
                range(len(sample_keys)),
                format_func=lambda i: sample_labels[i],
                index=0,
            )
            selected_sample_key = sample_keys[chosen_idx]
            selected_image = load_demo_image(selected_sample_key, base_dir=PROJECT_ROOT)
            if selected_image:
                sample_desc = DEMO_SAMPLES[selected_sample_key]["description_ur" if is_urdu else "description_en"]
                st.caption(f"ℹ️ {sample_desc}")

        else:
            uploaded_file = st.file_uploader(
                "پتے کی تصویر اپلوڈ کریں (JPG, PNG):" if is_urdu else "Upload Leaf Photograph (JPG, PNG):",
                type=["jpg", "jpeg", "png", "JPG", "PNG"],
                help="Upload a close-up photograph of a diseased or healthy crop leaf.",
            )
            if uploaded_file is not None:
                try:
                    selected_image = Image.open(uploaded_file).convert("RGB")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    selected_image = None

        st.markdown("---")
        st.markdown("### 🎛️ " + ("ایکس رے کنٹرولز" if is_urdu else "XAI Controls"))
        alpha_slider = st.slider(
            "حرارتی نقشے کا امتزاج (Heatmap Opacity α):" if is_urdu else "Grad-CAM Heatmap Blend (α):",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Adjust the transparency of the Grad-CAM activation heatmap overlaid on the leaf.",
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #64748b;">
                <strong>KP Smart Crop Doctor v1.0</strong><br/>
                Backbone: MobileNetV3-Large<br/>
                XAI: Grad-CAM (features.16)<br/>
                RAG: FAISS + all-MiniLM-L6-v2<br/>
                Region: Khyber Pakhtunkhwa, Pakistan
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Main Page Layout
    # -------------------------------------------------------------------------
    inject_custom_css(language=lang_code)
    render_header(language=lang_code)
    render_disclaimer(language=lang_code)

    if selected_image is None:
        # Prompt user to choose an image
        st.info(
            "👈 براہ کرم سائیڈ بار سے نمونہ پتا منتخب کریں یا پتے کی تصویر اپلوڈ کریں۔"
            if is_urdu
            else "👈 Please select a sample diseased leaf or upload your own leaf image from the sidebar to begin diagnosis."
        )
        return

    # Load resources with cached helpers
    with st.spinner("ماڈل اور تشخیصی نظام لوڈ ہو رہا ہے..." if is_urdu else "Loading model checkpoint and knowledge base..."):
        model, class_names = get_cached_vision_model()
        generator = get_cached_advisory_generator()

    # Step 1 & 2: Prediction and Grad-CAM Generation
    with st.spinner("تصویر کا تجزیہ اور حرارتی نقشہ تیار کیا جا رہا ہے..." if is_urdu else "Running vision inference and generating Grad-CAM heatmap..."):
        try:
            prediction_result = predict_and_explain(
                model=model,
                image=selected_image,
                class_names=class_names,
                alpha=alpha_slider,
            )
        except Exception as e:
            st.error(f"Inference error: {e}")
            return

    pred_class = prediction_result["predicted_class"]
    confidence = prediction_result["confidence"]
    crop_name = extract_crop_name(pred_class)

    # Step 3: Render Classification and Top-2 Prediction Metrics
    render_prediction_card(prediction_result, language=lang_code)

    st.markdown("---")
    st.markdown(
        f"### {'🔬 بصری وضاحت (Grad-CAM Explainable AI)' if is_urdu else '🔬 Visual Explainability (Grad-CAM Feature Activation)'}"
    )
    # Step 4: Render Grad-CAM 3-Column Inspection View
    render_gradcam_view(prediction_result, alpha=alpha_slider, language=lang_code)

    st.markdown("---")
    st.markdown(
        f"### {'📚 تصدیق شدہ زرعی رہنمائی و تدارک (Curated Agronomic Advisory)' if is_urdu else '📚 Grounded Agronomic Advisory & Management Guide'}"
    )

    # Step 5: RAG Advisory Generation
    with st.spinner("خیبر پختونخوا کے تصدیق شدہ زرعی ذخیرے سے رہنمائی اخذ کی جا رہی ہے..." if is_urdu else "Retrieving grounded advisory from KP Agricultural Extension knowledge base..."):
        query_prompt = (
            f"What are the diagnostic symptoms, causes, and recommended management for {crop_name} {pred_class}?"
        )
        advisory: StructuredAdvisory = generator.generate_advisory(
            query=query_prompt,
            crop=crop_name,
            disease_class=pred_class,
            confidence=confidence,
            language=lang_code,
        )

    # Step 6: Render Structured Advisory Tabs
    render_structured_advisory(advisory, language=lang_code)

    # Step 7: Interactive Follow-up Q&A
    render_followup_qa(
        generator=generator,
        context_crop=crop_name,
        context_disease=pred_class,
        language=lang_code,
    )


if __name__ == "__main__":
    main()
