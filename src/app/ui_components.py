"""Modular UI components and layout helpers for KP Smart Crop Doctor Streamlit Dashboard.

Supports:
1. Bilingual styling (English LTR and fluent Urdu RTL typography).
2. Prediction metric cards with Top-2 comparison and calibration disclaimers.
3. Grad-CAM interactive explanation view with dynamic opacity blending.
4. Structured agronomic advisory tabs with institutional citation links.
5. Follow-up conversational Q&A interface with scope guardrails.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import streamlit as st

from src.explainability.gradcam import generate_overlay
from src.rag.generator import StructuredAdvisory, GroundedAdvisoryGenerator


def inject_custom_css(language: str = "en") -> None:
    """Inject responsive custom CSS styling with bilingual typography and RTL support."""
    is_urdu = (language == "ur")

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Nastaliq+Urdu:wght@400;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {{
        font-family: {"'Noto Nastaliq Urdu', 'Segoe UI', Tahoma, sans-serif" if is_urdu else "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"};
        {"direction: rtl !important; text-align: right !important;" if is_urdu else "direction: ltr; text-align: left;"}
    }}

    /* Title and Subtitle */
    .app-header-title {{
        color: #1b5e20;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
        {"direction: rtl; text-align: right;" if is_urdu else "direction: ltr; text-align: left;"}
    }}
    .app-header-subtitle {{
        color: #455a64;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
        {"direction: rtl; text-align: right;" if is_urdu else "direction: ltr; text-align: left;"}
    }}

    /* Card styling */
    .metric-card {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .metric-title {{
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin: 4px 0;
    }}
    .metric-badge-high {{
        display: inline-block;
        background: #dcfce7;
        color: #15803d;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .metric-badge-warning {{
        display: inline-block;
        background: #fef3c7;
        color: #b45309;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }}

    /* Advisory sections */
    .advisory-box {{
        background: #ffffff;
        border-left: {"none" if is_urdu else "4px solid #16a34a"};
        border-right: {"4px solid #16a34a" if is_urdu else "none"};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        border-top: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }}
    .advisory-title {{
        font-weight: 700;
        color: #1e293b;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }}

    /* Citation badge */
    .citation-tag {{
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
        direction: ltr !important;
        unicode-bidi: isolate !important;
    }}

    /* Disclaimer banner */
    .disclaimer-banner {{
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 18px;
        color: #92400e;
        font-size: 0.88rem;
        {"direction: rtl; text-align: right;" if is_urdu else "direction: ltr; text-align: left;"}
    }}

    /* Urdu text enhancements */
    .urdu-text {{
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', Arial, sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
        line-height: 2.1 !important;
        font-size: 1.05rem !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header(language: str = "en") -> None:
    """Render application header and subtitle in selected language."""
    if language == "ur":
        st.markdown(
            """
            <div class="app-header-title">🌱 کے پی سمارٹ کراپ ڈاکٹر</div>
            <div class="app-header-subtitle">
                خیبر پختونخوا کے لیے قابلِ فہم اور تصدیق شدہ زرعی تشخیصی نظام (سیب، مکئی، آلو)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="app-header-title">🌱 KP Smart Crop Doctor</div>
            <div class="app-header-subtitle">
                Explainable, Knowledge-Grounded Agricultural Advisory System for Khyber Pakhtunkhwa (Apple, Maize, Potato)
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_disclaimer(language: str = "en") -> None:
    """Render prominent academic prototype and agricultural advisory disclaimer."""
    if language == "ur":
        st.markdown(
            """
            <div class="disclaimer-banner">
                ⚠️ <strong>ضروری انتباہ:</strong> یہ نظام زرعی تحقیق اور باغبانی کی مدد کے لیے ایک تعلیمی اور تشخیصی پروٹوٹائپ ہے۔ 
                کسی بھی کیڑے مار دوا یا کیمیائی سپرے کے عملی استعمال سے قبل اپنی تحصیل کے مقامی محکمہ زراعت توسیع (KP Agri Extension) کے افسر سے لازمی تصدیق کریں۔
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="disclaimer-banner">
                ⚠️ <strong>Decision-Support Notice:</strong> KP Smart Crop Doctor is an AI research prototype designed for agricultural decision support. 
                Before applying chemical sprays or significant crop interventions, always verify diagnoses and product registrations with your local KP Agriculture Extension officer.
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_class_name(raw_name: str, language: str = "en") -> str:
    """Format raw dataset class name (e.g. 'Potato___Late_blight') into clean readable text."""
    parts = raw_name.split("___")
    crop = parts[0].replace("_", " ").replace("Corn (maize)", "Maize")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

    if language == "ur":
        crop_ur_map = {
            "Apple": "سیب",
            "Corn (maize)": "مکئی",
            "Maize": "مکئی",
            "Potato": "آلو",
        }
        disease_ur_map = {
            "Apple scab": "پپڑی روگ (ایپل سکيب)",
            "Black rot": "سیاہ سڑن (بلیک راٹ)",
            "Cedar apple rust": "سیب اور صنوبر کا زنگ",
            "healthy": "صحت مند پتا",
            "Cercospora leaf spot Gray leaf spot": "سرمئی پتا دھبہ روگ (گرے لیف سپاٹ)",
            "Common rust": "عام زنگ (کامن رسٹ)",
            "Northern Leaf Blight": "شمالی پتوں کا جھلساؤ (ناردن لیف بلائٹ)",
            "Early blight": "اگیتی جھلساؤ (ارلی بلائٹ)",
            "Late blight": "پچھیتی جھلساؤ (لیٹ بلائٹ)",
        }
        clean_crop = crop_ur_map.get(crop, crop)
        clean_disease = disease_ur_map.get(disease, disease)
        return f"{clean_crop}: {clean_disease}"

    return f"{crop}: {disease.title()}"


def render_prediction_card(prediction_result: Dict[str, Any], language: str = "en") -> None:
    """Render prediction metrics, confidence score, and top-2 comparison bars."""
    top1_name = prediction_result["predicted_class"]
    top1_conf = prediction_result["confidence"]
    top2_preds = prediction_result.get("top2_predictions", [])

    formatted_title = format_class_name(top1_name, language=language)
    conf_pct = top1_conf * 100.0

    is_high_conf = conf_pct >= 85.0
    badge_class = "metric-badge-high" if is_high_conf else "metric-badge-warning"
    badge_label = ("اعلیٰ اعتماد" if is_high_conf else "معتدل اعتماد") if language == "ur" else ("High Confidence" if is_high_conf else "Moderate Confidence")

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{'تشخیص شدہ کیفیت' if language == 'ur' else 'Primary Vision Diagnosis'}</div>
            <div class="metric-value">{formatted_title}</div>
            <div>
                <span class="{badge_class}">{badge_label}: {conf_pct:.1f}%</span>
                <span style="font-size: 0.8rem; color: #64748b; margin-left: 8px;">
                    {'ماڈل کا حسابی تناسب (Softmax)' if language == 'ur' else 'Softmax confidence estimate'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top-2 Comparison Section
    st.markdown(
        f"<strong>{'اوّلین ۲ احتمالات (Top-2 Predictions):' if language == 'ur' else 'Top-2 Class Probability Comparison:'}</strong>",
        unsafe_allow_html=True,
    )

    for i, pred in enumerate(top2_preds):
        p_name = format_class_name(pred["class_name"], language=language)
        p_conf = pred["confidence"]
        p_pct = p_conf * 100.0

        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(min(max(p_conf, 0.0), 1.0))
        with col2:
            st.markdown(f"**{p_pct:.1f}%** - {p_name}")


def render_gradcam_view(
    prediction_result: Dict[str, Any],
    alpha: float = 0.45,
    language: str = "en",
) -> None:
    """Render 3-column Grad-CAM explanation: Original, Heatmap, and Blended Overlay."""
    orig_img = prediction_result["original_image"]
    raw_cam = prediction_result["raw_cam"]

    # Re-blend dynamically using the user's alpha slider
    orig_np = np.array(orig_img)
    overlay_rgb, heatmap_rgb = generate_overlay(orig_np, raw_cam, alpha=alpha)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(
            orig_img,
            caption="اصل پتہ (Original Leaf)" if language == "ur" else "1. Original Leaf Image",
            use_container_width=True,
        )
    with c2:
        st.image(
            heatmap_rgb,
            caption="حرارتی نقشہ (Grad-CAM Heatmap)" if language == "ur" else "2. Grad-CAM Activation Heatmap",
            use_container_width=True,
        )
    with c3:
        st.image(
            overlay_rgb,
            caption=f"مشترکہ تصویر (Overlay, α={alpha:.2f})" if language == "ur" else f"3. Blended Heatmap Overlay (α={alpha:.2f})",
            use_container_width=True,
        )

    explanation_text = (
        "🔍 <strong>بصری وضاحت (Explainable AI):</strong> گرم رنگ (سرخ اور زرد) پتے کے ان اہم حصوں کی نشان دہی کرتے ہیں جن پر ماڈل نے بیماری کی شناخت کے لیے سب سے زیادہ توجہ مرکوز کی۔ ٹھنڈے رنگ (نیلے) غیر متعلقہ پس منظر کی عکاسی کرتے ہیں۔"
        if language == "ur"
        else "🔍 <strong>Explainable AI Inspection:</strong> Warm colors (red and yellow) highlight the primary visual features (lesions, margins, pustules) that drove the model's prediction. Cool colors (blue) represent background or unweighted leaf tissue."
    )
    st.markdown(f"<div style='font-size: 0.88rem; color: #475569; margin-top: 6px;'>{explanation_text}</div>", unsafe_allow_html=True)


def render_structured_advisory(advisory: StructuredAdvisory, language: str = "en") -> None:
    """Render grounded agricultural advisory with structured tabs and citation badges."""
    is_ur = (language == "ur")

    if advisory.status == "information_unavailable":
        refusal_msg = advisory.uncertainty_limitations or "The requested agricultural information is outside the supported KP knowledge base."
        st.warning(f"🚫 {refusal_msg}")
        return

    # Tabs for structured advisory sections
    tab_labels = (
        ["تشخیص و سبب", "اہم علامات", "علاج و تدارک", "حفاظتی تدابیر", "فصل کا مرحلہ", "سرکاری حوالہ جات"]
        if is_ur
        else ["Diagnosis & Cause", "Key Symptoms", "Management & Control", "Prevention & Sanitation", "Crop-Stage Guidance", "Institutional Citations"]
    )

    t1, t2, t3, t4, t5, t6 = st.tabs(tab_labels)

    with t1:
        st.markdown(f"### {advisory.disease}")
        if advisory.likely_cause:
            st.markdown(
                f"<div class='advisory-box'><strong>{'پیتھوجن / حیاتیاتی سبب:' if is_ur else 'Pathogen / Biological Cause:'}</strong><br/>{advisory.likely_cause}</div>",
                unsafe_allow_html=True,
            )
        st.info(advisory.uncertainty_limitations)

    with t2:
        st.markdown(f"### {'تشخیصی علامات' if is_ur else 'Diagnostic Symptoms'}")
        if advisory.key_symptoms:
            for s in advisory.key_symptoms:
                st.markdown(f"- {s}")
        else:
            st.write("تفصیلی علامات کے لیے بنیادی حوالہ ملاحظہ کریں۔" if is_ur else "Refer to primary citation for symptom checklist.")

    with t3:
        st.markdown(f"### {'مجوزہ کنٹرول اور ادویات' if is_ur else 'Recommended Management & Control'}")
        st.markdown(
            f"<div style='font-size: 0.85rem; color: #b45309; margin-bottom: 8px;'>⚠️ {'صرف منظور شدہ اور رجسٹرڈ طریقہ کار استعمال کریں:' if is_ur else 'Use only verified, registered agricultural extension treatments:'}</div>",
            unsafe_allow_html=True,
        )
        if advisory.recommended_management:
            for m in advisory.recommended_management:
                st.markdown(f"- {m}")
        else:
            st.write("مقامی زرعی افسر سے تصدیق شدہ علاج معلوم کریں۔" if is_ur else "Consult local extension officer for recommended treatments.")

    with t4:
        st.markdown(f"### {'حفاظتی اقدامات اور زرعی طریقے' if is_ur else 'Preventive Cultural Practices'}")
        if advisory.prevention:
            for p in advisory.prevention:
                st.markdown(f"- {p}")
        else:
            st.write("فصلوں کا ہیر پھیر اور تصدیق شدہ بیج استعمال کریں۔" if is_ur else "Practice crop rotation and use certified seed.")

    with t5:
        st.markdown(f"### {'فصل کے نازک مراحل' if is_ur else 'Critical Crop Stages'}")
        stage_text = advisory.crop_stage_guidance or ("نم آلود مراحل کے دوران باقاعدہ نگرانی کریں۔" if is_ur else "Monitor foliage during humid vegetative and reproductive stages.")
        st.markdown(f"<div class='advisory-box'>{stage_text}</div>", unsafe_allow_html=True)

    with t6:
        st.markdown(f"### {'سرکاری زرعی ماخذ اور حوالہ جات' if is_ur else 'Curated Institutional Provenance Citations'}")
        st.markdown(
            f"<p style='color: #64748b; font-size: 0.88rem;'>{'اس مشورے کے تمام نکات درج ذیل تصدیق شدہ دستاویزات سے اخذ کیے گئے ہیں:' if is_ur else 'All agronomic recommendations above are grounded strictly in the following curated institutional sources:'}</p>",
            unsafe_allow_html=True,
        )
        for cite in advisory.source_citations:
            st.markdown(f"<span class='citation-tag'>{cite}</span>", unsafe_allow_html=True)


def render_followup_qa(
    generator: GroundedAdvisoryGenerator,
    context_crop: Optional[str] = None,
    context_disease: Optional[str] = None,
    language: str = "en",
) -> None:
    """Render interactive follow-up Q&A assistant grounded in RAG with scope guardrails."""
    is_ur = (language == "ur")

    st.markdown("---")
    st.markdown(
        f"### {'💬 فالو اپ زرعی سوالات پوچھیں' if is_ur else '💬 Follow-up Agricultural Advisory Q&A'}"
    )
    st.caption(
        "سیب، مکئی یا آلو سے متعلق سوال پوچھیں۔ دیگر فصلیں خودکار سیکیورٹی گارڈریل کے تحت مسترد کر دی جائیں گی۔"
        if is_ur
        else "Ask follow-up agronomic questions grounded in curated KP knowledge. Questions outside supported crops (Apple, Maize, Potato) will be safely rejected."
    )

    # Pre-canned suggested query buttons
    suggested_queries = (
        [
            "اس بیماری کے لیے کیڑے مار ادویات کے متبادل کون سے طریقے ہیں؟",
            "کونپلیں نکلنے کے مرحلے پر کیا حفاظتی سپرے تجویز کی جاتی ہے؟",
            "گندم کے زنگ کی دوا کیا ہے؟ (غیر معاون فصل ٹیسٹ)",
        ]
        if is_ur
        else [
            f"What certified seed practices prevent {context_disease or 'this condition'}?",
            f"What are the organic sanitation steps for {context_crop or 'this crop'}?",
            "What fungicide cures wheat yellow rust? (Out-of-Scope Test)",
        ]
    )

    cols = st.columns(len(suggested_queries))
    selected_suggestion = None
    for idx, (col, sq) in enumerate(zip(cols, suggested_queries)):
        with col:
            if st.button(sq, key=f"sq_btn_{idx}", use_container_width=True):
                selected_suggestion = sq

    # Text input
    input_placeholder = "اپنا زرعی سوال یہاں درج کریں..." if is_ur else "Enter your agronomic question here..."
    user_query = st.text_input(
        "سوال (Question):" if is_ur else "Agronomic Question:",
        value=selected_suggestion or "",
        placeholder=input_placeholder,
        key="qa_input_box",
    )

    if st.button("رہنمائی حاصل کریں" if is_ur else "Get Grounded Advisory", key="submit_qa_btn", type="primary"):
        if not user_query.strip():
            st.warning("براہ کرم سوال درج کریں۔" if is_ur else "Please enter a question.")
            return

        with st.spinner("تصدیق شدہ علمی ذخیرے سے تلاش جاری ہے..." if is_ur else "Retrieving verified agronomic evidence..."):
            qa_advisory = generator.generate_advisory(
                query=user_query,
                crop=context_crop,
                disease_class=context_disease,
                confidence=None,
                language=language,
            )

        if qa_advisory.status == "information_unavailable":
            st.error(f"🚫 {qa_advisory.uncertainty_limitations}")
        else:
            st.success("✅ تصدیق شدہ مشورہ حاصل ہوا:" if is_ur else "✅ Grounded Advisory Retrieved:")
            render_structured_advisory(qa_advisory, language=language)
