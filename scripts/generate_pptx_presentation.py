"""Generate the Final Capstone Presentation for KP Smart Crop Doctor in PPTX format.

Builds an 18-22 slide widescreen (16:9) presentation using python-pptx.
Incorporates actual figures from evaluation/figures/, metric cards, concise bullets,
and tables suitable for CS/AI and agronomy examiners.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from src import config

# Palette
DARK_NAVY = RGBColor(0x0A, 0x25, 0x40)
FOREST_GREEN = RGBColor(0x1B, 0x4D, 0x3E)
SLATE_GRAY = RGBColor(0x4A, 0x55, 0x68)
LIGHT_BG = RGBColor(0xF7, 0xFA, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT_BLUE = RGBColor(0x2B, 0x6C, 0xB0)
ACCENT_AMBER = RGBColor(0xD9, 0x77, 0x06)
CARD_BG = RGBColor(0xED, 0xF2, 0xF7)
BORDER_GRAY = RGBColor(0xCB, 0xD5, 0xE0)

def add_header(slide, title_text, category_text="KP SMART CROP DOCTOR"):
    """Add standardized clean header to slide."""
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = FOREST_GREEN

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.5), Inches(0.65))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(20)
    p_title.font.bold = True
    p_title.font.color.rgb = DARK_NAVY

def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_GRAY):
    """Add rounded-style card box for content grouping."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1)
    return shape

def add_metric_box(slide, left, top, width, height, value_text, label_text, sub_text=""):
    """Add a bold KPI metric callout card."""
    shape = add_card(slide, left, top, width, height, bg_color=WHITE, border_color=FOREST_GREEN)
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    
    p_val = tf.paragraphs[0]
    p_val.text = value_text
    p_val.font.size = Pt(22)
    p_val.font.bold = True
    p_val.font.color.rgb = FOREST_GREEN
    p_val.alignment = PP_ALIGN.CENTER
    
    p_lbl = tf.add_paragraph()
    p_lbl.text = label_text
    p_lbl.font.size = Pt(10)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = DARK_NAVY
    p_lbl.alignment = PP_ALIGN.CENTER

    if sub_text:
        p_sub = tf.add_paragraph()
        p_sub.text = sub_text
        p_sub.font.size = Pt(8.5)
        p_sub.font.color.rgb = SLATE_GRAY
        p_sub.alignment = PP_ALIGN.CENTER

def build_presentation():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # -------------------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE (Dark Navy Theme)
    # -------------------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = DARK_NAVY
    bg1.line.fill.background()

    tb1 = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(3.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "AI/ML CAPSTONE DEFENSE PRESENTATION"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = FOREST_GREEN
    p0.space_after = Pt(14)

    p1 = tf1.add_paragraph()
    p1.text = "KP Smart Crop Doctor"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_after = Pt(8)

    p2 = tf1.add_paragraph()
    p2.text = "An Explainable, Knowledge-Grounded, Multilingual Plant Disease Decision-Support System for Khyber Pakhtunkhwa Agriculture"
    p2.font.size = Pt(15)
    p2.font.color.rgb = RGBColor(0xDF, 0xE2, 0xE6)
    p2.space_after = Pt(24)

    p3 = tf1.add_paragraph()
    p3.text = "Candidate Capstone Team | Department of Computer Science & AI | Academic Year 2025–2026"
    p3.font.size = Pt(11)
    p3.font.color.rgb = RGBColor(0xA0, 0xAE, 0xC0)

    # -------------------------------------------------------------------------
    # SLIDE 2: THE PROBLEM (Three Failure Modes)
    # -------------------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "The Diagnostic Dilemma in Rural Agriculture", "PROBLEM STATEMENT")

    # Card 1: Black Box
    add_card(slide2, Inches(0.8), Inches(1.6), Inches(3.6), Inches(4.8))
    tb_c1 = slide2.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(3.2), Inches(4.4))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "1. Black-Box Opacity"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_c1.add_paragraph()
    p.text = "\n• Standard CNNs predict class labels without spatial justification.\n• Farmers cannot verify if prediction is based on true lesions or background soil/fingers.\n• Severe trust deficit prevents clinical and field adoption."
    p.font.size = Pt(10.5); p.font.color.rgb = SLATE_GRAY

    # Card 2: Generative Hallucination
    add_card(slide2, Inches(4.8), Inches(1.6), Inches(3.6), Inches(4.8))
    tb_c2 = slide2.shapes.add_textbox(Inches(5.0), Inches(1.8), Inches(3.2), Inches(4.4))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "2. LLM Hallucinations"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_c2.add_paragraph()
    p.text = "\n• Generic LLMs invent unverified chemical formulations and toxic dosages.\n• Recommend banned pesticides or chemicals unavailable in Pakistan.\n• Ignore provincial agro-climatic spray windows."
    p.font.size = Pt(10.5); p.font.color.rgb = SLATE_GRAY

    # Card 3: Language & Extension Deficit
    add_card(slide2, Inches(8.8), Inches(1.6), Inches(3.7), Inches(4.8))
    tb_c3 = slide2.shapes.add_textbox(Inches(9.0), Inches(1.8), Inches(3.3), Inches(4.4))
    tf_c3 = tb_c3.text_frame
    tf_c3.word_wrap = True
    p = tf_c3.paragraphs[0]
    p.text = "3. Severe Extension Shortage"
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_c3.add_paragraph()
    p.text = "\n• < 1 extension officer per 1,500 farming families in rural KP.\n• Diagnostics delayed until necrosis is irreversible.\n• Language barrier: literature in English; farmers speak Urdu/Pashto."
    p.font.size = Pt(10.5); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 3: MOTIVATION & KP CONTEXT
    # -------------------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Regional Agro-Ecological Significance of KP", "MOTIVATION & SCOPE")

    add_metric_box(slide3, Inches(0.8), Inches(1.5), Inches(3.6), Inches(1.5), "> 20%", "Contribution to KP GDP", "Employs ~40% of provincial labor force")
    add_metric_box(slide3, Inches(4.8), Inches(1.5), Inches(3.6), Inches(1.5), "30% – 70%", "Potential Crop Losses", "Apple Scab & Late Blight epidemic risk")
    add_metric_box(slide3, Inches(8.8), Inches(1.5), Inches(3.7), Inches(1.5), "3 Target Crops", "Apple, Maize, Potato", "Selected for economic and nutritional security")

    add_card(slide3, Inches(0.8), Inches(3.4), Inches(11.7), Inches(3.4), bg_color=WHITE)
    tb_m = slide3.shapes.add_textbox(Inches(1.1), Inches(3.6), Inches(11.1), Inches(3.0))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True
    p = tf_m.paragraphs[0]
    p.text = "Geographic Focus across Khyber Pakhtunkhwa Valleys & Plains:"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_m.add_paragraph()
    p.text = "• Swat, Upper Dir & Hazara: Commercial apple orchards heavily threatened by Apple Scab (Venturia inaequalis), Black Rot, and Cedar Apple Rust.\n" \
             "• Swabi, Mardan & Nowshera: Major Kharif maize production vulnerable to Gray Leaf Spot (Cercospora zeae-maydis) and Northern Corn Leaf Blight (Exserohilum turcicum).\n" \
             "• Kalam, Kaghan & Battagram: Highland seed and table potato clusters vulnerable to sudden, devastating Late Blight (Phytophthora infestans) epidemics.\n" \
             "• Institutional Backing: Anchored directly on research bulletins from ARI Tarnab, CCRI Pirsabak, and PRC Abbottabad."
    p.font.size = Pt(11); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 4: RESEARCH OBJECTIVES
    # -------------------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Project Goals & Concrete Research Objectives", "SYSTEM OBJECTIVES")

    objs = [
        ("Obj 1: Edge Vision Backbone", "Train MobileNetV3-Large on 11 classes across Apple, Maize, and Potato with zero data leakage, high Macro-F1 (>0.95), and sub-50ms CPU inference."),
        ("Obj 2: Spatial Interpretability", "Implement Grad-CAM on layer features.16 to visualize discriminative lesion regions, validated via parameter-randomization sanity checks."),
        ("Obj 3: Localized Agricultural RAG", "Index 11 verified KP extension bulletins in a FAISS vector store for fast, traceable retrieval without relying on ungrounded generative models."),
        ("Obj 4: Grounded Advisory & Safety", "Synthesize bilingual diagnostic cards with 100% provenance citations and strict abstention guardrails for out-of-scope crops."),
        ("Obj 5: Interactive Web Prototype", "Deploy an accessible Streamlit interface supporting dynamic Grad-CAM opacity adjustment, English/Urdu RTL toggling, and offline execution."),
    ]
    top_pos = 1.6
    for title, desc in objs:
        add_card(slide4, Inches(0.8), Inches(top_pos), Inches(11.7), Inches(0.95), bg_color=WHITE)
        tb = slide4.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.08), Inches(11.3), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title + " — "
        p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = FOREST_GREEN
        run = p.add_run()
        run.text = desc
        run.font.size = Pt(10.5); run.font.bold = False; run.font.color.rgb = SLATE_GRAY
        top_pos += 1.1

    # -------------------------------------------------------------------------
    # SLIDE 5: SYSTEM ARCHITECTURE
    # -------------------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "End-to-End Multi-Stage Pipeline Architecture", "SYSTEM ARCHITECTURE")

    pipeline_steps = [
        ("1. Input", "Leaf image upload or demo sample selection"),
        ("2. Preprocessing", "256x256 resize, 224x224 crop, ImageNet norm"),
        ("3. Vision Model", "MobileNetV3-Large forward pass (4.2M params)"),
        ("4. Top-2 & ECE", "Softmax logits, Top-2 confidence, calibration note"),
        ("5. Grad-CAM", "features.16 hooks, gradient weighting, overlay"),
        ("6. Query Dispatch", "Crop + Disease context passed to retriever"),
        ("7. FAISS Search", "all-MiniLM-L6-v2 embeddings, IndexFlatIP"),
        ("8. Grounded Synth", "Extractive advisory: Symptoms, Cause, IPM"),
        ("9. Bilingual UI", "English & Urdu RTL Nastaliq in Streamlit"),
    ]
    left_p = 0.8
    top_p = 1.6
    for idx, (stitle, sdesc) in enumerate(pipeline_steps):
        add_card(slide5, Inches(left_p), Inches(top_p), Inches(3.6), Inches(1.5), bg_color=WHITE, border_color=ACCENT_BLUE)
        tb = slide5.shapes.add_textbox(Inches(left_p + 0.15), Inches(top_p + 0.15), Inches(3.3), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = stitle
        p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = DARK_NAVY
        p = tf.add_paragraph()
        p.text = sdesc
        p.font.size = Pt(9.5); p.font.color.rgb = SLATE_GRAY
        left_p += 4.0
        if (idx + 1) % 3 == 0:
            left_p = 0.8
            top_p += 1.75

    # -------------------------------------------------------------------------
    # SLIDE 6: DATASET & LEAKAGE PREVENTION
    # -------------------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Dataset Curation & Leakage-Free Splitting", "DATA ENGINEERING")

    add_metric_box(slide6, Inches(0.8), Inches(1.5), Inches(2.7), Inches(1.4), "9,175", "Total Leaf Images", "Standardized 256x256 RGB captures")
    add_metric_box(slide6, Inches(3.8), Inches(1.5), Inches(2.7), Inches(1.4), "11 Classes", "3 Major KP Crops", "Apple (4), Maize (4), Potato (3)")
    add_metric_box(slide6, Inches(6.8), Inches(1.5), Inches(2.7), Inches(1.4), "70 / 15 / 15", "Train / Val / Test", "6,422 / 1,376 / 1,377 images")
    add_metric_box(slide6, Inches(9.8), Inches(1.5), Inches(2.7), Inches(1.4), "0.00%", "Data Leakage", "Group-stratified by leaf ID")

    add_card(slide6, Inches(0.8), Inches(3.2), Inches(11.7), Inches(3.6), bg_color=WHITE)
    tb_d = slide6.shapes.add_textbox(Inches(1.1), Inches(3.4), Inches(11.1), Inches(3.2))
    tf_d = tb_d.text_frame
    tf_d.word_wrap = True
    p = tf_d.paragraphs[0]
    p.text = "Strict Methodology for Preventing Data Leakage:"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_d.add_paragraph()
    p.text = "1. Leaf-Group Clustering: Captures sharing identical physical leaf IDs were grouped to prevent multi-shot splits.\n" \
             "2. Perceptual Difference Hashing (dHash): 64-bit image gradient hashes detected near-duplicates (Hamming dist <= 4), locking them to identical split buckets.\n" \
             "3. Group-Stratified Isolation: Partitioned using GroupKFold. Automated tests confirm: Set(Train_Leaves) ∩ Set(Test_Leaves) = ∅.\n" \
             "4. Class-Weighted Loss: Balanced inverse-frequency weights applied during training to counter class imbalance (Potato Healthy N=152 vs Apple Healthy N=1645)."
    p.font.size = Pt(10.5); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 7: VISION BACKBONE SELECTION
    # -------------------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Vision Backbone Selection: MobileNetV3 vs. EfficientNet-B0", "MODEL SELECTION")

    # Table of comparison
    t_shape = slide7.shapes.add_table(7, 3, Inches(0.8), Inches(1.6), Inches(7.0), Inches(4.8))
    table = t_shape.table
    table.columns[0].width = Inches(2.8)
    table.columns[1].width = Inches(2.2)
    table.columns[2].width = Inches(2.0)

    rows_data = [
        ("Evaluation Metric", "MobileNetV3-Large", "EfficientNet-B0"),
        ("Validation Macro-F1 (Selection)", "0.9773 (Selected)", "0.9535"),
        ("Held-out Test Top-1 Accuracy", "97.53% (1,343 / 1,377)", "95.06% (1,309 / 1,377)"),
        ("Held-out Test Macro-F1", "0.9657", "0.9352"),
        ("Mean CPU Latency per Image", "28.38 ms (35.2 FPS)", "48.01 ms (20.8 FPS)"),
        ("Expected Calibration Error (ECE)", "0.0140 (1.40%)", "0.0589 (5.89%)"),
        ("Model Checkpoint Size", "16.28 MB (4.2M params)", "15.63 MB (4.0M params)"),
    ]
    for r_idx, r_vals in enumerate(rows_data):
        for c_idx, val in enumerate(r_vals):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9.5)
            if r_idx == 0:
                p.font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = BORDER_GRAY

    # Right text card explaining selection
    add_card(slide7, Inches(8.1), Inches(1.6), Inches(4.4), Inches(4.8), bg_color=WHITE)
    tb_sel = slide7.shapes.add_textbox(Inches(8.3), Inches(1.8), Inches(4.0), Inches(4.4))
    tf_sel = tb_sel.text_frame
    tf_sel.word_wrap = True
    p = tf_sel.paragraphs[0]
    p.text = "Predefined Selection Decision:"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_sel.add_paragraph()
    p.text = "• Model selection was formally locked based on Validation Macro-F1 prior to test set evaluation.\n\n" \
             "• MobileNetV3-Large outperformed EfficientNet-B0 by +2.49% on Validation Macro-F1 and +3.26% on Test Macro-F1.\n\n" \
             "• MobileNetV3 is 40.9% faster on CPU (28.38 ms vs 48.01 ms), crucial for edge responsiveness.\n\n" \
             "• Probability outputs are significantly better calibrated (1.40% ECE vs 5.89%)."
    p.font.size = Pt(10); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 8: CLASSIFICATION PERFORMANCE
    # -------------------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Final Vision Classification Performance (Test N = 1,377)", "CLASSIFICATION RESULTS")

    add_metric_box(slide8, Inches(0.8), Inches(1.5), Inches(2.7), Inches(1.4), "97.53%", "Top-1 Accuracy", "1,343 / 1,377 correct")
    add_metric_box(slide8, Inches(3.8), Inches(1.5), Inches(2.7), Inches(1.4), "99.78%", "Top-2 Accuracy", "1,374 / 1,377 in Top-2")
    add_metric_box(slide8, Inches(6.8), Inches(1.5), Inches(2.7), Inches(1.4), "0.9657", "Macro F1-Score", "Balanced across all 11 classes")
    add_metric_box(slide8, Inches(9.8), Inches(1.5), Inches(2.7), Inches(1.4), "1.40%", "Calibration (ECE)", "10-bin expected calibration error")

    # Embed Confusion Matrix
    fig_cm = config.PROJECT_ROOT / "evaluation/figures/confusion_matrix_mobilenet_v3_large.png"
    if fig_cm.exists():
        slide8.shapes.add_picture(str(fig_cm), Inches(0.8), Inches(3.1), Inches(6.5), Inches(4.0))

    # Right Card: Observations
    add_card(slide8, Inches(7.6), Inches(3.1), Inches(4.9), Inches(4.0), bg_color=WHITE)
    tb_cobs = slide8.shapes.add_textbox(Inches(7.8), Inches(3.3), Inches(4.5), Inches(3.6))
    tf_cobs = tb_cobs.text_frame
    tf_cobs.word_wrap = True
    p = tf_cobs.paragraphs[0]
    p.text = "Key Test Set Findings:"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_cobs.add_paragraph()
    p.text = "• Zero Cross-Crop Confusions: The model never confused Maize for Apple or Potato.\n\n" \
             "• Perfect Classes: Common Rust (179/179) and Healthy Maize (175/175) achieved 100% precision and recall.\n\n" \
             "• Main Confusion: Maize Gray Leaf Spot (F1 0.8435) vs. Northern Leaf Blight (F1 0.9236) accounted for 67.6% of all test errors (23/34 cases) due to elongated foliar streak morphology."
    p.font.size = Pt(10); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 9: EXPLAINABLE AI (GRAD-CAM)
    # -------------------------------------------------------------------------
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Visual Interpretability via Grad-CAM (features.16)", "EXPLAINABLE AI")

    fig_grid = config.PROJECT_ROOT / "evaluation/figures/gradcam_11_classes_grid.png"
    if fig_grid.exists():
        slide9.shapes.add_picture(str(fig_grid), Inches(0.8), Inches(1.5), Inches(7.5), Inches(5.4))

    add_card(slide9, Inches(8.6), Inches(1.5), Inches(3.9), Inches(5.4), bg_color=WHITE)
    tb_xai = slide9.shapes.add_textbox(Inches(8.8), Inches(1.7), Inches(3.5), Inches(5.0))
    tf_xai = tb_xai.text_frame
    tf_xai.word_wrap = True
    p = tf_xai.paragraphs[0]
    p.text = "Grad-CAM Hook Mechanism:"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_xai.add_paragraph()
    p.text = "• Target Layer: features.16 (final 1x1 conv block; 7x7 spatial resolution with 960 channels).\n\n" \
             "• Gradient Weighting: Computes global average pooling of gradients of target logit score w.r.t. feature maps.\n\n" \
             "• Spatial Saliency: ReLU isolates positive evidence; overlay is bilinearly upsampled to 224x224.\n\n" \
             "• Dynamic Inspection: Users adjust overlay opacity in real time via Streamlit slider without re-running backpropagation."
    p.font.size = Pt(10); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 10: GRAD-CAM SANITY CHECKS
    # -------------------------------------------------------------------------
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Adebayo et al. Parameter Randomization Sanity Checks", "XAI VALIDATION")

    fig_sc = config.PROJECT_ROOT / "evaluation/figures/gradcam_sanity_check_randomization.png"
    if fig_sc.exists():
        slide10.shapes.add_picture(str(fig_sc), Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.4))

    add_card(slide10, Inches(7.9), Inches(1.5), Inches(4.6), Inches(5.4), bg_color=WHITE)
    tb_sc = slide10.shapes.add_textbox(Inches(8.1), Inches(1.7), Inches(4.2), Inches(5.0))
    tf_sc = tb_sc.text_frame
    tf_sc.word_wrap = True
    p = tf_sc.paragraphs[0]
    p.text = "Cascading Randomization Results:"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_sc.add_paragraph()
    p.text = "• Why It Matters: Saliency maps often act as edge detectors invariant to trained weights (Adebayo et al., NeurIPS 2018).\n\n" \
             "• Empirical Test: Replaced features.16 weights with random Gaussian noise N(0, 0.02).\n\n" \
             "• Quantitative Correlation:\n" \
             "  - Pearson r: 0.0000 across all classes\n" \
             "  - Spearman rho: -0.0410 to 0.1859\n" \
             "  - Mean Absolute Error: 0.2381 to 0.4097\n\n" \
             "• Verdict: Heatmaps genuinely depend on learned representations, passing the scientific sanity check."
    p.font.size = Pt(10); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 11: KNOWLEDGE BASE PROVENANCE
    # -------------------------------------------------------------------------
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Curated KP Knowledge Base & Source Provenance", "KNOWLEDGE BASE")

    add_card(slide11, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3), bg_color=WHITE)
    tb_kb = slide11.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.9))
    tf_kb = tb_kb.text_frame
    tf_kb.word_wrap = True
    p = tf_kb.paragraphs[0]
    p.text = "11 Official Agricultural Extension Bulletins (164 Contextual Chunks):"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_kb.add_paragraph()
    p.text = "• Apple Bulletins (Swat, Dir, Hazara): [KP-AGRI-APP-001] Apple Scab Management (ARI Tarnab); [KP-AGRI-APP-002] Black Rot & Frogeye Leaf Spot; [KP-AGRI-APP-003] Cedar Apple Rust; [KP-AGRI-APP-004] Integrated Orchard Canopy Management.\n\n" \
             "• Maize Bulletins (Swabi, Mardan, Nowshera): [KP-AGRI-MAI-001] Gray Leaf Spot Guide (CCRI Pirsabak & PARC); [KP-AGRI-MAI-002] Common Rust Advisory; [KP-AGRI-MAI-003] Northern Corn Leaf Blight; [KP-AGRI-MAI-004] Kharif Cultural Practices.\n\n" \
             "• Potato Bulletins (Kalam, Kaghan, Abbottabad): [KP-AGRI-POT-001] Early Blight Management (Potato Research Centre Abbottabad); [KP-AGRI-POT-002] Late Blight Epidemic Early-Warning (PRC Abbottabad & ARI Tarnab); [KP-AGRI-POT-003] Certified Seed Tuber Care.\n\n" \
             "• Zero Hallucination Policy: All chemical actives, dosage rates, and cultural controls are verified directly against government-approved extension advisories."
    p.font.size = Pt(10.5); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 12: RAG PIPELINE & VECTOR STORE
    # -------------------------------------------------------------------------
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "FAISS Dense Retrieval & Vision-Guided Query Routing", "RAG PIPELINE")

    add_metric_box(slide12, Inches(0.8), Inches(1.5), Inches(3.6), Inches(1.5), "all-MiniLM-L6-v2", "Dense Bi-Encoder", "384-dimensional normalized vectors")
    add_metric_box(slide12, Inches(4.8), Inches(1.5), Inches(3.6), Inches(1.5), "FAISS IndexFlatIP", "Exact Cosine Search", "164 indexed passages (<16 ms search)")
    add_metric_box(slide12, Inches(8.8), Inches(1.5), Inches(3.7), Inches(1.5), "tau = 0.45", "Similarity Threshold", "Calibrated out-of-domain abstention")

    add_card(slide12, Inches(0.8), Inches(3.3), Inches(11.7), Inches(3.6), bg_color=WHITE)
    tb_rag_arch = slide12.shapes.add_textbox(Inches(1.1), Inches(3.5), Inches(11.1), Inches(3.2))
    tf_rag_arch = tb_rag_arch.text_frame
    tf_rag_arch.word_wrap = True
    p = tf_rag_arch.paragraphs[0]
    p.text = "Vision-Guided Context Routing:"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_rag_arch.add_paragraph()
    p.text = "1. Context Injection: Vision classification output (e.g. 'Apple Apple_scab') automatically prefixes user queries.\n" \
             "2. Metadata Filtering: Retrieval filters on target crop and disease, eliminating cross-crop document leakage.\n" \
             "3. Top-k Retrieval: Returns Top-3 passages containing exact symptoms, cause, cultural practices, and chemical recommendations.\n" \
             "4. Offline Capability: Sentence-Transformers and FAISS index execute entirely locally on CPU without network calls."
    p.font.size = Pt(10.5); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 13: RETRIEVAL BENCHMARK RESULTS
    # -------------------------------------------------------------------------
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "Empirical Retrieval Evaluation (33 Domain Queries)", "RETRIEVAL RESULTS")

    add_metric_box(slide13, Inches(0.8), Inches(1.5), Inches(2.7), Inches(1.4), "96.97%", "Hit@1 Rate", "32 / 33 target docs at Rank 1")
    add_metric_box(slide13, Inches(3.8), Inches(1.5), Inches(2.7), Inches(1.4), "100.0%", "Hit@3 Rate", "33 / 33 target docs in Top-3")
    add_metric_box(slide13, Inches(6.8), Inches(1.5), Inches(2.7), Inches(1.4), "0.9848", "Mean Reciprocal Rank", "High top-rank concentration")
    add_metric_box(slide13, Inches(9.8), Inches(1.5), Inches(2.7), Inches(1.4), "21.84 ms", "Mean Retrieval Latency", "Embedding + FAISS search on CPU")

    # Table of query categories
    t_shape_q = slide13.shapes.add_table(6, 3, Inches(0.8), Inches(3.2), Inches(11.7), Inches(3.6))
    table_q = t_shape_q.table
    table_q.columns[0].width = Inches(3.2)
    table_q.columns[1].width = Inches(4.5)
    table_q.columns[2].width = Inches(4.0)

    q_data = [
        ("Query Category Evaluated", "Sample Agronomic Question", "Retrieval Outcome"),
        ("1. Symptoms & Morphology", "What are the leaf and fruit symptoms of apple scab in Swat?", "Hit@1 (KP-AGRI-APP-001, score: 0.727)"),
        ("2. Causes & Pathogen Biology", "How does Phytophthora infestans survive winter in potato soil?", "Hit@1 (KP-AGRI-POT-002, score: 0.684)"),
        ("3. Cultural Management", "What crop rotation and stubble disposal controls maize leaf spot?", "Hit@1 (KP-AGRI-MAI-001, score: 0.712)"),
        ("4. Chemical & Fungicide Options", "Recommended spray schedule for late blight in high-humidity periods?", "Hit@1 (KP-AGRI-POT-002, score: 0.749)"),
        ("5. Out-of-Scope Guardrails", "How to control cotton leaf curl virus in D.I. Khan?", "100% Rejected (status: information_unavailable)"),
    ]
    for r_idx, r_vals in enumerate(q_data):
        for c_idx, val in enumerate(r_vals):
            cell = table_q.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9.5)
            if r_idx == 0:
                p.font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = BORDER_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 14: GROUNDED MULTILINGUAL ADVISORY
    # -------------------------------------------------------------------------
    slide14 = prs.slides.add_slide(blank_layout)
    add_header(slide14, "Grounded Multilingual Advisory Generation", "ADVISORY SYNTHESIS")

    add_metric_box(slide14, Inches(0.8), Inches(1.5), Inches(2.7), Inches(1.4), "100.0%", "Groundedness Rate", "15/15 cases strictly grounded")
    add_metric_box(slide14, Inches(3.8), Inches(1.5), Inches(2.7), Inches(1.4), "100.0%", "Citation Fidelity", "Every advice links institutional ID")
    add_metric_box(slide14, Inches(6.8), Inches(1.5), Inches(2.7), Inches(1.4), "100.0%", "Out-of-Scope Abstention", "4/4 unsupported crops rejected")
    add_metric_box(slide14, Inches(9.8), Inches(1.5), Inches(2.7), Inches(1.4), "15.63 ms", "Urdu Synthesis Latency", "Dictionary-cached localization")

    add_card(slide14, Inches(0.8), Inches(3.2), Inches(11.7), Inches(3.6), bg_color=WHITE)
    tb_adv = slide14.shapes.add_textbox(Inches(1.1), Inches(3.4), Inches(11.1), Inches(3.2))
    tf_adv = tb_adv.text_frame
    tf_adv.word_wrap = True
    p = tf_adv.paragraphs[0]
    p.text = "Extractive Synthesis & Multilingual Architecture:"
    p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = DARK_NAVY
    p = tf_adv.add_paragraph()
    p.text = "• Extractive Synthesis: Assembles 5 structured sections (Symptoms, Pathogen, IPM, Prevention, Stage Guidance) strictly from retrieved chunks without unconstrained generation.\n\n" \
             "• Entity Protection: Regex shielding preserves Latin binomials (Venturia inaequalis, Phytophthora infestans) and institutional citation keys ([KP-AGRI-*]) from translation corruption.\n\n" \
             "• Agricultural Urdu Glossary: 33 curated agronomic terms map English concepts to native farmer terminology (e.g. Late Blight → پچھیتی جھلساؤ; Scab → پپڑی روگ).\n\n" \
             "• Right-to-Left (RTL) Layout: Injects custom CSS with Google Noto Nastaliq Urdu typography and right-aligned reading direction."
    p.font.size = Pt(10.5); p.font.color.rgb = SLATE_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 15: STREAMLIT APPLICATION
    # -------------------------------------------------------------------------
    slide15 = prs.slides.add_slide(blank_layout)
    add_header(slide15, "Streamlit Interactive Web Application", "USER INTERFACE")

    features = [
        ("Dual Input Mode", "Drag-and-drop leaf image uploader or 1-click selection from an 11-class preloaded demo gallery."),
        ("Bilingual Header Switcher", "Seamless toggle between English and Urdu with automatic RTL CSS layout switching."),
        ("Calibrated Diagnostic Card", "Displays Top-1 diagnosis with confidence score alongside Top-2 comparative bars and calibration warnings."),
        ("Interactive Grad-CAM Viewer", "3-column comparative view (Original, Raw Heatmap, Blended Overlay) with dynamic alpha slider."),
        ("Tabbed Structured Advisory", "Symptoms, Likely Cause, Integrated Management, Cultural Prevention, and Institutional Provenance Badges."),
        ("Grounded Follow-up Q&A", "Context-aware conversational assistant constrained strictly by knowledge-base guardrails."),
    ]
    left_f = 0.8
    top_f = 1.6
    for idx, (ftitle, fdesc) in enumerate(features):
        add_card(slide15, Inches(left_f), Inches(top_f), Inches(5.6), Inches(1.5), bg_color=WHITE, border_color=FOREST_GREEN)
        tb = slide15.shapes.add_textbox(Inches(left_f + 0.15), Inches(top_f + 0.15), Inches(5.3), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = ftitle
        p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = DARK_NAVY
        p = tf.add_paragraph()
        p.text = fdesc
        p.font.size = Pt(9.5); p.font.color.rgb = SLATE_GRAY
        left_f += 6.0
        if (idx + 1) % 2 == 0:
            left_f = 0.8
            top_f += 1.75

    # -------------------------------------------------------------------------
    # SLIDE 16: END-TO-END WORKFLOW BENCHMARK
    # -------------------------------------------------------------------------
    slide16 = prs.slides.add_slide(blank_layout)
    add_header(slide16, "End-to-End Multimodal System Evaluation (N = 17)", "END-TO-END BENCHMARK")

    add_metric_box(slide16, Inches(0.8), Inches(1.5), Inches(2.7), Inches(1.4), "100.0%", "Pipeline Execution Rate", "17 / 17 cases without exceptions")
    add_metric_box(slide16, Inches(3.8), Inches(1.5), Inches(2.7), Inches(1.4), "184.22 ms", "Mean End-to-End Latency", "P95: 271.51 ms on standard CPU")
    add_metric_box(slide16, Inches(6.8), Inches(1.5), Inches(2.7), Inches(1.4), "92.31%", "Image Diagnostic Acc", "12/13 correct (incl. difficult)")
    add_metric_box(slide16, Inches(9.8), Inches(1.5), Inches(2.7), Inches(1.4), "100.0%", "Citation Linkage", "All image workflows cited KP docs")

    # Table of latency breakdown
    t_shape_lat = slide16.shapes.add_table(6, 4, Inches(0.8), Inches(3.2), Inches(11.7), Inches(3.6))
    table_lat = t_shape_lat.table
    table_lat.columns[0].width = Inches(4.5)
    table_lat.columns[1].width = Inches(2.4)
    table_lat.columns[2].width = Inches(2.4)
    table_lat.columns[3].width = Inches(2.4)

    lat_rows = [
        ("Pipeline Processing Stage", "Mean Latency (ms)", "P95 Latency (ms)", "% of Total Runtime"),
        ("1. Vision Forward Inference (MobileNetV3)", "71.56 ms", "116.52 ms", "38.8%"),
        ("2. Grad-CAM Activation & Overlay (features.16)", "58.55 ms", "95.33 ms", "31.8%"),
        ("3. FAISS Vector Retrieval (Top-3 Chunks)", "15.94 ms", "18.88 ms", "8.7%"),
        ("4. Grounded Advisory Synthesis (English)", "18.62 ms", "22.09 ms", "10.1%"),
        ("5. Urdu Localization & RTL Formatting", "19.54 ms", "23.73 ms", "10.6%"),
    ]
    for r_idx, r_vals in enumerate(lat_rows):
        for c_idx, val in enumerate(r_vals):
            cell = table_lat.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9.5)
            if r_idx == 0:
                p.font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = BORDER_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 17: CONTROLLED ABLATION EXPERIMENTS
    # -------------------------------------------------------------------------
    slide17 = prs.slides.add_slide(blank_layout)
    add_header(slide17, "Controlled Ablation Experiments & Key Findings", "ABLATION STUDIES")

    abls = [
        ("Ablation 1: Classifier Only vs. Classifier + Grad-CAM",
         "• Classifier Only: 21.84 ms mean CPU latency (No spatial explanation).\n"
         "• Classifier + Grad-CAM: 80.90 ms mean CPU latency (Full 7x7 heatmap overlay).\n"
         "• Finding: Backpropagation adds +59 ms overhead, but provides indispensable visual transparency well below interactive limits (<100 ms)."),

        ("Ablation 2: Unfiltered vs. Metadata-Filtered Retrieval",
         "• Unfiltered Semantic Search: 13.33% cross-crop document contamination (Potato Early Blight retrieved for generic Maize query).\n"
         "• Vision-Conditioned Retrieval: 0.00% cross-crop contamination.\n"
         "• Finding: Coupling predicted vision metadata with query text completely eliminates cross-crop advisory leakage."),

        ("Ablation 3: English Query vs. Urdu Query Localization",
         "• English Synthesis Latency: 20.77 ms mean | Urdu Synthesis Latency (Cached): 19.54 ms mean.\n"
         "• Text Length Expansion: 1.03x (Urdu expands by 3% in character count).\n"
         "• Finding: Dictionary caching achieves latency parity with English while strictly preserving Latin pathogen nomenclature."),

        ("Ablation 4: In-Scope vs. Out-of-Scope Query Guardrails",
         "• In-Scope Mean Cosine Similarity: 0.5458 on supported agricultural topics.\n"
         "• Out-of-Scope Abstention Rate: 100.00% rejection on Cotton, Wheat, Sugarcane, Citrus.\n"
         "• Finding: Dual lexical/threshold guardrails prevent hallucinated chemical advice on unsupported regional crops."),
    ]
    left_a = 0.8
    top_a = 1.6
    for atitle, adesc in abls:
        add_card(slide17, Inches(left_a), Inches(top_a), Inches(5.6), Inches(2.55), bg_color=WHITE)
        tb = slide17.shapes.add_textbox(Inches(left_a + 0.15), Inches(top_a + 0.15), Inches(5.3), Inches(2.25))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = atitle
        p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = DARK_NAVY
        p = tf.add_paragraph()
        p.text = adesc
        p.font.size = Pt(9); p.font.color.rgb = SLATE_GRAY
        left_a += 6.0
        if left_a > 7.0:
            left_a = 0.8
            top_a += 2.75

    # -------------------------------------------------------------------------
    # SLIDE 18: ERROR ANALYSIS & FAILURE MODES
    # -------------------------------------------------------------------------
    slide18 = prs.slides.add_slide(blank_layout)
    add_header(slide18, "Detailed Error Analysis (34 Errors / 1,377 Test Images)", "ERROR ANALYSIS")

    add_metric_box(slide18, Inches(0.8), Inches(1.5), Inches(2.7), Inches(1.4), "34 / 1,377", "Total Test Misclassifications", "2.47% error rate (97.53% acc)")
    add_metric_box(slide18, Inches(3.8), Inches(1.5), Inches(2.7), Inches(1.4), "0 Errors", "Cross-Crop Confusions", "No Maize mistaken for Potato/Apple")
    add_metric_box(slide18, Inches(6.8), Inches(1.5), Inches(2.7), Inches(1.4), "67.6%", "Maize Foliar Ambiguity", "23/34 errors: Gray Spot vs Blight")
    add_metric_box(slide18, Inches(9.8), Inches(1.5), Inches(2.7), Inches(1.4), "11.8%", "Glare & Subtle Scab", "4/34 errors: Specular reflection")

    # Table of error cases
    t_shape_err = slide18.shapes.add_table(5, 4, Inches(0.8), Inches(3.2), Inches(11.7), Inches(3.6))
    table_err = t_shape_err.table
    table_err.columns[0].width = Inches(2.5)
    table_err.columns[1].width = Inches(2.5)
    table_err.columns[2].width = Inches(1.2)
    table_err.columns[3].width = Inches(5.5)

    err_data = [
        ("True Ground Truth", "Predicted Class", "Errors", "Morphological & Visual Root Cause"),
        ("Corn: Gray Leaf Spot", "Corn: Northern Leaf Blight", "15", "Elongated grayish lesions run parallel to veins, mimicking early blight streaks"),
        ("Corn: Northern Leaf Blight", "Corn: Gray Leaf Spot", "8", "Blunt-ended immature cigar blights mimic rectangular Gray Leaf Spot lesions"),
        ("Apple: Healthy Leaf", "Apple: Apple Scab", "3", "Specular lighting glare and foliar dust specks mistaken for subtle scab spots"),
        ("Potato: Late Blight", "Potato: Healthy Leaf", "2", "Incipient water-soaked lesions (<2% of leaf blade) missed by global pooling"),
    ]
    for r_idx, r_vals in enumerate(err_data):
        for c_idx, val in enumerate(r_vals):
            cell = table_err.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9.5)
            if r_idx == 0:
                p.font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = BORDER_GRAY

    # -------------------------------------------------------------------------
    # SLIDE 19: LIMITATIONS & RESPONSIBLE USE
    # -------------------------------------------------------------------------
    slide19 = prs.slides.add_slide(blank_layout)
    add_header(slide19, "Academic Limitations & Responsible Deployment", "LIMITATIONS")

    lims = [
        ("Controlled vs. Field Imagery", "Trained on standardized laboratory foliar captures. Real field photos with complex soil/weed backgrounds, motion blur, direct midday glare, or co-occurring pest damage may show reduced accuracy. Field readiness cannot be claimed without prospective field trials."),
        ("Coarse Spatial Resolution of Grad-CAM", "Grad-CAM operates on a 7x7 feature grid (features.16). While effective for macroscopic regional localization, it cannot resolve individual fungal spores or microscopic bacterial pustules. Saliency does not prove biological causality."),
        ("Closed-Set Softmax Constraint", "The vision model supports strictly 3 crops and 11 classes. Inputting an unsupported crop (e.g. Cotton) forces an output into one of the 11 classes. The UI mitigates this via Top-2 comparative bars, confidence caveats, and text guardrails."),
        ("Non-Replacement of Human Agronomists", "Chemical pesticides carry severe toxicity and environmental risks. The software is strictly an academic decision-support prototype and must never supersede on-site inspection by licensed KP agricultural extension officers."),
    ]
    top_l = 1.6
    for title, desc in lims:
        add_card(slide19, Inches(0.8), Inches(top_l), Inches(11.7), Inches(1.15), bg_color=WHITE)
        tb = slide19.shapes.add_textbox(Inches(1.0), Inches(top_l + 0.1), Inches(11.3), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title + " — "
        p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = ACCENT_AMBER
        run = p.add_run()
        run.text = desc
        run.font.size = Pt(10); run.font.bold = False; run.font.color.rgb = SLATE_GRAY
        top_l += 1.3

    # -------------------------------------------------------------------------
    # SLIDE 20: CORE CONTRIBUTIONS
    # -------------------------------------------------------------------------
    slide20 = prs.slides.add_slide(blank_layout)
    add_header(slide20, "Summary of Primary Capstone Contributions", "CONTRIBUTIONS")

    conts = [
        ("1. Leakage-Free Edge Vision", "Trained MobileNetV3-Large on 11 KP crop classes with verified leaf-group isolation (97.53% Top-1, 0.9657 Macro-F1, 1.40% ECE, 28.38 ms CPU latency)."),
        ("2. Scientifically Validated XAI", "Implemented Grad-CAM on features.16, verified via cascading parameter-randomization sanity checks (r=0.0000) to confirm learned feature dependency."),
        ("3. Localized Agricultural RAG", "Indexed 11 official KP extension bulletins in FAISS (96.97% Hit@1, 100.0% Hit@3, 0.9848 MRR), eliminating generative chemical hallucinations."),
        ("4. Bilingual RTL Localization", "Engineered an entity-preserving translation engine and 33-term agricultural glossary for fluent Urdu advisory synthesis with Latin binomial protection."),
        ("5. Guardrailed Decision Support", "Calibrated out-of-scope abstention filters (100% rejection on unsupported crops) and deployed a responsive, offline-capable Streamlit dashboard."),
    ]
    top_c = 1.6
    for title, desc in conts:
        add_card(slide20, Inches(0.8), Inches(top_c), Inches(11.7), Inches(0.95), bg_color=WHITE)
        tb = slide20.shapes.add_textbox(Inches(1.0), Inches(top_c + 0.08), Inches(11.3), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title + " — "
        p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = FOREST_GREEN
        run = p.add_run()
        run.text = desc
        run.font.size = Pt(10.5); run.font.bold = False; run.font.color.rgb = SLATE_GRAY
        top_c += 1.1

    # -------------------------------------------------------------------------
    # SLIDE 21: FUTURE WORK
    # -------------------------------------------------------------------------
    slide21 = prs.slides.add_slide(blank_layout)
    add_header(slide21, "Roadmap for Future Research & Field Scaling", "FUTURE WORK")

    fw = [
        ("1. Prospective Field Data Collection", "Partner with the KP Department of Agriculture Extension to collect and annotate 10,000 in-situ smartphone photographs across farming clusters in Swat, Mardan, and Abbottabad to train a domain-adapted vision model."),
        ("2. Multi-Crop Expansion", "Curate verified training imagery and extension literature for additional major KP crops: Wheat (Stripe Rust), Sugarcane (Red Rot), and Citrus (Bacterial Canker)."),
        ("3. Pashto Voice Interface", "Expand the translation engine to support regional Pashto dialects with audio text-to-speech, enabling accessibility for non-literate smallholder farmers."),
        ("4. Mobile On-Device Deployment", "Quantize the MobileNetV3-Large vision model and FAISS vector index to INT8 via ONNX Runtime / TFLite for zero-connectivity offline operation on Android smartphones."),
    ]
    top_fw = 1.6
    for title, desc in fw:
        add_card(slide21, Inches(0.8), Inches(top_fw), Inches(11.7), Inches(1.15), bg_color=WHITE)
        tb = slide21.shapes.add_textbox(Inches(1.0), Inches(top_fw + 0.1), Inches(11.3), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title + " — "
        p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = ACCENT_BLUE
        run = p.add_run()
        run.text = desc
        run.font.size = Pt(10.5); run.font.bold = False; run.font.color.rgb = SLATE_GRAY
        top_fw += 1.3

    # -------------------------------------------------------------------------
    # SLIDE 22: CONCLUSION & Q&A (Dark Navy Theme)
    # -------------------------------------------------------------------------
    slide22 = prs.slides.add_slide(blank_layout)
    bg22 = slide22.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg22.fill.solid()
    bg22.fill.fore_color.rgb = DARK_NAVY
    bg22.line.fill.background()

    tb22 = slide22.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.0))
    tf22 = tb22.text_frame
    tf22.word_wrap = True

    p0 = tf22.paragraphs[0]
    p0.text = "KP SMART CROP DOCTOR"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = FOREST_GREEN
    p0.space_after = Pt(12)

    p1 = tf22.add_paragraph()
    p1.text = "Thank You for Your Time & Attention"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_after = Pt(14)

    p2 = tf22.add_paragraph()
    p2.text = "An Explainable, Knowledge-Grounded, Multilingual Decision-Support Prototype\nDesigned for the Farming Communities of Khyber Pakhtunkhwa"
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(0xDF, 0xE2, 0xE6)
    p2.space_after = Pt(28)

    p3 = tf22.add_paragraph()
    p3.text = "We welcome questions, technical critiques, and viva defense discussion."
    p3.font.size = Pt(13)
    p3.font.bold = True
    p3.font.color.rgb = FOREST_GREEN

    # Save presentation
    out_pptx = config.PROJECT_ROOT / "KP_Smart_Crop_Doctor_Final_Presentation.pptx"
    prs.save(str(out_pptx))
    print(f"Successfully generated Presentation PPTX at: {out_pptx}")
    return out_pptx


if __name__ == "__main__":
    build_presentation()
