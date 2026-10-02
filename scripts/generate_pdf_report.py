"""Generate the Final Technical Report for KP Smart Crop Doctor in PDF format.

Uses ReportLab to compile a publication-grade academic report with embedded figures,
tables, headers, footers, and page numbers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas

from src import config


class NumberedCanvas(canvas.Canvas):
    """Custom canvas that computes total pages dynamically for running footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#666666"))

            # Running header
            self.drawString(54, 750, "KP Smart Crop Doctor — Final Academic Technical Report")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

            # Running footer
            self.line(54, 45, 558, 45)
            self.drawString(54, 32, "Academic Decision-Support Prototype | Department of Computer Science & AI")
            self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
            self.restoreState()


def build_pdf_report():
    pdf_path = config.PROJECT_ROOT / "KP_Smart_Crop_Doctor_Final_Report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0A2540"),
        alignment=1, # Center
        spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#444444"),
        alignment=1,
        spaceAfter=20,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1B4D3E"), # Forest green
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0A2540"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#222222"),
        leftIndent=15,
        spaceAfter=3,
    )
    caption_style = ParagraphStyle(
        "FigCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceAfter=10,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#222222"),
    )
    table_hdr = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0A2540"),
    )

    story = []

    # -------------------------------------------------------------------------
    # COVER PAGE
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 40))
    p_inst = Paragraph("DEPARTMENT OF COMPUTER SCIENCE & ARTIFICIAL INTELLIGENCE<br/>FINAL YEAR CAPSTONE PROJECT REPORT", ParagraphStyle(
        "InstHdr", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#1B4D3E"), alignment=1
    ))
    story.append(p_inst)
    story.append(Spacer(1, 24))
    story.append(Paragraph("KP Smart Crop Doctor", title_style))
    story.append(Paragraph(
        "An Explainable, Knowledge-Grounded, Multilingual Plant Disease Decision-Support System<br/>"
        "for Khyber Pakhtunkhwa Agriculture", subtitle_style
    ))
    story.append(HRFlowable(width="60%", thickness=1.5, color=colors.HexColor("#1B4D3E"), spaceBefore=10, spaceAfter=25))

    p_desc = Paragraph(
        "<b>A Multi-Stage Decision Support Prototype Integrating:</b><br/>"
        "• MobileNetV3-Large Transfer Learning Classifier (97.53% Top-1 Accuracy, 1.40% ECE)<br/>"
        "• Spatial Visual Explainability via Grad-CAM (features.16) with Cascading Sanity Checks<br/>"
        "• Traceable Retrieval-Augmented Generation (FAISS dense vector index of 11 KP extension bulletins)<br/>"
        "• Grounded Bilingual Advisory Synthesis (English & Urdu RTL with Entity Protection)<br/>"
        "• Edge-Compatible Interactive Streamlit Dashboard (184.22 ms mean CPU latency)",
        ParagraphStyle("CoverDesc", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=15, alignment=1, textColor=colors.HexColor("#333333"))
    )
    story.append(p_desc)
    story.append(Spacer(1, 60))

    meta_table_data = [
        [Paragraph("<b>Authors:</b>", table_hdr), Paragraph("Final Year Capstone Team (AI/ML Specialization)", table_cell)],
        [Paragraph("<b>Supervision:</b>", table_hdr), Paragraph("Departmental Faculty Advisory Committee", table_cell)],
        [Paragraph("<b>Academic Session:</b>", table_hdr), Paragraph("2025 – 2026", table_cell)],
        [Paragraph("<b>Target Region:</b>", table_hdr), Paragraph("Khyber Pakhtunkhwa (KP), Pakistan", table_cell)],
        [Paragraph("<b>Repository:</b>", table_hdr), Paragraph("Master Branch (All 8 Milestones Completed & Fully Audited)", table_cell)],
    ]
    meta_table = Table(meta_table_data, colWidths=[110, 320])
    meta_table.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#EEEEEE")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)

    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # EXECUTIVE ABSTRACT & DECLARATION
    # -------------------------------------------------------------------------
    story.append(Paragraph("Declaration of Academic Integrity", h1_style))
    story.append(Paragraph(
        "We declare that this capstone technical report describes our original work in developing 'KP Smart Crop Doctor'. "
        "Every empirical metric, training split manifest, Grad-CAM explanation, vector retrieval score, and bilingual translation "
        "documented herein is derived directly from executable repository artifacts. This system is strictly an academic "
        "decision-support prototype designed to augment agricultural extension workers and does not replace in-person consultation "
        "with certified KP agricultural authorities.", body_style
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Executive Abstract", h1_style))
    story.append(Paragraph(
        "Foliar plant diseases inflict severe yield losses on smallholder farmers in Khyber Pakhtunkhwa (KP), Pakistan, "
        "where agricultural extension officer shortages delay timely diagnosis. Conventional deep learning diagnostics "
        "suffer from black-box opacity and lack regional knowledge grounding, while generic Large Language Models (LLMs) "
        "frequently hallucinate hazardous chemical recommendations. This project presents <b>KP Smart Crop Doctor</b>, "
        "a cohesive decision-support architecture for 11 critical disease and healthy classes across Apple, Maize, and Potato.", body_style
    ))
    story.append(Paragraph(
        "Trained on an 11-class leaf-group-isolated dataset (N = 9,175 images) with mathematically verified zero data leakage, "
        "the selected <b>MobileNetV3-Large</b> model achieves a Top-1 Accuracy of <b>97.53%</b>, Top-2 Accuracy of <b>99.78%</b>, "
        "a Macro-F1 score of <b>0.9657</b>, an Expected Calibration Error (ECE) of <b>1.40%</b>, and a CPU latency of <b>28.38 ms</b>, "
        "outperforming an EfficientNet-B0 baseline. Spatial interpretability is provided via <b>Grad-CAM</b> on layer features.16, "
        "validated by cascading parameter-randomization sanity checks (Pearson r drops to 0.0000). Diagnostic predictions query "
        "a curated FAISS vector database of 11 verified regional bulletins from ARI Tarnab and CCRI Pirsabak, achieving a <b>96.97% Hit@1</b>, "
        "<b>100.0% Hit@3</b>, and <b>0.9848 MRR</b>. An extractive synthesizer generates bilingual (English and Urdu) advisories "
        "with 100% groundedness and institutional citations, while domain guardrails intercept out-of-scope crops with 100% abstention. "
        "End-to-end multimodal workflows execute on CPU in an average of <b>184.22 ms</b>.", body_style
    ))

    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # SYSTEM ARCHITECTURE & LEAKAGE PREVENTION
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. System Architecture & Methodology", h1_style))
    story.append(Paragraph(
        "The system operates as a modular, decoupled sequence of micro-components optimized for local CPU inference in rural KP. "
        "A leaf image is preprocessed (256x256 resize, 224x224 center crop, ImageNet normalization) and passed to MobileNetV3-Large. "
        "PyTorch hooks intercept activations and gradients at features.16 to generate a Grad-CAM heatmap. Vision predictions "
        "dispatch a hybrid query to the FAISS vector store. Retrieved passages from official KP bulletins are synthesized into "
        "bilingual advisory cards with Latin pathogen preservation, and delivered via a reactive Streamlit web dashboard.", body_style
    ))
    story.append(Paragraph(
        "<b>Leakage Prevention Protocol:</b> Data leakage was prevented via leaf-group isolation (grouping multi-shot captures "
        "of the same physical leaf) and 64-bit perceptual difference hashing (dHash with Hamming distance <= 4). Group-stratified "
        "splitting (70/15/15) ensured that the Train (6,422), Val (1,376), and Test (1,377) sets share zero identical leaf clusters.", body_style
    ))

    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # VISION RESULTS & BASELINE COMPARISON
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Vision Model Evaluation & Baseline Comparison", h1_style))
    story.append(Paragraph(
        "Model selection was finalized strictly on <b>Validation Macro-F1 prior to evaluating the held-out test set</b>. "
        "MobileNetV3-Large was selected over EfficientNet-B0 due to superior validation F1 (0.9773 vs 0.9535), lower CPU latency "
        "(28.38 ms vs 48.01 ms), and tighter probability calibration (ECE 1.40% vs 5.89%).", body_style
    ))

    # Comparison Table
    t_comp_data = [
        [Paragraph("<b>Metric</b>", table_hdr), Paragraph("<b>MobileNetV3-Large (Selected)</b>", table_hdr), Paragraph("<b>EfficientNet-B0 (Benchmark)</b>", table_hdr)],
        [Paragraph("Pre-Selection Val Macro-F1", table_cell), Paragraph("<b>0.9773</b>", table_cell), Paragraph("0.9535", table_cell)],
        [Paragraph("Validation Accuracy", table_cell), Paragraph("98.18%", table_cell), Paragraph("96.08%", table_cell)],
        [Paragraph("Held-Out Test Top-1 Accuracy", table_cell), Paragraph("<b>97.53% (1,343/1,377)</b>", table_cell), Paragraph("95.06% (1,309/1,377)", table_cell)],
        [Paragraph("Held-Out Test Top-2 Accuracy", table_cell), Paragraph("<b>99.78% (1,374/1,377)</b>", table_cell), Paragraph("99.35% (1,368/1,377)", table_cell)],
        [Paragraph("Held-Out Test Macro-F1", table_cell), Paragraph("<b>0.9657</b>", table_cell), Paragraph("0.9352", table_cell)],
        [Paragraph("Parameters / Model Size", table_cell), Paragraph("4,216,123 (16.28 MB)", table_cell), Paragraph("4,021,639 (15.63 MB)", table_cell)],
        [Paragraph("Mean CPU Latency (ms)", table_cell), Paragraph("<b>28.38 ms (35.2 FPS)</b>", table_cell), Paragraph("48.01 ms (20.8 FPS)", table_cell)],
        [Paragraph("Expected Calibration Error", table_cell), Paragraph("<b>0.0140 (1.40%)</b>", table_cell), Paragraph("0.0589 (5.89%)", table_cell)],
    ]
    t_comp = Table(t_comp_data, colWidths=[150, 170, 160])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_comp)
    story.append(Paragraph("Table 1: Empirical Comparison between MobileNetV3-Large and EfficientNet-B0 (Test N = 1,377)", caption_style))

    # Embed training curves figure if present
    fig_train = config.PROJECT_ROOT / "evaluation/figures/training_curves_comparison.png"
    if fig_train.exists():
        story.append(Spacer(1, 4))
        story.append(Image(str(fig_train), width=5.2*inch, height=2.2*inch))
        story.append(Paragraph("Figure 1: Training & Validation Curves Comparison (Loss and Macro-F1 over 10 Epochs)", caption_style))

    # Embed confusion matrix
    fig_cm = config.PROJECT_ROOT / "evaluation/figures/confusion_matrix_mobilenet_v3_large.png"
    if fig_cm.exists():
        story.append(Spacer(1, 4))
        story.append(Image(str(fig_cm), width=4.8*inch, height=3.6*inch))
        story.append(Paragraph("Figure 2: Confusion Matrix for MobileNetV3-Large on Held-Out Test Set (N = 1,377)", caption_style))

    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # GRAD-CAM & SANITY CHECKS
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Explainable AI (Grad-CAM) & Sanity Checks", h1_style))
    story.append(Paragraph(
        "Grad-CAM attaches to layer features.16 (the final 1x1 convolution producing 7x7x960 activations). Saliency maps are "
        "computed via gradient weighting and ReLU. To verify that explanations depend on learned parameters rather than edge artifacts, "
        "cascading parameter randomization (Adebayo et al., NeurIPS 2018) was performed.", body_style
    ))

    t_sc_data = [
        [Paragraph("<b>Pathology Class</b>", table_hdr), Paragraph("<b>Pearson r</b>", table_hdr), Paragraph("<b>Spearman rho</b>", table_hdr), Paragraph("<b>MAE</b>", table_hdr), Paragraph("<b>Sanity Verdict</b>", table_hdr)],
        [Paragraph("Corn: Gray Leaf Spot", table_cell), Paragraph("<b>0.0000</b>", table_cell), Paragraph("-0.0410", table_cell), Paragraph("0.4097", table_cell), Paragraph("Passed (Feature Dependent)", table_cell)],
        [Paragraph("Potato: Early Blight", table_cell), Paragraph("<b>0.0000</b>", table_cell), Paragraph("-0.0084", table_cell), Paragraph("0.2952", table_cell), Paragraph("Passed (Feature Dependent)", table_cell)],
        [Paragraph("Apple: Apple Scab", table_cell), Paragraph("<b>0.0000</b>", table_cell), Paragraph("0.1208", table_cell), Paragraph("0.2381", table_cell), Paragraph("Passed (Feature Dependent)", table_cell)],
        [Paragraph("Potato: Healthy Leaf", table_cell), Paragraph("<b>0.0000</b>", table_cell), Paragraph("0.1859", table_cell), Paragraph("0.2539", table_cell), Paragraph("Passed (Feature Dependent)", table_cell)],
    ]
    t_sc = Table(t_sc_data, colWidths=[120, 75, 85, 75, 125])
    t_sc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_sc)
    story.append(Paragraph("Table 2: Grad-CAM Cascading Weight Randomization Quantitative Similarity Results", caption_style))

    fig_sc = config.PROJECT_ROOT / "evaluation/figures/gradcam_sanity_check_randomization.png"
    if fig_sc.exists():
        story.append(Spacer(1, 4))
        story.append(Image(str(fig_sc), width=5.2*inch, height=2.4*inch))
        story.append(Paragraph("Figure 3: Adebayo Sanity Check Heatmaps (Trained Weights vs. Randomized Weights)", caption_style))

    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # RAG RETRIEVAL & ADVISORY GENERATION
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. RAG Retrieval & Grounded Bilingual Advisory", h1_style))
    story.append(Paragraph(
        "The knowledge base encompasses 11 verified provincial bulletins (164 chunks) indexed in FAISS IndexFlatIP using "
        "all-MiniLM-L6-v2 embeddings. On 33 domain-specific agricultural queries, the retrieval pipeline achieved: "
        "<b>Hit@1: 96.97%</b>, <b>Hit@3: 100.0%</b>, <b>Hit@5: 100.0%</b>, <b>MRR: 0.9848</b>, and mean CPU latency of <b>21.84 ms</b>.", body_style
    ))
    story.append(Paragraph(
        "<b>Grounded Synthesis & Safety:</b> Generative chemical hallucinations are eliminated through deterministic extractive "
        "synthesis. Advisories link to immutable provenance citations (ARI Tarnab, CCRI Pirsabak, PRC Abbottabad). "
        "A 33-term agricultural glossary and regex entity shielding protect Latin binomials and chemical active ingredients "
        "during Urdu translation, while custom CSS renders native Right-to-Left (RTL) Nastaliq layouts.", body_style
    ))

    # RAG Table
    t_rag_data = [
        [Paragraph("<b>Criteria / Metric</b>", table_hdr), Paragraph("<b>Benchmark Result</b>", table_hdr), Paragraph("<b>Operational Evaluation Note</b>", table_hdr)],
        [Paragraph("Groundedness Rate", table_cell), Paragraph("<b>100.00%</b>", table_cell), Paragraph("All facts extracted directly from retrieved passages", table_cell)],
        [Paragraph("Citation Correctness", table_cell), Paragraph("<b>100.00%</b>", table_cell), Paragraph("All supported cases link verified institutional citations", table_cell)],
        [Paragraph("Evidence Faithfulness", table_cell), Paragraph("<b>100.00%</b>", table_cell), Paragraph("Zero invented dosages, intervals, or trade names", table_cell)],
        [Paragraph("Guardrail Abstention", table_cell), Paragraph("<b>100.00% (4/4)</b>", table_cell), Paragraph("Rejects unsupported crops (Cotton, Wheat, Sugarcane, Citrus)", table_cell)],
        [Paragraph("English Advisory Latency", table_cell), Paragraph("<b>22.05 ms</b>", table_cell), Paragraph("Extractive structured synthesis on CPU", table_cell)],
        [Paragraph("Urdu Advisory Latency", table_cell), Paragraph("<b>15.63 ms</b>", table_cell), Paragraph("Dictionary-cached localization on CPU", table_cell)],
    ]
    t_rag = Table(t_rag_data, colWidths=[130, 110, 240])
    t_rag.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_rag)
    story.append(Paragraph("Table 3: Grounded Advisory Synthesis and Safety Benchmark (N = 15)", caption_style))

    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # END-TO-END WORKFLOW & ABLATIONS
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. End-to-End System Evaluation & Controlled Ablations", h1_style))
    story.append(Paragraph(
        "The complete multimodal workflow was benchmarked across 17 representative cases (11 canonical, 2 difficult edge cases, "
        "and 4 out-of-scope queries). The pipeline achieved a <b>100.0% execution success rate</b> with a mean end-to-end CPU latency "
        "of <b>184.22 ms</b> (P95: 271.51 ms).", body_style
    ))

    t_e2e_lat = [
        [Paragraph("<b>Pipeline Stage</b>", table_hdr), Paragraph("<b>Mean Latency (ms)</b>", table_hdr), Paragraph("<b>P95 Latency (ms)</b>", table_hdr), Paragraph("<b>% of Total</b>", table_hdr)],
        [Paragraph("1. Vision Inference (MobileNetV3)", table_cell), Paragraph("71.56 ms", table_cell), Paragraph("116.52 ms", table_cell), Paragraph("38.8%", table_cell)],
        [Paragraph("2. Grad-CAM Generation (features.16)", table_cell), Paragraph("58.55 ms", table_cell), Paragraph("95.33 ms", table_cell), Paragraph("31.8%", table_cell)],
        [Paragraph("3. FAISS Dense Retrieval (Top-3)", table_cell), Paragraph("15.94 ms", table_cell), Paragraph("18.88 ms", table_cell), Paragraph("8.7%", table_cell)],
        [Paragraph("4. Advisory Synthesis (English)", table_cell), Paragraph("18.62 ms", table_cell), Paragraph("22.09 ms", table_cell), Paragraph("10.1%", table_cell)],
        [Paragraph("5. Urdu Localization & RTL Packaging", table_cell), Paragraph("19.54 ms", table_cell), Paragraph("23.73 ms", table_cell), Paragraph("10.6%", table_cell)],
        [Paragraph("<b>Total End-to-End Multimodal Pipeline</b>", table_hdr), Paragraph("<b>184.22 ms</b>", table_hdr), Paragraph("<b>271.51 ms</b>", table_hdr), Paragraph("<b>100.0%</b>", table_hdr)],
    ]
    t_lat_tab = Table(t_e2e_lat, colWidths=[170, 100, 100, 110])
    t_lat_tab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#F5F5F5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_lat_tab)
    story.append(Paragraph("Table 4: CPU End-to-End Latency Profile Across Pipeline Components (N = 17)", caption_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Controlled Ablation Studies:</b>", h2_style))
    story.append(Paragraph(
        "• <b>Ablation 1 (Classifier Only vs. Classifier + Grad-CAM):</b> Classifier only: 21.84 ms; Classifier + Grad-CAM: 80.90 ms. "
        "The backward pass and interpolation introduce a +59.06 ms overhead (+270.4%), highly acceptable for visual validation.<br/>"
        "• <b>Ablation 2 (Unfiltered vs. Metadata-Filtered Retrieval):</b> Unfiltered semantic search suffered 13.33% cross-crop document leakage "
        "(e.g., retrieving Potato Early Blight for a generic Maize query). Conditioning search on vision metadata achieved <b>0.00% leakage</b>.<br/>"
        "• <b>Ablation 3 (English vs. Urdu Localization):</b> English latency: 20.77 ms; Urdu cached latency: 19.54 ms. Translation expands text by 3% while strictly preserving Latin pathogen terms.<br/>"
        "• <b>Ablation 4 (In-Scope vs. Out-of-Scope Queries):</b> In-scope mean similarity was 0.5458; out-of-scope queries triggered <b>100.0% abstention</b> via lexical and cosine-threshold filters.", body_style
    ))

    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # ERROR ANALYSIS & LIMITATIONS
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Error Analysis & Limitations", h1_style))
    story.append(Paragraph(
        "Out of 1,377 held-out test images, exactly <b>34 images (2.47%) were misclassified</b>. Analysis revealed <b>zero cross-crop errors</b>. "
        "67.6% of errors (23 cases) occurred between Maize Gray Leaf Spot and Northern Leaf Blight due to their shared morphological appearance "
        "as elongated necrotic foliar blights before full rectangular or cigar margins mature. 11.8% of errors (4 cases) involved healthy leaves "
        "with specular lighting glare or surface dust mistaken for early scab or black rot. 8.8% of errors (3 cases) involved tiny incipient "
        "late blight lesions occupying less than 2% of the leaf margin.", body_style
    ))
    story.append(Paragraph(
        "<b>Academic Limitations:</b> (1) The training corpus consists of controlled laboratory captures; field accuracy on smartphone "
        "photos with intense midday glare, weed backgrounds, or multiple concurrent infections may be lower. Field readiness cannot be "
        "claimed without prospective field trials in KP. (2) Grad-CAM provides macroscopic regional localization (7x7 feature map) and "
        "does not prove microbiological causality. (3) Softmax outputs can exhibit overconfidence on completely out-of-distribution non-leaf objects.", body_style
    ))

    # -------------------------------------------------------------------------
    # CONCLUSION & REFERENCES
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(Paragraph("7. Conclusion & Institutional References", h1_style))
    story.append(Paragraph(
        "KP Smart Crop Doctor demonstrates that combining lightweight edge transfer learning with spatial explainability, "
        "strict knowledge-base provenance, and domain guardrails produces a reliable, sub-200ms decision-support tool. "
        "All agricultural recommendations are traceable to published advisories from the Agriculture Research Institute (ARI) "
        "Tarnab Peshawar, Cereal Crops Research Institute (CCRI) Pirsabak Nowshera, and Potato Research Centre (PRC) Abbottabad.", body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated Technical Report PDF at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    build_pdf_report()
