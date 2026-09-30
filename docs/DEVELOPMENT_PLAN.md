# KP Smart Crop Doctor: Capstone Development Plan

## 1. Project Overview & Capstone Objectives

**Project Title:** KP Smart Crop Doctor: An Explainable, Knowledge-Grounded Agricultural Advisory System for Khyber Pakhtunkhwa  
**Domain:** Applied Machine Learning, Computer Vision, Explainable AI (XAI), Retrieval-Augmented Generation (RAG)  
**Target Region:** Khyber Pakhtunkhwa (KP), Pakistan  
**Key Crops Targeted:** Wheat, Maize, Tomato, Potato, Citrus, Apple, Sugarcane, Tobacco  

### Core Problem Statement
Farmers in Khyber Pakhtunkhwa face significant yield loss due to plant diseases and pest infestations. Traditional agricultural extension services are constrained by geographic barriers and staffing limitations. While standard AI models can classify leaf diseases, they function as "black boxes" lacking interpretability, and generic Large Language Models frequently hallucinate chemical treatments or prescribe recommendations unsuited to KP's regional agro-ecological zones and local regulatory frameworks.

### Solution Pillars
1. **Transfer Learning Vision Classifier:** Lightweight backbones (e.g., MobileNetV3-Large, EfficientNet-B0) fine-tuned for high accuracy on leaf pathogens with calibrated top-1 and top-2 predictions.
2. **Visual Explainability (Grad-CAM):** Gradient-weighted Class Activation Mapping highlighting pathological regions (lesions, chlorosis, necrosis) to build user trust and verify model focus.
3. **Curated Agricultural Knowledge Base & RAG:** Context-grounded retrieval from Khyber Pakhtunkhwa Agriculture Extension literature, PARC, and verified pest management manuals to prevent LLM hallucinations.
4. **Bilingual Advisory (English & Urdu):** Dual-language interface and translated management advice accessible to provincial farmers and field extension workers.
5. **Interactive Streamlit Interface:** User-friendly diagnostic tool with interactive heatmap overlays, confidence metrics, and grounded treatment guidance.

---

## 2. Milestone Roadmap

```
+------------------------------------------------------------------------------------+
| Milestone 1: Scaffolding, Architecture & Reproducibility Setup [CURRENT]           |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Milestone 2: Dataset Selection, KP Crop Filtering & Stratified Partitioning       |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Milestone 3: Transfer Learning Classifier with Top-2 Confidence Estimation        |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Milestone 4: Explainable AI Pipeline (Custom Grad-CAM Implementation)              |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Milestone 5: Curated KP Knowledge Base & Source-Grounded RAG Pipeline              |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Milestone 6: Multilingual Advisory Integration (English & Urdu)                   |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Milestone 7: Interactive Streamlit Web Application Dashboard                       |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
| Milestone 8: Comprehensive Evaluation, Benchmarking & Capstone Defense Pack        |
+------------------------------------------------------------------------------------+
```

---

### Milestone 1: Scaffolding, Architecture & Reproducibility (Completed)
* **Goal:** Establish a modular repository layout, centralized configuration, dependency specifications, and reproducibility foundations.
* **Deliverables:**
  - Project directory structure (`src/`, `data/`, `models/`, `knowledge_base/`, `evaluation/`, `tests/`).
  - `requirements.txt` with pinned/compatible versions for Vision, RAG, and Streamlit.
  - `.gitignore` preventing data, secret, and heavy binary leaks.
  - Centralized `src/config.py` for uniform paths, hyperparameters, and seeds.
  - Seed-setting and device-management utilities in `src/utils.py`.
  - Academic-grade `README.md` and detailed `docs/DEVELOPMENT_PLAN.md`.
  - Baseline environment validation tests in `tests/test_environment.py`.
* **Verification:** `pytest tests/test_environment.py` passes cleanly.

---

### Milestone 2: Dataset Selection, KP Crop Filtering & Stratified Partitioning (Completed)
* **Goal:** Define and assemble the leaf disease dataset focusing strictly on crops of economic importance to Khyber Pakhtunkhwa, establishing reproducible data manifests without data leakage.
* **Key Tasks:**
  - Audit open-access agricultural datasets against 9 objective criteria.
  - Finalized 11 classes across Maize, Potato, and Apple (9,175 total images, 256x256 RGB, 0 corrupt files, CC0 license).
  - Implemented `src/data/download.py`, `src/data/profiling.py`, `src/data/split.py`, `src/data/run_pipeline.py`.
  - Generated group-stratified (leaf-group isolated) splits with zero leakage: Train: 6,422 (70.0%), Val: 1,376 (15.0%), Test: 1,377 (15.0%) with SHA-256 manifests in `data/splits/`.
* **Deliverables:** `dataset_manifest.csv`, `train.csv`, `val.csv`, `test.csv`, and data verification unit tests in `tests/test_data.py`.
* **Exit Criteria:** Zero overlap between splits; class distributions match stratified ratios; verifiable with `pytest tests/test_data.py`.

---

### Milestone 3: Transfer Learning Classifier with Top-2 Confidence Estimation (Completed)
* **Goal:** Build, train, and evaluate a lightweight deep learning vision classifier tailored for resource-constrained agricultural deployment.
* **Key Tasks:**
  - Implemented model wrappers for `MobileNetV3-Large` and `EfficientNet-B0`.
  - Built training pipeline under strictly identical experimental setups (AdamW, CosineAnnealingLR, seed 42, class-weighted CE loss).
  - Validation Macro-F1 model selection:
    - **MobileNetV3-Large (Selected):** Val Macro-F1: **97.73%**, Test Top-1: **97.53%**, Top-2: **99.78%**, Test Macro-F1: **96.57%**, ECE: **1.40%**, CPU Latency: **28.4 ms**, Size: **16.3 MB**.
    - **EfficientNet-B0:** Val Macro-F1: **95.35%**, Test Top-1: **95.06%**, Top-2: **99.35%**, Test Macro-F1: **93.52%**, ECE: **5.89%**, CPU Latency: **48.0 ms**, Size: **15.6 MB**.
  - Generated evaluation artifacts, confusion matrices, reliability diagrams in `evaluation/figures/`.
* **Deliverables:** Trained model checkpoint `models/best_crop_model.pt`, label mapping JSON `models/label_encoder.json`, evaluation metric report `evaluation/comparative_evaluation_report.json`.
* **Exit Criteria:** Top-1 Accuracy >= 92%, Top-2 Accuracy >= 97% on unseen test split; deterministic inference tested.

---

### Milestone 4: Explainable AI Pipeline (Custom Grad-CAM Implementation) (Completed)
* **Goal:** Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize convolutional feature representations and verify diagnostic reasoning.
* **Key Tasks:**
  - Built `src/explainability/gradcam.py` using forward and backward PyTorch hooks on the final convolutional feature layer (`model.features[16]`).
  - Mathematical formulation implemented with global average pooling of gradients:
    $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{i,j}}$$
    $$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
  - Implemented bilinear interpolation to input dimensions (224x224), min-max normalization, colormap heatmap generation, and alpha blending overlays.
  - Implemented reusable inference & explainability API `predict_and_explain` returning predicted class, confidence, top-2 predictions, original image, Grad-CAM heatmap, and overlay.
  - Implemented Adebayo et al. (NeurIPS 2018) model-randomization sanity checks with Pearson correlation ($r$), Spearman rank correlation ($\rho$), and MAE.
  - Evaluated on all 11 classes, specifically verifying difficult classes (e.g., Maize Cercospora/Gray Leaf Spot and Potato healthy).
  - Generated publication-grade figures in `evaluation/figures/gradcam_11_classes_grid.png` and `evaluation/figures/gradcam_sanity_check_randomization.png`.
* **Deliverables:** `src/explainability/gradcam.py`, `src/explainability/evaluate_gradcam.py`, `evaluation/gradcam_evaluation_report.json`, test suite `tests/test_gradcam.py` (6 passing unit/integration tests).
* **Exit Criteria:** Heatmaps cleanly localize diagnostic symptoms across all 11 classes; sanity check proves parameter sensitivity ($r \approx 0 < 0.35$); all 14 tests passing.

---

### Milestone 5: Curated KP Knowledge Base & Source-Grounded RAG Pipeline (Completed)
* **Goal:** Construct a verified agricultural knowledge repository and an independent, source-grounded retrieval pipeline that delivers localized agronomic advice with authoritative citations.
* **Key Tasks:**
  - Curated 11 comprehensive agronomic advisory documents with full provenance metadata for all 11 supported classes across Apple, Maize, and Potato.
  - Sourced from authoritative KP and Pakistan agricultural research institutions (ARI Tarnab Peshawar, CCRI Pirsabak Nowshera, PRC Abbottabad, UAP, PARC, FSC&RD, and CABI Plantwise Pakistan).
  - Built master provenance catalog in `knowledge_base/metadata/provenance_catalog.json`.
  - Implemented document ingestion (`src/rag/document_loader.py`) and section-aware chunking (`src/rag/chunker.py`) producing 164 contextualized passages.
  - Generated 384-dimensional dense embeddings with `sentence-transformers/all-MiniLM-L6-v2` (`src/rag/embedder.py`).
  - Built local FAISS `IndexFlatIP` vector index (`knowledge_base/vector_store/faiss_index.bin`) and chunk metadata store.
  - Implemented `src/rag/retriever.py` with crop/disease filtering, formatted citations, and two-tiered safety guardrails against unsupported/out-of-domain queries.
  - Executed quantitative retrieval benchmark on 37 test queries: Hit@1: **96.97%**, Hit@3: **100.00%**, Hit@5: **100.00%**, MRR: **0.9848**, Latency: **21.84 ms**, Guardrail Accuracy: **100.00%**.
* **Deliverables:** `knowledge_base/documents/`, `knowledge_base/metadata/provenance_catalog.json`, `knowledge_base/vector_store/`, `src/rag/`, `evaluation/retrieval_evaluation_report.json`, `tests/test_rag.py` (12 unit tests; 26/26 tests passing).
* **Exit Criteria:** Hit@3 = 100%, MRR >= 0.95; zero unverified recommendations; all tests passing cleanly.

---

### Milestone 6: Multilingual Advisory Integration (English & Urdu) (Next Phase)
* **Goal:** Enable bilingual support across the entire diagnostic and advisory pipeline to serve diverse user demographics in KP.
* **Key Tasks:**
  - Implement `src/rag/translator.py` with translation caching to minimize latency.
  - Localize diagnostic reports: disease names, severity levels, confidence summaries, and treatment instructions translated into fluent, natural Urdu.
  - Maintain an agricultural vocabulary glossary (English <-> Urdu) for specialized terms (e.g., Early Blight -> اگیتی جھلساؤ, Powdery Mildew -> سفوفی پھپھوندی, Fungicide -> پھپھوند کش).
  - Add RTL (Right-to-Left) formatting handlers for clean Urdu text rendering in Streamlit.
* **Deliverables:** `src/rag/translator.py`, Urdu agricultural dictionary JSON, translation test cases.
* **Exit Criteria:** Accurate translation verified on 20 standard crop disease advisories without mangled Urdu script.

---

### Milestone 7: Interactive Streamlit Web Application Dashboard
* **Goal:** Unify vision classification, Grad-CAM visualization, RAG advisory, and bilingual translation into an intuitive, polished web dashboard.
* **Key Tasks:**
  - Build `app.py` and modular layout components in `src/app/ui_components.py`.
  - Implement responsive controls:
    - Leaf image uploader (drag & drop, sample selector)
    - Language switcher (English / اردو)
    - Disease diagnosis card with Top-1 and Top-2 confidence progress bars
    - Interactive Grad-CAM inspection viewer (slider to adjust heatmap transparency overlay)
    - Grounded Advisory Section: Symptoms, Causes, Organic Management, Chemical Treatments, and Source Citations
    - Interactive QA chat assistant grounded in the retrieved documents for the diagnosed crop
  - Cache heavy models (`@st.cache_resource`) for fast sub-second inference.
* **Deliverables:** Fully functioning `app.py` adhering to Streamlit best practices.
* **Exit Criteria:** App launches with `streamlit run app.py`; clean UX on desktop and mobile viewports.

---

### Milestone 8: Comprehensive Evaluation, Benchmarking & Capstone Defense Pack
* **Goal:** Perform rigorous quantitative evaluation, prepare validation reports, and assemble all academic capstone deliverables.
* **Key Tasks:**
  - Run end-to-end evaluation benchmark across test split:
    - Classification metrics: Precision, Recall, F1-Score, Confusion Matrix
    - Inference latency (CPU vs GPU in ms per sample)
    - XAI visual sanity check
    - RAG answer faithfulness score
  - Package sample test cases and demonstration notebook in `notebooks/capstone_demo.ipynb`.
  - Assemble capstone presentation slides, final thesis documentation chapter references, and reproducibility instructions.
* **Deliverables:** `evaluation/benchmark_report.json`, confusion matrix figures, capstone demonstration notebook.
* **Exit Criteria:** All automated tests passing; full pipeline executable with a single reproduction command.

---

## 3. Reproducibility Standards

To meet university capstone requirements, KP Smart Crop Doctor adheres to the following reproducibility criteria:
1. **Deterministic Random Seeds:** All stochastic operations (data splitting, weight initialization, augmentations) are seeded using `src.utils.set_seed(42)`.
2. **Environment Isolation:** Pinned dependencies in `requirements.txt` with tested compatibility.
3. **Data Integrity:** Dataset splits are versioned via deterministic CSV manifests referencing file paths and class indices.
4. **Model Checkpoint Traceability:** Every saved checkpoint records the architecture name, training hyperparameters, validation loss, and timestamp.
5. **Configuration Decoupling:** All tunable parameters are isolated in `src/config.py` and `.env`, avoiding hardcoded constants.
