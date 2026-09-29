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

### Milestone 1: Scaffolding, Architecture & Reproducibility (Current Phase)
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

### Milestone 2: Dataset Selection, KP Crop Filtering & Stratified Partitioning
* **Goal:** Define and assemble the leaf disease dataset focusing strictly on crops of economic importance to Khyber Pakhtunkhwa, establishing reproducible data manifests without data leakage.
* **Key Tasks:**
  - Audit open-access agricultural datasets (e.g., PlantVillage, PlantDoc, regional repositories).
  - Select classes relevant to KP agricultural profiles:
    - *Wheat:* Leaf Rust, Septoria, Healthy
    - *Maize:* Common Rust, Northern Leaf Blight, Gray Leaf Spot, Healthy
    - *Tomato:* Early Blight, Late Blight, Bacterial Spot, Yellow Leaf Curl Virus, Healthy
    - *Potato:* Early Blight, Late Blight, Healthy
    - *Citrus / Apple:* Black Rot, Scab, Citrus Greening, Healthy
  - Create `src/data/download.py` (scripted download or local extraction instructions).
  - Implement `src/data/preprocess.py` for image normalization, duplicate removal, and corruption validation.
  - Generate deterministic stratified train / validation / test splits (70% / 15% / 15%) and save CSV manifests with SHA256 checksums in `data/splits/`.
* **Deliverables:** `dataset_manifest.csv`, `train.csv`, `val.csv`, `test.csv`, and data verification unit tests.
* **Exit Criteria:** Zero overlap between splits; class distributions match stratified ratios; verifiable with `pytest tests/test_data.py`.

---

### Milestone 3: Transfer Learning Classifier with Top-2 Confidence Estimation
* **Goal:** Build, train, and evaluate a lightweight deep learning vision classifier tailored for resource-constrained agricultural deployment.
* **Key Tasks:**
  - Implement model wrapper in `src/vision/model.py` supporting `MobileNetV3-Large` and `EfficientNet-B0` with customizable classifier heads and dropout.
  - Build PyTorch dataset & data loader pipeline with data augmentations (random horizontal flip, rotation, color jitter) in `src/vision/dataset.py`.
  - Implement modular training script with early stopping, learning rate scheduler, mixed precision, and checkpointing in `src/vision/train.py`.
  - Build inference pipeline `src/vision/predict.py` calculating softmax probabilities, top-1 and top-2 class predictions, and confidence margins.
  - Implement classification evaluation suite in `evaluation/metrics.py` (Top-1 Accuracy, Top-2 Accuracy, Macro-F1, Confusion Matrix).
* **Deliverables:** Trained model checkpoint `models/best_crop_model.pt`, label mapping JSON, evaluation metric report.
* **Exit Criteria:** Top-1 Accuracy >= 92%, Top-2 Accuracy >= 97% on unseen test split; deterministic inference tested.

---

### Milestone 4: Explainable AI Pipeline (Custom Grad-CAM Implementation)
* **Goal:** Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize convolutional feature representations and verify diagnostic reasoning.
* **Key Tasks:**
  - Build `src/explainability/gradcam.py` using forward and backward PyTorch hooks on the final convolutional feature layer.
  - Compute class-specific gradients with respect to activation maps:
    $$A^k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{i,j}}$$
    $$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
  - Implement heatmap generation, bilateral filtering / interpolation to original image dimensions (224x224), and colormap blending.
  - Generate visual explanation artifacts showing original leaf, heatmap, and overlay side-by-side for both top-1 and top-2 predicted classes.
* **Deliverables:** `src/explainability/gradcam.py`, unit test verifying hook registration and non-empty heatmap generation.
* **Exit Criteria:** Heatmap cleanly localizes diseased spots (verified qualitatively on test samples); no gradient leaks or memory leaks during inference.

---

### Milestone 5: Curated KP Knowledge Base & Source-Grounded RAG Pipeline
* **Goal:** Construct a verified agricultural knowledge repository and a Retrieval-Augmented Generation pipeline that synthesizes localized agronomic advice while citing verified sources.
* **Key Tasks:**
  - Curate official agricultural documents:
    - KP Department of Agriculture Extension advisories
    - Pakistan Agricultural Research Council (PARC) pest management guidelines
    - Organic and chemical treatment protocols with recommended dosage and safety precautions
  - Implement document processing in `src/rag/kb_indexer.py`: text extraction, metadata extraction (crop, disease, region, treatment type), chunking with overlap.
  - Generate dense embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2`) and build FAISS vector index in `src/rag/retriever.py`.
  - Implement context-grounded prompt synthesis in `src/rag/generator.py` enforcing strict source citation and fallback warning when context is insufficient.
  - Implement hallucination evaluation in `evaluation/rag_eval.py` (source overlap and groundedness checks).
* **Deliverables:** Vector index in `knowledge_base/vector_store/`, document metadata in `knowledge_base/metadata/`, retrieval & generation pipeline.
* **Exit Criteria:** RAG system returns answers citing specific sections/bulletins; 100% adherence to retrieved facts on standard evaluation prompts.

---

### Milestone 6: Multilingual Advisory Integration (English & Urdu)
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
