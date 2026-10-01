# KP Smart Crop Doctor: Capstone Defense Pack

**Undergraduate / Graduate Capstone Project Defense Reference Document**  
**Project Title:** KP Smart Crop Doctor: An Explainable, Knowledge-Grounded Agricultural Advisory System for Khyber Pakhtunkhwa  
**Domain:** Applied Computer Vision, Explainable AI (XAI), and Retrieval-Augmented Generation (RAG) for Agronomy  
**Target Region:** Khyber Pakhtunkhwa (KP), Pakistan  
**Repository Architecture:** PyTorch, Torchvision, FAISS, Sentence-Transformers, Streamlit  
**Date:** October 2026  

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Motivation for KP Agriculture](#2-motivation-for-kp-agriculture)
3. [Research Objectives](#3-research-objectives)
4. [System Architecture](#4-system-architecture)
5. [Dataset and Preprocessing](#5-dataset-and-preprocessing)
6. [Leakage Prevention](#6-leakage-prevention)
7. [Model Selection](#7-model-selection)
8. [Vision Results](#8-vision-results)
9. [Grad-CAM Methodology](#9-grad-cam-methodology)
10. [RAG Architecture](#10-rag-architecture)
11. [Knowledge-Base Provenance](#11-knowledge-base-provenance)
12. [Multilingual Design](#12-multilingual-design)
13. [Streamlit Workflow](#13-streamlit-workflow)
14. [Evaluation Methodology](#14-evaluation-methodology)
15. [Results Summary](#15-results-summary)
16. [Limitations](#16-limitations)
17. [Ethical and Safety Considerations](#17-ethical-and-safety-considerations)
18. [Future Work](#18-future-work)
19. [Likely Viva Questions and Concise Answers](#19-likely-viva-questions-and-concise-answers)

---

## 1. Problem Statement

Smallholder farmers and agricultural extension officers across Pakistan, particularly in Khyber Pakhtunkhwa (KP), face severe crop losses due to foliar plant diseases. The existing diagnostic workflow exhibits critical structural failure modes:
1. **Black-box Deep Learning:** Standard convolutional vision classifiers provide class labels without visual justification. When a model misclassifies a disease, farmers cannot verify which region of the foliage triggered the prediction, causing distrust.
2. **Generative LLM Hallucinations:** Direct deployment of large language models (LLMs) for agronomy produces hazardous ungrounded advice, including fabricated chemical dosages, non-existent pesticides, and dangerous mixing protocols absent from regional extension recommendations.
3. **Lack of Regional Institutional Grounding:** Global agricultural models cite international chemicals unavailable in Pakistan, overlooking locally registered formulations, cultural practices, and climatic conditions specific to KP valleys.
4. **Linguistic Barrier:** Scientific advisories are predominantly published in English, whereas KP farming communities require native, culturally fluent **Urdu** guidance rendered in correct Right-to-Left (RTL) typography.

**KP Smart Crop Doctor** addresses these problems by integrating a lightweight transfer learning classifier (**MobileNetV3-Large**), spatial interpretability (**Grad-CAM**), and a verifiable **Retrieval-Augmented Generation (RAG)** pipeline grounded exclusively in verified institutional bulletins from KP agricultural authorities.

---

## 2. Motivation for KP Agriculture

Agriculture constitutes over **20% of KP's Gross Regional Product** and employs nearly **40% of the provincial labor force**. The province is characterized by distinct agro-ecological zones ranging from high-altitude temperate valleys (Swat, Dir, Abbottabad, Mansehra) to semi-arid southern plains (D.I. Khan, Bannu):

* **Apple (*Malus domestica*):** The cornerstone cash crop of Swat and Hazara divisions. Apple Scab (*Venturia inaequalis*), Black Rot (*Botryosphaeria obtusa*), and Cedar Apple Rust cause seasonal yield drops of $30\text{--}60\%$ when early infection sprays are missed.
* **Maize (*Zea mays*):** The primary staple cereal across Swabi, Mardan, and Nowshera during the Kharif season. Foliar blights—Gray Leaf Spot (*Cercospora zeae-maydis*) and Northern Corn Leaf Blight (*Exserohilum turcicum*)—frequently decimate grain filling under humid monsoon conditions.
* **Potato (*Solanum tuberosum*):** Extensively cultivated across Kalam, Swat, Kaghan, and Battagram. Late Blight (*Phytophthora infestans*) is notoriously devastating, capable of causing total field loss within 7 to 10 days under persistent fog and high humidity.

Due to the limited ratio of government extension officers to farming households (estimated at less than 1 officer per 1,500 farming families in rural KP), an automated, verifiable decision-support tool can drastically reduce diagnostic latency.

---

## 3. Research Objectives

1. **Robust Computer Vision:** Train and benchmark an edge-compatible transfer learning vision model capable of distinguishing 11 crop disease and control classes across Apple, Maize, and Potato with high macro-F1 and tight probability calibration.
2. **Scientific Visual Verification:** Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to generate spatial attribution heatmaps, and scientifically validate that explanations depend on learned network parameters using cascading weight randomization.
3. **Traceable Knowledge Retrieval:** Build a localized agricultural vector repository using dense bi-encoder embeddings and FAISS, enabling sub-30ms semantic retrieval of verified extension bulletins from ARI Tarnab and PARC.
4. **Grounded Non-Hallucinatory Synthesis:** Develop an extractive advisory generation engine that synthesizes symptoms, causes, IPM practices, chemical options, and institutional citations strictly from retrieved evidence.
5. **Calibrated Guardrails:** Implement explicit domain guardrails to detect and abstain from answering queries regarding out-of-scope crops (e.g., Cotton, Wheat, Sugarcane, Citrus) to prevent hazardous out-of-distribution hallucinations.
6. **Bilingual Presentation:** Deploy a full-stack interactive Streamlit dashboard supporting seamless toggle between English and Urdu (with native Right-to-Left styling) and dynamic opacity inspection.

---

## 4. System Architecture

The end-to-end pipeline operates as a modular, decoupled sequence of micro-components:

```
[Leaf Image Input] (Upload or Preloaded Gallery)
       |
       v
[Image Preprocessing Pipeline] (Resize 256x256 -> CenterCrop 224x224 -> ImageNet Normalization)
       |
       v
[MobileNetV3-Large Classifier] (4.2M Params, PyTorch CPU Inference: ~28 ms)
       |
       +---> [Logits & Softmax] ---> Top-1 Diagnosis + Top-2 Comparison + ECE Notice
       |
       v
[Grad-CAM Hook Engine] (features.16: 7x7 Feature Map -> Gradient Weighting -> ReLU)
       |
       v
[Spatial Heatmap & Alpha Overlay] (Dynamic vector blending with user opacity slider)
       |
       +======================================================+
       | Vision Metadata (Predicted Crop & Disease Class)    |
       +==========================+===========================+
                                  |
                                  v
       [RAG Query Dispatcher] (Hybrid Query = Crop + Disease + User Query)
                                  |
                                  v
       [Agricultural Guardrail] (Checks out-of-scope crop whitelist/blacklist)
              |                                            |
              | (In-Scope)                                 | (Unsupported Crop)
              v                                            v
       [FAISS Vector Store]                       [Refusal Controller]
       - sentence-transformers/all-MiniLM-L6-v2   - status="information_unavailable"
       - Cosine Top-k Similarity Search           - Zero Citations Emitted
       - 11 Curated KP Bulletins (164 Chunks)     - Safe Extension Officer Referral
              |
              v
       [Retrieved Evidence Chunks] (Doc ID, Title, Text, Institutional Metadata)
              |
              v
       [Local Grounded Synthesizer] (Extracts Symptoms, Pathogens, IPM, Citations)
              |
              v
       [Bilingual Localization Engine]
       - English Advisory (LTR, Latin binomials preserved)
       - Urdu Advisory (RTL, Noto Nastaliq Urdu, protected chemical terms)
              |
              v
       [Streamlit UI Dashboard] (Tabbed cards, interactive opacity, follow-up Q&A)
```

---

## 5. Dataset and Preprocessing

* **Core Dataset:** Curated subset of standardized foliar captures derived from PlantVillage and regional repositories, restricted to **3 target crops and 11 distinct classes**:
  1. `Apple___Apple_scab`
  2. `Apple___Black_rot`
  3. `Apple___Cedar_apple_rust`
  4. `Apple___healthy`
  5. `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
  6. `Corn_(maize)___Common_rust_`
  7. `Corn_(maize)___Northern_Leaf_Blight`
  8. `Corn_(maize)___healthy`
  9. `Potato___Early_blight`
  10. `Potato___Late_blight`
  11. `Potato___healthy`
* **Total Image Count:** $N = 9,175$ images.
  - Training Split: $6,422$ images ($70\%$)
  - Validation Split: $1,376$ images ($15\%$)
  - Held-out Test Split: $1,377$ images ($15\%$)
* **Preprocessing Protocol:**
  - Training: Resize to $256\times 256$, Random Resized Crop ($224\times 224$, scale $0.8\text{--}1.0$), Random Horizontal Flip ($p=0.5$), Random Rotation ($\pm 15^\circ$), Color Jitter (brightness, contrast, saturation $\pm 0.1$), ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  - Validation & Testing: Resize to $256\times 256$, deterministic Center Crop ($224\times 224$), ImageNet normalization.

---

## 6. Leakage Prevention

A fundamental flaw in naive agricultural computer vision studies is **data leakage caused by multiple photographic shots of the exact same leaf appearing across both train and test splits**. This inflates test accuracy artificially.

### Prevention Protocol Implemented:
1. **Leaf-Group Isolation:** PlantVillage file naming conventions encode identical leaves sharing photo sessions (e.g. `00075aa8-d81a-4e84...`). An automated parser (`extract_leaf_id`) grouped identical leaves into distinct clusters.
2. **Perceptual Difference Hashing (dHash):** Near-duplicate images taken from slightly altered camera angles were identified via $64$-bit gradient perceptual hashing. Pairs with Hamming distance $\le 4$ were constrained to identical split buckets.
3. **Group Stratification:** Splitting was performed using `GroupKFold` / `StratifiedGroupSplit` conditioned on `leaf_id`, ensuring that **no leaf or near-duplicate represented in the training set ever appeared in the validation or test sets**.
4. **Leakage Verification:** Automated tests (`tests/test_data_pipeline.py`) verify mathematically that:
   $$\text{Set}(\text{LeafIDs}_{\text{train}}) \cap \text{Set}(\text{LeafIDs}_{\text{val}}) = \emptyset$$
   $$\text{Set}(\text{LeafIDs}_{\text{train}}) \cap \text{Set}(\text{LeafIDs}_{\text{test}}) = \emptyset$$

---

## 7. Model Selection

Model selection was conducted under strict academic guidelines: **the decision was locked based on Validation Macro-F1 prior to evaluating the held-out test set**.

Two backbones representing state-of-the-art mobile edge vision were evaluated under identical training conditions:
* **Candidate 1:** MobileNetV3-Large (Width multiplier 1.0, 4.2M params)
* **Candidate 2:** EfficientNet-B0 (Compound scaling, 4.0M params)

### Selection Outcome:
* MobileNetV3-Large attained **$0.9773$ Validation Macro-F1** vs. EfficientNet-B0's **$0.9535$** ($+2.49\%$ margin).
* MobileNetV3-Large exhibited **$28.38\text{ ms}$ mean CPU latency** vs. EfficientNet-B0's **$48.01\text{ ms}$** ($69\%$ faster throughput).
* MobileNetV3-Large exhibited superior probability calibration: **$\text{ECE} = 1.40\%$** vs. $5.89\%$.
* Consequently, **MobileNetV3-Large was formally selected as the project's primary vision model**.

---

## 8. Vision Results

On the finalized, held-out test split ($N = 1,377$):
* **Top-1 Accuracy:** **$97.53\%$** ($1,343 / 1,377$).
* **Top-2 Accuracy:** **$99.78\%$** ($1,374 / 1,377$).
* **Macro Precision:** **$96.47\%$**.
* **Macro Recall:** **$96.72\%$**.
* **Macro F1-Score:** **$96.57\%$**.
* **Expected Calibration Error (ECE):** **$0.0140$ ($1.40\%$)**, demonstrating that softmax probability outputs closely match empirical ground-truth accuracy.
* **Inference Speed:** **$28.38\text{ ms}$ on standard CPU** ($35.2$ FPS).

### Key Per-Class Observations:
* Common Rust in Maize and Healthy Maize achieved **$100\%$ precision and recall**.
* Potato Early Blight achieved **$0.9967$ F1-score** ($150 / 150$ recall).
* The lowest individual class score was **Maize Gray Leaf Spot ($0.8435$ F1, $0.8052$ recall)**, caused by morphological ambiguity with Northern Leaf Blight during early lesion stages.

---

## 9. Grad-CAM Methodology

Gradient-weighted Class Activation Mapping (Grad-CAM) [Selvaraju et al., ICCV 2017] computes a visual localization map $L_{\text{Grad-CAM}}^c \in \mathbb{R}^{u \times v}$ for target class $c$:

1. **Activation Capture:** Forward hook captures activations $A^k \in \mathbb{R}^{u \times v}$ of feature map $k$ at target layer $L =$ `features.16`.
2. **Neuron Importance Weights:** Backward hook captures gradients of class score $y^c$ with respect to activations, followed by global average pooling:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Rectified Linear Combination:**
   $$L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
4. **Sanity Check Validation (Adebayo et al., NeurIPS 2018):** When network weights were replaced with randomized parameters, the Pearson correlation between trained and randomized heatmaps collapsed from $1.00$ to **$0.0000$** ($MAE \approx 0.24\text{--}0.41$). This verifies that Grad-CAM reflects true learned discriminative features rather than acting as an invariant edge detector.

---

## 10. RAG Architecture

To eliminate the hallucination risks of generative LLMs, the advisory engine uses a **source-grounded RAG pipeline**:
1. **Document Corpus:** 11 comprehensive bulletins covering each supported disease and healthy crop management.
2. **Chunking Engine:** Structural paragraph chunking ($max\_chunk\_size = 500$, $overlap = 50$) with contextual headers and metadata preservation.
3. **Bi-Encoder Dense Retrieval:** `sentence-transformers/all-MiniLM-L6-v2` encodes text chunks into 384-dimensional unit vectors.
4. **Vector Store:** Local FAISS `IndexFlatIP` computes exact inner-product (cosine) similarity.
5. **Vision-Guided Query Formulation:** Diagnostic context from the vision model (predicted crop + disease class) is injected into the user query, guaranteeing that retrieval prioritizes the correct agronomic context.
6. **Guardrail Interceptor:** Lexical and cosine-threshold filters verify that queries match the 11 supported classes before retrieval execution.

---

## 11. Knowledge-Base Provenance

Every document in the knowledge base is traceable to an official public agricultural institution. Fabricated sources and internet blog posts were strictly excluded.

### Provenance Catalog:
| Document ID | Target Crop & Disease | Authoring Institution | Publication | Source Traceability |
| :--- | :--- | :--- | :---: | :--- |
| `KP-AGRI-APP-001` | Apple: Scab | ARI Tarnab, Peshawar & KP Ext. | 2023-04 | Verified Extension Advisory |
| `KP-AGRI-APP-002` | Apple: Black Rot | ARI Tarnab & UAP Peshawar | 2022-06 | Academic Research Guide |
| `KP-AGRI-APP-003` | Apple: Cedar Rust | Plant Pathology ARI Tarnab | 2021-08 | Disease Control Bulletin |
| `KP-AGRI-APP-004` | Apple: Healthy Care | KP Dept of Agriculture Extension | 2023-01 | Orchard Management Manual |
| `KP-AGRI-MAI-001` | Maize: Gray Leaf Spot | CCRI Pirsabak, Nowshera & PARC | 2022-05 | Cereal Protection Guide |
| `KP-AGRI-MAI-002` | Maize: Common Rust | CCRI Pirsabak & ARI Tarnab | 2021-07 | Fungal Rust Bulletin |
| `KP-AGRI-MAI-003` | Maize: Northern Blight | Maize Program CCRI Pirsabak | 2023-03 | Foliar Blight Advisory |
| `KP-AGRI-MAI-004` | Maize: Healthy Care | KP Agriculture Extension | 2023-02 | Production Technology Guide |
| `KP-AGRI-POT-001` | Potato: Early Blight | Potato Res. Centre Abbottabad | 2022-09 | Tuber Crop Advisory |
| `KP-AGRI-POT-002` | Potato: Late Blight | PRC Abbottabad & ARI Tarnab | 2023-05 | Epidemic Warning Bulletin |
| `KP-AGRI-POT-003` | Potato: Healthy Care | KP Agriculture Extension | 2023-01 | Seed & Tuber Best Practices |

---

## 12. Multilingual Design

To serve KP's linguistic context, the system provides a dual-language interface:
* **English:** Standard academic/extension terminology, preserving binomial nomenclature (*Venturia inaequalis*, *Phytophthora infestans*).
* **Urdu:** Culturally fluent translation powered by:
  1. A curated 33-term agricultural bilingual glossary (`knowledge_base/metadata/agricultural_glossary_ur.json`).
  2. Entity protection regex that shields scientific Latin pathogen names and institutional IDs from phonetic corruption.
  3. Persistent translation caching (`knowledge_base/cache/translation_cache.json`) for deterministic, sub-20ms rendering.
  4. Proper Right-to-Left (RTL) HTML containers with Google Noto Nastaliq Urdu typography.

---

## 13. Streamlit Workflow

The user-facing application (`app.py`) is structured as a single-page reactive dashboard:
1. **Header & Non-Replacement Disclaimer:** Prominently alerts users that the system is an academic decision-support prototype and does not replace in-person consultation with certified KP extension officers.
2. **Input Controller:** Allows users to choose between uploading field photos or clicking representative samples from an 11-class preloaded gallery.
3. **Bilingual Switcher:** Toggles the entire UI, diagnosis cards, and advisories between English and Urdu in real time.
4. **Diagnostic Card:** Displays Top-1 class diagnosis with confidence percentage alongside Top-2 comparative probability bars and a softmax calibration disclaimer.
5. **Interactive Grad-CAM Viewer:** Side-by-side display of the original leaf, raw jet heatmap, and blended overlay, equipped with a dynamic opacity slider ($\alpha = 0.1\text{--}0.9$).
6. **Structured Advisory Tabs:** Tabbed navigation presenting Symptoms, Likely Cause, Integrated Management, Cultural Prevention, Crop-Stage Guidance, and Institutional Citation Badges.
7. **Conversational Follow-up Q&A:** Context-constrained chat interface enabling users to ask follow-up questions regarding the diagnosed leaf, strictly restricted by knowledge-base guardrails.

---

## 14. Evaluation Methodology

Evaluation was structured to assess every layer of the system independently and collectively:
1. **Vision Classification:** Top-1, Top-2 accuracy, Macro Precision, Macro Recall, Macro F1, Per-Class F1, Confusion Matrix, and Expected Calibration Error (ECE) across 1,377 held-out test images.
2. **Explainability Sanity Checks:** Pearson $r$, Spearman $\rho$, and MAE computed between trained and weight-randomized models across multiple disease classes.
3. **Retrieval Benchmark:** Hit@1, Hit@3, Hit@5, MRR, Precision@k, and latency across 33 domain-specific agricultural queries and 4 adversarial queries.
4. **Advisory Grounding:** Groundedness rate, citation correctness, evidence faithfulness, and out-of-scope abstention evaluated across 15 structured cases.
5. **End-to-End System Evaluation:** 17 representative workflows (11 canonical, 2 difficult edge cases, 4 out-of-scope queries) measuring end-to-end success rate, latency breakdown, and citation linkage.
6. **Controlled Ablations:** Controlled empirical comparisons of Vision vs. Grad-CAM, Filtered vs. Unfiltered Retrieval, English vs. Urdu, and In-Scope vs. Out-of-Scope queries.

---

## 15. Results Summary

*(For complete tabulated figures, refer to [docs/RESULTS_SUMMARY.md](file:///c:/Users/DELL/Downloads/kp-smart-crop-doctor/docs/RESULTS_SUMMARY.md))*

* **Vision Model:** $97.53\%$ Top-1 Accuracy, $99.78\%$ Top-2 Accuracy, $0.9657$ Macro-F1, $1.40\%$ ECE, $28.38\text{ ms}$ CPU latency.
* **Grad-CAM Sanity Check:** $r = 0.0000$, $\rho \approx -0.04\text{--}0.18$, $MAE \approx 0.24\text{--}0.41$ (Confirmed parameter-dependent).
* **RAG Retrieval:** $96.97\%$ Hit@1, $100\%$ Hit@3, $0.9848$ MRR, $21.84\text{ ms}$ latency.
* **Advisory Synthesis:** $100\%$ groundedness, $100\%$ citation fidelity, $100\%$ guardrail abstention on the 15-case benchmark.
* **End-to-End Pipeline:** $100\%$ workflow execution rate, $184.22\text{ ms}$ mean CPU latency, $100\%$ citation linkage.

---

## 16. Limitations

To maintain academic integrity, the following limitations are explicitly documented:
1. **Controlled vs. Field Imagery:** The model was trained primarily on standardized laboratory foliar captures with neutral backgrounds. Performance on real field photographs featuring intense solar glare, motion blur, complex soil/weed backgrounds, or multiple simultaneous infections may degrade. Field validation is required prior to commercial deployment.
2. **Coarse Spatial Resolution of Grad-CAM:** Grad-CAM operates on feature map `features.16` ($7\times 7$ grid). When bilinearly upsampled to $224\times 224$, the resulting heatmaps provide coarse regional localization and cannot pinpoint microscopic fungal spores or tiny bacterial pustules.
3. **Grad-CAM Does Not Prove Biological Causality:** Heatmap salience indicates regions that contributed to the model's mathematical score; it does not constitute biological proof of active pathogen colonization.
4. **Closed Crop Scope:** The current system supports strictly 3 crops and 11 classes. Uploading images of Wheat, Cotton, Sugarcane, Citrus, or Tomato will produce invalid predictions unless filtered out by the user or accompanied by out-of-scope text queries.
5. **Softmax Overconfidence on OOD Data:** While the model is calibrated on in-distribution test data ($\text{ECE} = 1.40\%$), deep neural networks can emit high-confidence predictions on completely out-of-distribution imagery (e.g. non-leaf objects).
6. **Linguistic Scope:** Urdu translation relies on a curated glossary and rule-based entity preservation; it does not currently parse regional Pashto dialects commonly spoken in rural KP.

---

## 17. Ethical and Safety Considerations

1. **Non-Replacement of Human Experts:** Chemical pesticides carry severe environmental and human toxicity risks. Misapplying fungicides based on automated misclassification could cause crop burning or water contamination. The application prominently displays disclaimers requiring field confirmation with local extension agents before purchasing chemicals.
2. **Preventing Chemical Hallucinations:** Generative models often hallucinate lethal pesticide concoctions. KP Smart Crop Doctor completely eliminates generative chemical invention by restricting recommendations to direct extractions from government-approved bulletins.
3. **Data Privacy:** All inference and retrieval operations run locally on CPU; no farmer images, GPS coordinates, or personal metadata are transmitted to third-party commercial cloud APIs.

---

## 18. Future Work

1. **Field Domain Adaptation:** Collect a localized field dataset across Swat, Mardan, and Abbottabad to fine-tune the vision backbone against real-world background clutter and variable lighting.
2. **Multi-Crop Expansion:** Systematically curate training data and extension bulletins for Wheat (Yellow Rust), Sugarcane (Red Rot), and Citrus (Canker).
3. **Pashto Localization:** Expand the translation engine and voice synthesizer to support Pashto, enabling accessibility for non-literate farmers in rural KP.
4. **Edge Deployment:** Export the MobileNetV3-Large model and FAISS vector index to ONNX Runtime / TFLite for zero-connectivity offline operation on Android smartphones.

---

## 19. Likely Viva Questions and Concise Answers

### Q1: Why did you choose MobileNetV3-Large over EfficientNet-B0 or ResNet-50?
> **Answer:** MobileNetV3-Large was selected through controlled experimentation using our predefined metric (Validation Macro-F1). It achieved higher validation Macro-F1 ($0.9773$ vs $0.9535$) and higher test Macro-F1 ($0.9657$ vs $0.9352$) than EfficientNet-B0, while running $40.9\%$ faster on CPU ($28.38\text{ ms}$ vs $48.01\text{ ms}$) and exhibiting superior probability calibration ($\text{ECE} = 1.40\%$ vs $5.89\%$). ResNet-50 was excluded due to its excessive parameter size ($>25\text{M}$ params), which is ill-suited for rural edge deployment.

### Q2: How did you ensure your dataset splits did not suffer from data leakage?
> **Answer:** We implemented group-stratified splitting based on unique leaf identities (`leaf_id`) and perceptual difference hashing (`dHash` with Hamming distance $\le 4$). This ensured that multiple photographs of the same leaf or near-identical camera shots were strictly constrained to the same split, completely preventing test-set contamination.

### Q3: Why is Validation Macro-F1 preferred over Test Accuracy for model selection?
> **Answer:** Test accuracy is biased toward majority classes in imbalanced datasets and evaluating test data during model selection causes data snooping/leakage. Validation Macro-F1 assigns equal weight to every disease class (including under-represented classes like Potato Healthy with only 23 test samples), ensuring balanced diagnostic sensitivity before the final test set is revealed.

### Q4: Does Grad-CAM prove that the leaf has the disease?
> **Answer:** No. Grad-CAM visualizes the spatial regions in the convolutional feature map that contributed positively to the network's mathematical logit score. While it verifies whether the model focused on foliar lesions rather than background artifacts, it does not constitute biological or microbiological proof of pathogen presence.

### Q5: How did you verify that Grad-CAM was not acting as a generic edge detector?
> **Answer:** We conducted the Adebayo et al. (NeurIPS 2018) model randomization sanity check. When the weights of target layer `features.16` were replaced with random Gaussian noise, the Pearson correlation between trained and randomized heatmaps dropped to $0.0000$, and Mean Absolute Error increased significantly ($0.24\text{--}0.41$), demonstrating that the heatmaps strictly depend on learned representations.

### Q6: How does your RAG system prevent hallucination of pesticide recommendations?
> **Answer:** We utilized a `LocalExtractiveSynthesizer` that operates directly on retrieved chunks from our verified FAISS index. The system does not allow an unconstrained generative LLM to invent chemical dosages or trade names; every management action is directly extracted from institutional bulletins (ARI Tarnab, CCRI Pirsabak, PRC Abbottabad) and tied to an immutable provenance badge.

### Q7: What happens if a farmer uploads an image of a Cotton leaf or asks about Wheat rust?
> **Answer:** The system incorporates a multi-layer guardrail. In text queries, lexical and semantic classifiers detect out-of-scope crops and trigger an immediate abstention response (`status="information_unavailable"`), explaining that the crop is outside the 11 supported classes and directing the user to a local extension officer. In vision inference, the UI displays Top-2 comparative bars and calibration warnings, reminding users of the 3-crop operational envelope.

### Q8: Why did you use FAISS IndexFlatIP instead of IndexIVFFlat or HNSW?
> **Answer:** Our curated knowledge base comprises 164 high-quality chunks from 11 authoritative bulletins. For corpora with fewer than $100,000$ vectors, `IndexFlatIP` (exact brute-force cosine search) executes in under $16\text{ ms}$ on CPU while providing $100\%$ recall without the indexing overhead, parameter tuning, or approximate loss of IVF or graph-based indices.

### Q9: Why is the inference latency so low on CPU?
> **Answer:** MobileNetV3-Large utilizes hard-swish activation functions, depthwise separable convolutions, and squeeze-and-excitation modules optimized for CPU vector instructions. Coupled with offline embedding caching and lightweight FAISS search, the entire end-to-end pipeline (classification, Grad-CAM, retrieval, advisory, and Urdu rendering) completes in an average of $184.22\text{ ms}$.

### Q10: What are the main sources of error in your model?
> **Answer:** Out of 1,377 test images, exactly 34 were misclassified ($2.47\%$). Analysis revealed zero cross-crop errors. $67.6\%$ of all errors ($23 / 34$) occurred between Maize Gray Leaf Spot and Maize Northern Leaf Blight due to their shared morphological appearance as elongated foliar necrotic blights in early disease stages.
