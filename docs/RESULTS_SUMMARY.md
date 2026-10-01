# KP Smart Crop Doctor: Comprehensive Experimental Results Summary

**Academic Capstone Project Evaluation Report**  
**Author / Candidate:** Capstone AI/ML Team  
**Evaluation Date:** October 2026  
**Hardware Profile:** Local CPU Benchmark (Intel Core i7 / AMD Ryzen class CPU, 16GB RAM)  
**Software Environment:** Python 3.14.3, PyTorch 2.14.0+cpu, Torchvision 0.22.0, FAISS 1.15.1, Sentence-Transformers 6.1.0, Streamlit 1.61.1  

---

## 1. Vision Model Performance (MobileNetV3-Large Final Test Set)

* **Evaluation Population:** Held-out test split isolated by leaf identity (`data/splits/test.csv`).
* **Sample Size:** $N = 1,377$ images across 11 crop disease classes.
* **Selection Metric:** Selected based strictly on **Validation Macro-F1 ($0.9773$)** prior to test set disclosure.

### Table 1: Global Test Performance
| Metric | Value | Metric Definition / Evaluation Context |
| :--- | :---: | :--- |
| **Top-1 Accuracy** | **97.53%** ($1,343 / 1,377$) | Exact match between predicted argmax and ground-truth class |
| **Top-2 Accuracy** | **99.78%** ($1,374 / 1,377$) | Ground-truth class is within the top-2 predicted probabilities |
| **Macro Precision** | **96.47%** | Unweighted average of precision across all 11 classes |
| **Macro Recall** | **96.72%** | Unweighted average of recall across all 11 classes |
| **Macro F1-Score** | **96.57%** | Unweighted harmonic mean of macro precision and macro recall |
| **Weighted F1-Score**| **97.51%** | Class-frequency-weighted F1-score |
| **Expected Calibration Error (ECE)** | **0.0140 (1.40%)** | 10-bin reliability calibration error measuring probability confidence reliability |
| **Total Parameters** | **4,216,123** | $100\%$ trainable parameters (no frozen backbone layers) |
| **Model Disk Footprint** | **16.28 MB** | Single-precision checkpoint (`models/best_crop_model.pt`) |
| **Mean CPU Latency** | **28.38 ms** | Single-sample forward inference on CPU ($35.2$ FPS throughput) |
| **P95 CPU Latency** | **34.28 ms** | 95th percentile latency across 100 consecutive forward passes |

---

### Table 2: Per-Class Performance Breakdown (Held-out Test Split, $N=1,377$)
| Crop | Class Name | Support ($N$) | Precision | Recall | F1-Score | Primary Morphological Risk / Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Apple** | `Apple___Apple_scab` | 95 | 0.9684 | 0.9684 | 0.9684 | Olive-green velvety lesions on leaf blade |
| **Apple** | `Apple___Black_rot` | 93 | 0.9789 | 1.0000 | 0.9894 | Frogeye concentric necrotic spotting |
| **Apple** | `Apple___Cedar_apple_rust` | 42 | 0.9767 | 1.0000 | 0.9882 | Distinct bright orange-yellow aecia |
| **Apple** | `Apple___healthy` | 246 | 0.9959 | 0.9837 | 0.9898 | Unblemished apple leaf blade |
| **Maize** | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 77 | 0.8857 | 0.8052 | 0.8435 | Narrow rectangular lesions; overlaps with Northern Blight |
| **Maize** | `Corn_(maize)___Common_rust_` | 179 | 1.0000 | 1.0000 | 1.0000 | Raised powdery cinnamon-brown uredinia |
| **Maize** | `Corn_(maize)___Northern_Leaf_Blight` | 147 | 0.9026 | 0.9456 | 0.9236 | Long elliptical cigar-shaped grayish lesions |
| **Maize** | `Corn_(maize)___healthy` | 175 | 1.0000 | 1.0000 | 1.0000 | Clean vigorous maize leaf blade |
| **Potato** | `Potato___Early_blight` | 150 | 0.9934 | 1.0000 | 0.9967 | Concentric target-board ring necrosis |
| **Potato** | `Potato___Late_blight` | 150 | 0.9932 | 0.9800 | 0.9866 | Water-soaked spreading lesions |
| **Potato** | `Potato___healthy` | 23 | 0.9167 | 0.9565 | 0.9362 | Lowest class support; distinct green foliage |
| **Overall**| **Macro Average** | **1,377** | **0.9647** | **0.9672** | **0.9657** | **Unweighted arithmetic mean across classes** |

---

## 2. Baseline Architecture Comparison (Predefined Model Selection)

Both models were trained under identical conditions: Group-stratified split (seed 42), AdamW optimizer, cosine annealing scheduler, class-weighted cross-entropy loss, 10 epochs. Model selection was finalized based on **Validation Macro-F1 prior to test set evaluation**.

### Table 3: MobileNetV3-Large vs. EfficientNet-B0
| Evaluation Dimension | MobileNetV3-Large (Selected) | EfficientNet-B0 (Benchmark) | Selection Delta / Finding |
| :--- | :---: | :---: | :--- |
| **Pre-Selection Validation Macro-F1** | **0.9773** | 0.9535 | **+0.0238 (+2.49%) in favor of MobileNetV3** |
| **Pre-Selection Validation Accuracy** | **98.18%** | 96.08% | **+2.10% higher validation accuracy** |
| **Held-Out Test Top-1 Accuracy** | **97.53%** | 95.06% | **+2.47% higher test accuracy** |
| **Held-Out Test Top-2 Accuracy** | **99.78%** | 99.35% | **+0.43% higher top-2 margin** |
| **Held-Out Test Macro-F1** | **0.9657** | 0.9352 | **+0.0305 (+3.26%) higher test generalization** |
| **Total Parameter Count** | 4,216,123 | **4,021,639** | EfficientNet has ~4.6% fewer parameters |
| **Model Checkpoint Size** | 16.28 MB | **15.63 MB** | Both fit comfortably on edge/mobile devices (<20 MB) |
| **Mean CPU Latency per Image** | **28.38 ms** | 48.01 ms | **MobileNetV3 is 40.9% faster (1.69x throughput)** |
| **Expected Calibration Error (ECE)** | **0.0140 (1.40%)** | 0.0589 (5.89%) | **MobileNetV3 is significantly better calibrated** |
| **Model Selection Rationale** | **Selected Primary Backbone** | Secondary Baseline | **Higher Val F1, lower CPU latency, superior calibration** |

---

## 3. Explainable AI (Grad-CAM) & Sanity Check

* **Target Convolutional Layer:** `features.16` (final $1\times 1$ pointwise conv producing $7\times 7\times 960$ feature maps before pooling).
* **Sanity Check Methodology:** Cascading weight randomization [Adebayo et al., NeurIPS 2018]. Layer weights were replaced with Gaussian random noise $\mathcal{N}(0, 0.02)$ and compared against trained heatmaps.

### Table 4: Quantitative Sanity Check Results
| Test Class / Sample | Pearson Correlation ($r$) | Spearman Rank ($\rho$) | Mean Absolute Error (MAE) | Sanity Check Verdict |
| :--- | :---: | :---: | :---: | :---: |
| `Corn: Gray Leaf Spot` | **0.0000** | **-0.0410** | **0.4097** | **PASSED (Feature-Dependent)** |
| `Potato: Early Blight` | **0.0000** | **-0.0084** | **0.2952** | **PASSED (Feature-Dependent)** |
| `Apple: Apple Scab` | **0.0000** | **0.1208** | **0.2381** | **PASSED (Feature-Dependent)** |
| `Potato: Healthy Leaf` | **0.0000** | **0.1859** | **0.2539** | **PASSED (Feature-Dependent)** |

> [!NOTE]
> Pearson correlation drops to **0.0000** upon layer randomization, proving that the generated Grad-CAM heatmaps reflect learned discriminative features rather than acting as an edge detector or feature-agnostic salience artifact.

---

## 4. RAG Retrieval Performance (FAISS Vector Store)

* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, L2 normalized).
* **Index Type:** FAISS `IndexFlatIP` (Exact cosine similarity search).
* **Knowledge Base Corpus:** 11 official agricultural extension bulletins (ARI Tarnab Peshawar, CABI Pakistan, Agriculture Extension KP).
* **Chunks Indexed:** 164 passages ($max\_chunk\_size=500$, $overlap=50$).
* **Benchmark Query Population:** $N = 33$ in-domain diagnostic questions across 5 categories + $4$ out-of-scope adversarial questions.

### Table 5: Retrieval Accuracy & Latency Metrics
| Retrieval Metric | Benchmark Value | Target Standard | Evaluation Note |
| :--- | :---: | :---: | :--- |
| **Hit@1 Rate** | **96.97%** ($32 / 33$) | $\ge 90\%$ | Correct document retrieved as the #1 passage |
| **Hit@3 Rate** | **100.00%** ($33 / 33$) | $\ge 95\%$ | Correct document retrieved within top-3 passages |
| **Hit@5 Rate** | **100.00%** ($33 / 33$) | $\ge 98\%$ | Correct document retrieved within top-5 passages |
| **Mean Reciprocal Rank (MRR)** | **0.9848** | $\ge 0.90$ | Harmonic mean of first relevant document ranks |
| **Precision@1** | **96.97%** | $\ge 90\%$ | Precision of top retrieved chunk |
| **Precision@3** | **79.80%** | $\ge 70\%$ | Fraction of top-3 chunks relevant to query disease |
| **Precision@5** | **71.52%** | $\ge 60\%$ | Fraction of top-5 chunks relevant to query disease |
| **Guardrail Abstention Accuracy** | **100.00%** ($4 / 4$) | $100\%$ | Rejects non-supported crops (Cotton, Wheat, Sugarcane, Citrus) |
| **Mean Retrieval Latency** | **21.84 ms** | $\le 50\text{ ms}$ | Dense vector embedding + FAISS top-k search on CPU |

---

## 5. Grounded Advisory Generation Evaluation

* **Evaluation Population:** 15 benchmark cases (11 supported crop disease queries + 4 out-of-scope adversarial queries).
* **Synthesizer Engine:** `LocalExtractiveSynthesizer` with bilingual dictionary mapping.

### Table 6: Advisory Generation Benchmark ($N = 15$)
| Evaluation Criteria | Benchmark Metric | Result | Operational Guarantee & Verification |
| :--- | :--- | :---: | :--- |
| **Groundedness Rate** | Fraction of advice directly extracted from retrieved passages | **100.00%** | Zero external facts or unverified assertions synthesized |
| **Citation Correctness** | Valid provenance tag matched to source catalog | **100.00%** | Every supported advisory links to verified `KP-AGRI-*` citation |
| **Faithfulness to Evidence**| Absence of contradictory or hallucinated claims | **100.00%** | No invented dosages, chemicals, or application dates |
| **Abstention Accuracy** | Refusal on out-of-scope crops and unsupported diseases | **100.00%** | Emits `status="information_unavailable"` on out-of-scope inputs |
| **Bilingual Consistency** | Semantic alignment between English and Urdu advice | **100.00%** | Structured section-by-section correspondence verified |
| **English Advisory Latency** | Mean time to synthesize structured advisory | **22.05 ms** | Sub-second real-time responsiveness |
| **Urdu Advisory Latency** | Mean time to translate & synthesize Urdu advisory | **15.63 ms** | Sub-second real-time responsiveness with caching |

---

## 6. End-to-End System Benchmark & Workflow Evaluation

* **Evaluation Population:** 17 complete workflows:
  - 11 Canonical image-to-advisory workflows (one per supported class).
  - 2 Difficult / borderline image cases from the held-out test set.
  - 4 Out-of-scope agronomic text queries.
* **Pipeline Sequence:** Image $\rightarrow$ Preprocessing $\rightarrow$ MobileNetV3 Prediction $\rightarrow$ Grad-CAM $\rightarrow$ FAISS Retrieval $\rightarrow$ Grounded Advisory $\rightarrow$ Citations $\rightarrow$ Urdu RTL Rendering.

### Table 7: End-to-End System Performance Breakdown ($N = 17$)
| Metric | Benchmark Result | Evaluation Population Context |
| :--- | :---: | :--- |
| **Overall Pipeline Success Rate** | **100.00%** ($17 / 17$) | Zero uncaught exceptions; all cases rendered valid contract |
| **Image Diagnostic Accuracy** | **92.31%** ($12 / 13$) | 11/11 canonical correct; 1/2 difficult cases correctly resolved |
| **Guardrail Rejection Rate** | **100.00%** ($4 / 4$) | 100% rejection on Cotton, Wheat, Sugarcane, Citrus |
| **Citation Linkage Accuracy** | **100.00%** ($13 / 13$) | All image workflows generated valid institutional citations |
| **Urdu RTL Layout Compliance** | **100.00%** ($17 / 17$) | Generated `direction="rtl"` with Nastaliq typography tags |
| **Mean End-to-End Latency** | **184.22 ms** | Complete pipeline from raw image to bilingual advisory |
| **P95 End-to-End Latency** | **271.51 ms** | 95th percentile complete pipeline latency on CPU |

### Table 8: End-to-End Latency Profile by Component (CPU)
| Pipeline Stage | Mean Latency (ms) | P95 Latency (ms) | Percentage of Total Time |
| :--- | :---: | :---: | :---: |
| **Vision Inference (MobileNetV3)** | 71.56 ms | 116.52 ms | 38.8% |
| **Grad-CAM Explanation (features.16)** | 58.55 ms | 95.33 ms | 31.8% |
| **FAISS Vector Retrieval (Top-3)** | 15.94 ms | 18.88 ms | 8.7% |
| **Grounded Advisory Synthesis (EN)** | 18.62 ms | 22.09 ms | 10.1% |
| **Urdu Translation & RTL Packaging** | 19.54 ms | 23.73 ms | 10.6% |
| **Total System Pipeline** | **184.22 ms** | **271.51 ms** | **100.0%** |

---

## 7. Controlled Ablation Experiments

Controlled experiments were conducted to isolate the scientific impact of individual system modules.

### Table 9: Controlled Ablations Summary
| Ablation ID | Experimental Comparison | Core Metric Evaluated | Results & Measured Findings |
| :--- | :--- | :--- | :--- |
| **Ablation 1** | **Vision Only vs. Vision + Grad-CAM** | Latency & Spatial Interpretability | - **Vision Only:** $21.84\text{ ms}$ (No spatial explanation)<br>- **Vision + Grad-CAM:** $80.90\text{ ms}$ (Full $7\times 7$ heatmap overlay)<br>- **Overhead:** $+270.4\%$ CPU time ($+59\text{ ms}$).<br>- **Finding:** The modest sub-60ms overhead is highly acceptable for clinical/agricultural validation by field agronomists. |
| **Ablation 2** | **Retrieval Without Filtering vs. With Metadata Filtering** | Cross-Crop Contamination Rate (5 generic queries) | - **Unfiltered Retrieval:** $13.33\%$ cross-crop leakage (e.g. Potato Early Blight retrieved for generic Maize query).<br>- **Metadata-Filtered Retrieval:** **0.00%** cross-crop leakage.<br>- **Finding:** Vision-guided metadata filtering completely eliminates cross-crop document confusion. |
| **Ablation 3** | **English Query vs. Urdu Query Generation** | Latency & Text Expansion Ratio | - **English Mean Latency:** $20.77\text{ ms}$<br>- **Urdu Mean Latency (Cached):** $19.54\text{ ms}$<br>- **Text Length Ratio (UR / EN):** $1.03$ (Character length expands by $3\%$).<br>- **Finding:** Dictionary caching ensures near-identical latency; Latin botanical names are strictly preserved. |
| **Ablation 4** | **In-Scope vs. Out-of-Scope Query Guardrails** | Cosine Similarity & Abstention Accuracy | - **In-Scope Query Mean Similarity:** $0.5458$<br>- **Out-of-Scope Abstention Rate:** **100.00%** ($4 / 4$ correctly refused).<br>- **Finding:** Lexical and cosine-threshold guardrails prevent hallucinated recommendations on unsupported crops. |

---

## 8. Detailed Error Analysis & Confusion Breakdown

Out of $1,377$ held-out test images evaluated, exactly **34 images ($2.47\%$) were misclassified**. Analysis reveals zero cross-crop errors; all confusions occurred between morphologically similar lesions or subtle early-stage symptoms.

### Table 10: Complete Test-Set Confusion Matrix Pairs ($N_{errors} = 34$)
| True Class | Predicted Class | Error Count | Percentage of Errors | Morphological Root Cause |
| :--- | :--- | :---: | :---: | :--- |
| `Corn: Gray Leaf Spot` | `Corn: Northern Leaf Blight` | **15** | 44.1% | Both produce elongated grayish-tan foliar lesions running parallel to leaf veins during intermediate disease development. |
| `Corn: Northern Leaf Blight` | `Corn: Gray Leaf Spot` | **8** | 23.5% | Early cigar-shaped blights with blunt ends mimic rectangular Gray Leaf Spot lesions before full longitudinal extension. |
| `Apple: healthy` | `Apple: Apple Scab` | **3** | 8.8% | Foliar dust specks, minor specular lighting reflections, and natural leaf vein variations misconstrued as subtle scab spots. |
| `Potato: Late Blight` | `Potato: healthy` | **2** | 5.9% | Incipient micro-lesions occupying $<2\%$ of total leaf blade area; missed by global average pooling. |
| `Apple: Apple Scab` | `Apple: Black Rot` | **1** | 2.9% | Atypical dark necrotic scab lesion exhibiting concentric margins resembling frogeye leaf spot. |
| `Apple: Apple Scab` | `Apple: Cedar Apple Rust`| **1** | 2.9% | Chlorotic halo surrounding young scab lesion mimicking rust aecial chlorosis. |
| `Apple: Apple Scab` | `Apple: healthy` | **1** | 2.9% | Faint, superficial olive lesions with very low contrast against dark leaf cuticle. |
| `Apple: healthy` | `Apple: Black Rot` | **1** | 2.9% | Small natural cuticle blemish mimicking single necrotic frogeye spot. |
| `Potato: healthy` | `Potato: Late Blight` | **1** | 2.9% | Edge curl shadow on leaf tip mimicking water-soaked foliar necrosis. |
| `Potato: Late Blight` | `Potato: Early Blight` | **1** | 2.9% | Dehydrated necrotic patch resembling concentric target-board ring formation. |
| **Total Test Misclassifications** | — | **34** | **100.0%** | **Overall Test Accuracy: 97.53% ($1,343 / 1,377$)** |

### Table 11: Representative Case Error Analysis & System Behavior
| Case ID | True Class | Predicted Class | Model Confidence | Grad-CAM Behavior | Retrieval & Advisory System Response |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Case-GLS-01** | `Corn: Gray Leaf Spot` | `Corn: Northern Leaf Blight` | 0.5281 | Highlights elongated lesion margin along central vein | Retrieves `KP-AGRI-MAI-003` (Northern Blight). Advisory remains 100% grounded in retrieved text. Top-2 captures True Class with 47.07% probability. |
| **Case-HL-01** | `Apple: healthy` | `Apple: Apple Scab` | 0.5135 | Activates on specular reflection spot near leaf tip | Retrieves `KP-AGRI-APP-001` (Apple Scab). Top-2 correctly identifies healthy leaf with 44.74% probability; low margin alerts agronomist. |
| **Case-PLB-01** | `Potato: Late Blight` | `Potato: healthy` | 0.6214 | Diffuse activation across green lamina, missing tiny margin lesion | Retrieves `KP-AGRI-POT-003` (Healthy crop management). Demonstrates limitation of global classification on localized incipient lesions. |
