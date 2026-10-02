# KP Smart Crop Doctor: Final Capstone Defense Q&A

**Project:** KP Smart Crop Doctor: An Explainable, Knowledge-Grounded Agricultural Advisory System for Khyber Pakhtunkhwa  
**Purpose:** Comprehensive Oral Defense Reference for Examiners, Technical Panels, and Agronomic Evaluators  
**Date:** October 2026  
**Standards:** All answers are grounded strictly in the project's source code, test suites, and empirical evaluation reports.

---

## Table of Contents
- [Category 1: Machine Learning & Computer Vision](#category-1-machine-learning--computer-vision)
- [Category 2: Explainable AI (Grad-CAM)](#category-2-explainable-ai-grad-cam)
- [Category 3: Retrieval-Augmented Generation (RAG) & Knowledge Base](#category-3-retrieval-augmented-generation-rag--knowledge-base)
- [Category 4: Multilingual & Localization Architecture](#category-4-multilingual--localization-architecture)
- [Category 5: Application Design & Deployment](#category-5-application-design--deployment)
- [Category 6: System Evaluation, Limitations & Field Reality](#category-6-system-evaluation-limitations--field-reality)

---

## Category 1: Machine Learning & Computer Vision

### Q1: Why did you choose MobileNetV3-Large as your primary backbone?
> **Answer:** MobileNetV3-Large was selected through controlled empirical experimentation using our predefined selection criterion: **Validation Macro-F1 prior to test set disclosure**. In Milestone 3, MobileNetV3-Large achieved a Validation Macro-F1 of **$0.9773$** compared to **$0.9535$** for EfficientNet-B0 ($+2.49\%$ advantage). Furthermore, on standard CPU hardware, MobileNetV3-Large exhibited a mean latency of **$28.38\text{ ms}$** ($35.2$ FPS throughput) versus **$48.01\text{ ms}$** for EfficientNet-B0 ($40.9\%$ faster). It also demonstrated significantly superior probability calibration ($\text{ECE} = 1.40\%$ vs $5.89\%$). Its architecture utilizes depthwise separable convolutions, hard-swish non-linearities, and squeeze-and-excitation attention blocks that are exceptionally well-optimized for edge and mobile CPU inference in rural agricultural settings.

### Q2: Why did you not choose ResNet-50 or a Vision Transformer (ViT)?
> **Answer:** ResNet-50 contains over $25.5\text{ million}$ parameters and produces model checkpoints exceeding $100\text{ MB}$, with inference times often exceeding $100\text{ ms}$ on mobile/low-power CPUs. Vision Transformers (such as ViT-B/16) have over $86\text{ million}$ parameters, lack inductive bias for translation invariance on small datasets, and suffer from prohibitive computational overhead without dedicated GPU acceleration. Given that KP agricultural extension officers and farmers operate on low-to-mid-range mobile hardware or budget laptops with intermittent electrical access, a lightweight model ($\approx 4.2\text{M}$ parameters, $16.28\text{ MB}$) is the only academically sound and practical choice.

### Q3: Why did you benchmark against EfficientNet-B0 specifically?
> **Answer:** EfficientNet-B0 is the established academic benchmark for parameter-efficient convolutional networks. It has an almost identical parameter footprint ($4,021,639$ parameters vs. $4,216,123$ for MobileNetV3-Large) and uses compound scaling of depth, width, and resolution. Benchmarking against EfficientNet-B0 ensured an equitable, rigorous scientific comparison between two leading edge vision backbones trained under strictly identical protocols (seed 42, AdamW, cosine annealing, class-weighted cross-entropy, 10 epochs).

### Q4: Why is transfer learning appropriate here instead of training from scratch?
> **Answer:** Training deep convolutional backbones from scratch on small-to-medium agricultural datasets ($9,175$ images) leads to severe overfitting and poor representation learning in early convolutional filters. Pretraining on ImageNet-1K equips the network with generalized low-level visual primitives (Gabor-like edge detectors, texture filters, color gradient representations). Fine-tuning the entire network with a small learning rate ($3 \times 10^{-4}$) allowed the model to rapidly adapt these generalized representations to specialized foliar lesion morphology (concentric target rings, water-soaked margins, pustules) with high sample efficiency.

### Q5: How was data leakage prevented between training and testing sets?
> **Answer:** Data leakage is a critical methodological flaw in naive plant pathology studies where multiple photographic captures of the same physical leaf appear across both training and test splits, artificially inflating accuracy. We prevented this through a 3-step pipeline:
> 1. **Leaf-Group Isolation:** We parsed unique leaf identifiers from image filenames to ensure that all images from the same physical leaf session were grouped into the same cluster.
> 2. **Perceptual Difference Hashing (dHash):** Near-duplicate images taken from slightly altered camera angles were detected using $64$-bit gradient perceptual hashing. Pairs with a Hamming distance $\le 4$ were constrained to identical split buckets.
> 3. **Group Stratification:** Splits were created using `GroupKFold` conditioned on `leaf_id`. We verified mathematically via automated tests (`test_group_stratified_splits_zero_leakage`) that:
>    $$\text{Set}(\text{LeafIDs}_{\text{train}}) \cap \text{Set}(\text{LeafIDs}_{\text{val}}) = \emptyset, \quad \text{Set}(\text{LeafIDs}_{\text{train}}) \cap \text{Set}(\text{LeafIDs}_{\text{test}}) = \emptyset$$

### Q6: Why did you restrict the scope to 11 classes across Maize, Potato, and Apple?
> **Answer:** In Milestone 2, we conducted an objective multi-criteria evaluation of 9 candidate crops against regional relevance to Khyber Pakhtunkhwa, public image availability, class balance, disease severity, and the availability of verified agricultural extension literature from KP institutions. Apple, Maize, and Potato represent the highest-priority food security and cash crops in KP: Apple dominates Swat/Hazara valleys; Maize is the staple cereal in Swabi, Mardan, and Nowshera; and Potato is vital across Kalam and Battagram. Other crops like Wheat, Cotton, and Citrus were evaluated but excluded from the primary vision model because verified, high-resolution regional image sets were inadequate or severely confounded by studio artifacts. Quality and verifiable provenance were prioritized over superficial class counts.

### Q7: What does the confidence percentage actually represent?
> **Answer:** The confidence percentage is the normalized exponential output of the final linear layer (the softmax probability for the winning class):
> $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j} e^{z_j}}$$
> Softmax probabilities represent the relative activation strength among the 11 candidate classes, not an absolute statistical guarantee of ground-truth truth in the real world. To verify whether these numbers reflect true empirical frequencies, we evaluated the **Expected Calibration Error (ECE)** across 10 bins. Our MobileNetV3-Large model achieved an ECE of **$0.0140$ ($1.40\%$)**, indicating strong calibration on in-distribution test data. However, on out-of-distribution imagery, neural networks can emit artificially high confidence, which is why our Streamlit UI prominently displays calibration caveats and Top-2 comparative bars.

### Q8: What happens if an image of an unknown disease or non-leaf object is uploaded?
> **Answer:** A closed-set softmax classifier is mathematically constrained to distribute probability mass across the 11 classes it was trained on ($\sum_{i=1}^{11} P_i = 1$). If an unknown crop (such as Citrus) or a non-leaf object is uploaded, the classifier will still output one of the 11 classes. To mitigate this fundamental limitation:
> 1. We display **Top-2 predictions** side-by-side: out-of-distribution inputs typically exhibit higher entropy across top ranks or low margins.
> 2. The UI explicitly states the 3-crop operational envelope and displays an academic non-replacement disclaimer.
> 3. In the conversational follow-up interface, our **text-based RAG guardrails** intercept unsupported crops and refuse to emit agricultural recommendations.
> 4. Out-of-distribution visual rejection via energy-based scoring or autoencoders is identified as vital future work.

---

## Category 2: Explainable AI (Grad-CAM)

### Q9: What is Grad-CAM and how does it work?
> **Answer:** Gradient-weighted Class Activation Mapping (Grad-CAM) is an explainability technique for convolutional networks [Selvaraju et al., ICCV 2017]. It computes a visual localization map by taking the gradients of the target class score $y^c$ with respect to the activation maps $A^k$ of the final convolutional layer. These gradients are global-average-pooled to obtain importance weights $\alpha_k^c$:
> $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
> A weighted linear combination of the feature maps is computed and passed through a $\operatorname{ReLU}$ non-linearity to isolate features that have a positive influence on the target class:
> $$L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
> The resulting 2D activation map is bilinearly upsampled to the input image dimensions and blended as a colormap overlay.

### Q10: Why did you attach the Grad-CAM hooks to `features.16`?
> **Answer:** In the PyTorch MobileNetV3-Large architecture, `features.16` is the final convolutional block (a $1\times 1$ pointwise convolution following the last inverted residual block) before the adaptive average pooling layer and classifier head. This layer retains the highest-level spatial representations ($7\times 7$ feature grid with $960$ channels) where semantic concepts (lesion borders, necrotic centers, pustules) are fully synthesized. Earlier layers retain high spatial resolution but lack disease-level semantic abstraction, while subsequent layers discard spatial dimensions entirely through global pooling.

### Q11: How was your Grad-CAM implementation scientifically validated?
> **Answer:** We conducted the cascading parameter randomization sanity check established by Adebayo et al. (NeurIPS 2018). Many saliency methods act as parameter-agnostic edge detectors, producing identical heatmaps regardless of whether network weights are trained or random. To test this, we replaced the weights of `features.16` with random Gaussian noise $\mathcal{N}(0, 0.02)$ and measured the similarity between trained and randomized heatmaps across multiple classes. In all cases, the Pearson correlation coefficient dropped from $1.0000$ to **$0.0000$** ($MAE \approx 0.24\text{--}0.41$). This mathematically proved that our Grad-CAM heatmaps are strictly dependent on learned network weights and reflect true learned representations.

### Q12: What does the randomization sanity check demonstrate to an examiner?
> **Answer:** It demonstrates that the visual explanations are not spurious artifacts of bilinear upsampling, input image gradients, or generic edge detection. If a saliency map does not change when the model weights are scrambled, it cannot explain what the model learned. By proving that correlation collapses to zero upon weight randomization, we confirmed that our explanations genuinely reflect the model's trained parameters.

### Q13: What are the fundamental limitations of interpreting Grad-CAM heatmaps?
> **Answer:** Grad-CAM has three critical limitations:
> 1. **Coarse Resolution:** Operating on a $7\times 7$ feature map means upsampled heatmaps provide regional localization rather than pixel-level segmentation. It cannot resolve individual fungal spores or microscopic bacterial lesions.
> 2. **Saliency $\neq$ Biological Correctness:** Grad-CAM highlights pixels that contributed positively to the mathematical logit score. It does not provide biological or microbiological proof that the tissue is colonized by a specific pathogen.
> 3. **Diffuse Activations on Controls:** On healthy foliage, where no discrete lesion exists, activations distribute diffusely across the lamina or focus on prominent leaf veins, which non-expert users may misinterpret as disease focal points.

---

## Category 3: Retrieval-Augmented Generation (RAG) & Knowledge Base

### Q14: Why did you use RAG instead of directly prompting a Large Language Model (LLM)?
> **Answer:** Unconstrained generative LLMs (such as GPT-4, LLaMA, or Claude) exhibit well-documented hallucination rates in specialized domains. In agronomy, hallucinating a pesticide dosage, recommending a chemical banned in Pakistan, or inventing an application schedule can cause catastrophic crop loss, environmental contamination, or farmer poisoning. RAG enforces a deterministic, evidence-grounded architecture where the generation engine is strictly constrained to synthesize answers from pre-verified, local institutional bulletins, linking every recommendation to an immutable provenance badge.

### Q15: Why did you select FAISS IndexFlatIP for your vector database?
> **Answer:** Our curated agricultural corpus comprises 164 high-quality text chunks extracted from 11 verified provincial bulletins. For corpora with fewer than $100,000$ vectors, FAISS `IndexFlatIP` (exact brute-force inner product search on normalized vectors, equivalent to exact cosine similarity) executes in under **$16\text{ ms}$ on CPU** and guarantees **$100\%$ retrieval recall**. Approximate nearest neighbor methods (such as IVF clusters or HNSW graphs) introduce quantization error, hyperparameter complexity, and indexing overhead without any perceptible latency benefit at this corpus scale.

### Q16: Why did you select `sentence-transformers/all-MiniLM-L6-v2` as your embedding model?
> **Answer:** `all-MiniLM-L6-v2` maps sentences to a compact 384-dimensional dense vector space. It has only $22.7\text{ million}$ parameters ($80\text{ MB}$ footprint) and encodes a text query in under **$10\text{ ms}$ on CPU**. In our retrieval evaluation across 33 domain-specific agricultural queries, it achieved a **$96.97\%$ Hit@1 rate**, a **$100.0\%$ Hit@3 rate**, and a Mean Reciprocal Rank (MRR) of **$0.9848$**, proving that larger 768- or 1024-dimensional embedding models were unnecessary and would needlessly increase latency.

### Q17: Why did you use cosine similarity rather than Euclidean ($L_2$) distance?
> **Answer:** Cosine similarity measures the angular orientation between two vectors rather than their magnitude:
> $$\text{Cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
> Text chunks vary in length. Two passages discussing "apple scab ascospores" may differ in token count, which increases Euclidean distance. By normalizing embeddings to unit length ($\|\mathbf{u}\| = 1$), the inner product $\mathbf{u} \cdot \mathbf{v}$ directly computes cosine similarity, making semantic comparison invariant to chunk text length.

### Q18: How was your knowledge base chunked and constructed?
> **Answer:** Documents were segmented using structural paragraph chunking with a maximum chunk size of $500$ characters and an overlap of $50$ characters. Every chunk automatically inherited contextual metadata: document ID, document title, crop, disease class, section category (e.g., `symptoms`, `treatment_control`, `integrated_disease_management`), and authoring institution. This structural chunking preserved heading context and ensured that retrieved passages remained coherent when synthesized into advisory cards.

### Q19: How is source provenance preserved in the final output?
> **Answer:** Every document in `knowledge_base/raw_documents/` has a verified record in `knowledge_base/metadata/provenance_catalog.json`. When a chunk is retrieved by FAISS, its `doc_id` is linked to its institutional metadata (title, authoring institute, publication date, official URL). The synthesizer formats this into an immutable citation string (e.g., `[KP-AGRI-APP-001] Management of Apple Scab in KP. ARI Tarnab (2023-04)`), which is rendered as a prominent provenance badge on the Streamlit interface.

### Q20: How does the system handle unsupported or out-of-scope agricultural questions?
> **Answer:** We implemented a two-tier guardrail mechanism in `AgriculturalRetriever`:
> 1. **Lexical Whitelist/Blacklist Filter:** Queries are scanned for unsupported crop names (e.g., Cotton, Wheat, Sugarcane, Citrus, Rice, Mango, Tomato). If an unsupported crop is detected without supported context, the retriever immediately aborts retrieval.
> 2. **Cosine Similarity Threshold:** If a query does not trigger lexical filters but produces a top retrieval similarity score below the calibrated threshold ($\tau = 0.45$), it is flagged as out-of-distribution.
> In both cases, the retriever returns `status="information_unavailable"`. The advisory generator intercepts this status, emits zero citations, displays a refusal message in English or Urdu, and directs the farmer to contact their local KP Agriculture Extension officer.

### Q21: Why did you use a deterministic extractive synthesizer rather than calling an external LLM API?
> **Answer:** Using an external LLM API (such as OpenAI GPT-4 or Anthropic Claude) introduces four major failure points in a university capstone designed for rural Pakistan:
> 1. **Hallucination Risk:** Even with strict system prompts, generative LLMs occasionally paraphrase dosages, invent brand names, or drop safety intervals.
> 2. **API Dependency & Cost:** External APIs require paid subscriptions, API keys, and constant high-speed internet connectivity, which is unavailable in remote KP valleys.
> 3. **Latency:** Cloud API calls introduce $1\text{--}4\text{ seconds}$ of network latency compared to our local extractive engine, which executes in **$18.62\text{ ms}$**.
> 4. **Reproducibility:** Proprietary cloud models undergo silent backend updates, preventing scientific reproducibility over time.

---

## Category 4: Multilingual & Localization Architecture

### Q22: How does the system process Urdu user queries?
> **Answer:** When an Urdu query is submitted (detected via Unicode Arabic/Urdu script range `\u0600-\u06FF`), the query is processed by `AgriculturalTranslator`:
> 1. It extracts protected entities (pathogens, chemical names, institutional abbreviations).
> 2. It translates common agronomic terms into English using our bilingual agricultural glossary.
> 3. The translated query is combined with vision-predicted metadata (e.g., `Apple Apple_scab: [translated query]`) to search the English FAISS vector index.
> 4. Retrieved evidence is then synthesized and localized back into fluent Urdu for display.

### Q23: How is Right-to-Left (RTL) layout handled in the user interface?
> **Answer:** Standard web frameworks often render Urdu text Left-to-Right (LTR), resulting in reversed punctuation, broken font rendering, and disjointed reading flow. In `src/app/ui_components.py`, we inject custom CSS that inspects the active language. When Urdu is selected, containers apply:
> ```css
> direction: rtl;
> text-align: right;
> font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
> line-height: 1.8;
> ```
> Punctuation marks, bullet points, and tabbed cards align to the right, providing a native Nastaliq reading experience.

### Q24: How are Latin binomials and chemical names protected during Urdu translation?
> **Answer:** Scientific pathogen nomenclature (such as *Venturia inaequalis*, *Phytophthora infestans*) and standardized active ingredients (such as Mancozeb, Copper Oxychloride, Metalaxyl) must not be corrupted by phonetic machine translation. In `src/rag/translator.py`, regex patterns identify Latin binomials and bracketed citation identifiers (`[KP-AGRI-*]`), replacing them with temporary unique tokens (e.g., `__PROTECTED_ENTITY_0__`) before translation and restoring them exactly post-translation.

### Q25: What is the agricultural glossary and why was it created?
> **Answer:** Generic machine translation services often mistranslate agricultural terminology (e.g., translating "blight" literally as "destruction" or "scab" as "human skin crust"). We curated a 33-term domain-specific bilingual glossary (`knowledge_base/metadata/agricultural_glossary_ur.json`) in consultation with regional agronomic literature, mapping terms to their recognized Pakistani agricultural extension counterparts (e.g., Late Blight $\rightarrow$ پچھیتی جھلساؤ, Early Blight $\rightarrow$ اگیتی جھلساؤ, Apple Scab $\rightarrow$ سیب کا پپڑی روگ, Common Rust $\rightarrow$ عام زنگ, Fungicide $\rightarrow$ پھپھوند کش زہر).

---

## Category 5: Application Design & Deployment

### Q26: Why was Streamlit chosen over React, Flutter, or FastAPI?
> **Answer:** Streamlit provides a pure Python full-stack reactive framework ideal for machine learning capstones. It eliminates the architectural overhead of separate frontend/backend deployments while enabling direct, zero-overhead memory access to loaded PyTorch models and FAISS indices. It natively supports custom CSS injection for RTL layout, interactive sliders for real-time Grad-CAM opacity adjustment, and reactive file upload handling with minimal boilerplate.

### Q27: How does model and vector store caching work in Streamlit?
> **Answer:** Machine learning models and transformer tokenizers require significant initialization time ($1\text{--}3\text{ seconds}$). In `app.py`, heavy resources are decorated with `@st.cache_resource`:
> ```python
> @st.cache_resource
> def load_vision_model(): ...
> @st.cache_resource
> def load_rag_pipeline(): ...
> ```
> Streamlit executes the loader function once on startup and stores the instantiated objects in shared memory across user sessions and UI reruns. Consequently, subsequent page interactions execute in sub-second time without reallocating neural network weights.

### Q28: Can this system operate completely offline in remote KP farming areas?
> **Answer:** Yes. The entire pipeline was designed with offline autonomy as a core requirement:
> 1. At the top of `app.py`, offline HuggingFace environment variables are explicitly enforced (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), preventing blocking network HEAD requests.
> 2. The vision model weights (`models/best_crop_model.pt`), FAISS vector index (`faiss_index.bin`), and translation cache (`translation_cache.json`) reside entirely on local disk.
> 3. The system requires zero internet connectivity to perform end-to-end classification, Grad-CAM generation, FAISS retrieval, and bilingual advisory synthesis.

### Q29: What happens when the model's confidence is low?
> **Answer:** In `src/app/ui_components.py`, when the predicted Top-1 confidence falls below the calibrated threshold ($\tau = 0.60$), the UI dynamically flags a warning badge: *"Low Confidence / Diagnostic Ambiguity"*. It presents the Top-2 comparative bars side-by-side with equal prominence, advises the user to inspect the Grad-CAM heatmap to see if the network focused on non-leaf background artifacts, and urges in-person confirmation with a local extension agent before applying chemical sprays.

### Q30: What happens if a user uploads an image of an unsupported crop?
> **Answer:** If a user uploads an unsupported crop (e.g., Cotton leaf with curl virus), the vision model will still output one of the 11 classes due to closed-set softmax constraints. However:
> 1. The UI displays an explicit scope disclaimer listing the 3 supported crops (Apple, Maize, Potato).
> 2. If the user types a question regarding the crop in the follow-up Q&A ("How to treat cotton leaf curl?"), the RAG guardrail detects the out-of-scope crop, aborts retrieval, and outputs `status="information_unavailable"`.
> 3. The user is explicitly told that the knowledge base does not cover that crop, preventing dangerous ungrounded chemical advice.

---

## Category 6: System Evaluation, Limitations & Field Reality

### Q31: What does the $97.53\%$ test accuracy actually mean?
> **Answer:** It means that on our held-out test split of $1,377$ images isolated by unique physical leaf identity and perceptual hashing, the model correctly predicted the ground-truth disease class in $1,343$ instances. It reflects the model's mathematical discriminative capacity on controlled foliar imagery under standardized laboratory lighting. It **does NOT mean** the model will achieve $97.53\%$ accuracy on raw, uncurated smartphone photographs taken under harsh midday sun in a rural field in Swat.

### Q32: Why can benchmark accuracy not guarantee real-world field accuracy in KP?
> **Answer:** Controlled laboratory benchmarks do not capture the severe domain shifts encountered in real-world fields:
> 1. **Lighting & Glare:** Intense specular sunlight reflections, deep canopy shadows, and wet leaf cuticles after rainfall.
> 2. **Background Clutter:** Heterogeneous soil, weeds, irrigation pipes, human fingers holding the leaf, and neighboring crop canopies.
> 3. **Co-infections & Multiple Stresses:** Real plants frequently suffer from simultaneous nutrient deficiencies (nitrogen chlorosis), pest infestations (aphids), and fungal pathogens on the same leaf.
> 4. **Camera Sensor Variations:** Differences in focal depth, motion blur, and color balance across budget Android devices.
> Stating that a lab-evaluated model will achieve identical field accuracy is academically dishonest. Prospective field validation across KP farming clusters is essential future work.

### Q33: What were the major failure modes identified in your error analysis?
> **Answer:** Out of $1,377$ test images, exactly 34 were misclassified ($2.47\%$). Analysis revealed **zero cross-crop errors** (no Maize was ever classified as Apple or Potato). The errors followed three distinct morphological patterns:
> 1. **Maize Foliar Blight Ambiguity (67.6% of all errors; 23 cases):** Maize Gray Leaf Spot and Maize Northern Leaf Blight both produce elongated, grayish-tan necrotic lesions running parallel to leaf veins during early-to-intermediate stages before full characteristic rectangular or cigar shapes develop.
> 2. **Apple Foliar Glare vs. Scab (11.8% of errors; 4 cases):** Healthy apple leaves with slight specular lighting glare or surface dust specks were mistaken for subtle, early-stage olive scab spots.
> 3. **Incipient Potato Late Blight (8.8% of errors; 3 cases):** Tiny water-soaked lesions occupying less than $2\%$ of the total leaf blade margin were missed by global average pooling, resulting in a healthy prediction.

### Q34: What is the single strongest contribution of this capstone project?
> **Answer:** The strongest contribution is the **cohesive integration of spatial explainability and verifiable, non-hallucinatory knowledge grounding into a sub-200ms bilingual edge pipeline**. Rather than treating deep learning as an isolated black box or relying on generic generative LLMs that hallucinate pesticide advice, KP Smart Crop Doctor establishes a complete chain of trust:
> $$\text{Image} \rightarrow \text{Top-2 Prediction} \rightarrow \text{Grad-CAM Localization} \rightarrow \text{FAISS Retrieval} \rightarrow \text{Institutional Citation} \rightarrow \text{Urdu RTL Rendering}$$
> It demonstrates how AI can serve smallholder agriculture responsibly by combining computer vision with strict knowledge-base provenance and domain guardrails.

### Q35: If you were given a research grant to continue this work for 12 months, what would you do next?
> **Answer:** We would pursue four specific research directions:
> 1. **Prospective Field Data Collection:** Partner with the KP Department of Agriculture Extension to collect and annotate 10,000 in-situ smartphone photographs across agricultural clusters in Swat, Mardan, and Abbottabad to train a domain-adapted vision model.
> 2. **Multi-Crop Expansion:** Expand the knowledge base and vision model to cover Wheat (Yellow/Stripe Rust), Sugarcane (Red Rot), and Citrus (Bacterial Canker).
> 3. **Pashto Localization & Voice Interface:** Implement a Pashto translation engine and audio text-to-speech interface to accommodate non-literate smallholder farmers in rural KP valleys.
> 4. **On-Device Edge Deployment:** Quantize the MobileNetV3 model to INT8 via ONNX Runtime / TFLite and deploy the entire pipeline as a zero-connectivity Android mobile application.
