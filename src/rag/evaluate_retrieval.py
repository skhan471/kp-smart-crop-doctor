"""Quantitative retrieval evaluation benchmark suite for agricultural RAG pipeline.

Measures:
1. Hit Rate @ 1, Hit Rate @ 3, Hit Rate @ 5 across all 11 classes.
2. Mean Reciprocal Rank (MRR @ 3, MRR @ 5).
3. Precision @ 1, Precision @ 3, Precision @ 5.
4. Topic / Category Grounding Accuracy.
5. Guardrail Rejection Accuracy on unverified / out-of-scope queries.
6. Retrieval Latency (ms per query).
7. Structured export to evaluation/retrieval_evaluation_report.json.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src import config
from src.utils import get_logger
from src.rag.retriever import AgriculturalRetriever

logger = get_logger(__name__)


# 33 Standard Domain Queries (3 per class) + 4 Adversarial/Out-of-Scope Queries
BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    # 1. Apple Scab
    {
        "query_id": "Q-APP-SCAB-01",
        "query": "What are the leaf and fruit symptoms of apple scab in Swat?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___Apple_scab",
        "expected_doc_id": "KP-AGRI-APP-001",
        "category": "symptoms",
    },
    {
        "query_id": "Q-APP-SCAB-02",
        "query": "How does Venturia inaequalis overwinter and when are primary ascospores discharged?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___Apple_scab",
        "expected_doc_id": "KP-AGRI-APP-001",
        "category": "causes_pathogens",
    },
    {
        "query_id": "Q-APP-SCAB-03",
        "query": "What cultural sanitation practices like urea spray reduce apple scab in orchards?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___Apple_scab",
        "expected_doc_id": "KP-AGRI-APP-001",
        "category": "cultural_practices",
    },

    # 2. Apple Black Rot
    {
        "query_id": "Q-APP-BROT-01",
        "query": "How do you identify frogeye leaf spot and black rot cankers on apple limbs?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___Black_rot",
        "expected_doc_id": "KP-AGRI-APP-002",
        "category": "symptoms",
    },
    {
        "query_id": "Q-APP-BROT-02",
        "query": "What pruning sanitation and mummified fruit removal prevents Botryosphaeria obtusa?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___Black_rot",
        "expected_doc_id": "KP-AGRI-APP-002",
        "category": "prevention",
    },
    {
        "query_id": "Q-APP-BROT-03",
        "query": "What protectant sprays are recommended from bud break to petal fall for frogeye spot?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___Black_rot",
        "expected_doc_id": "KP-AGRI-APP-002",
        "category": "treatment_control",
    },

    # 3. Apple Cedar Rust
    {
        "query_id": "Q-APP-RUST-01",
        "query": "Why do yellow-orange spots with tubelike aecia develop on apple leaves in Hazara?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___Cedar_apple_rust",
        "expected_doc_id": "KP-AGRI-APP-003",
        "category": "symptoms",
    },
    {
        "query_id": "Q-APP-RUST-02",
        "query": "What is the relationship between wild juniper red cedar galls and cedar apple rust?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___Cedar_apple_rust",
        "expected_doc_id": "KP-AGRI-APP-003",
        "category": "causes_pathogens",
    },
    {
        "query_id": "Q-APP-RUST-03",
        "query": "Can cedar apple rust spread from apple leaf to another apple leaf during summer?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___Cedar_apple_rust",
        "expected_doc_id": "KP-AGRI-APP-003",
        "category": "prevention",
    },

    # 4. Apple Healthy
    {
        "query_id": "Q-APP-HLTH-01",
        "query": "What are the recommended winter pruning and canopy aeration practices for apple trees in KP?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___healthy",
        "expected_doc_id": "KP-AGRI-APP-004",
        "category": "cultural_practices",
    },
    {
        "query_id": "Q-APP-HLTH-02",
        "query": "What NPK fertilizer schedule and zinc boron foliar sprays support healthy apple foliage?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___healthy",
        "expected_doc_id": "KP-AGRI-APP-004",
        "category": "cultural_practices",
    },
    {
        "query_id": "Q-APP-HLTH-03",
        "query": "Why should chemical spraying be prohibited during full apple bloom for honeybee pollinators?",
        "expected_crop": "Apple",
        "expected_disease_class": "Apple___healthy",
        "expected_doc_id": "KP-AGRI-APP-004",
        "category": "safety_usage_notes",
    },

    # 5. Maize Gray Leaf Spot
    {
        "query_id": "Q-MAI-GLS-01",
        "query": "What causes narrow rectangular vein-delimited tan lesions on maize leaves?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "expected_doc_id": "KP-AGRI-MAI-001",
        "category": "symptoms",
    },
    {
        "query_id": "Q-MAI-GLS-02",
        "query": "How does crop rotation and deep mouldboard plowing suppress Cercospora zeae-maydis in maize stubble?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "expected_doc_id": "KP-AGRI-MAI-001",
        "category": "prevention",
    },
    {
        "query_id": "Q-MAI-GLS-03",
        "query": "At what stage from tasseling to silking is fungicide treatment justified for gray leaf spot?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "expected_doc_id": "KP-AGRI-MAI-001",
        "category": "crop_stage_guidance",
    },

    # 6. Maize Common Rust
    {
        "query_id": "Q-MAI-RUST-01",
        "query": "What are the characteristics of raised cinnamon-brown powdery pustules of Puccinia sorghi on maize?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___Common_rust_",
        "expected_doc_id": "KP-AGRI-MAI-002",
        "category": "symptoms",
    },
    {
        "query_id": "Q-MAI-RUST-02",
        "query": "Which resistant varieties developed by CCRI Pirsabak manage common rust in KP?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___Common_rust_",
        "expected_doc_id": "KP-AGRI-MAI-002",
        "category": "prevention",
    },
    {
        "query_id": "Q-MAI-RUST-03",
        "query": "Why does excessive urea top-dressing aggravate common rust pustule development on corn?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___Common_rust_",
        "expected_doc_id": "KP-AGRI-MAI-002",
        "category": "cultural_practices",
    },

    # 7. Maize Northern Leaf Blight
    {
        "query_id": "Q-MAI-NLB-01",
        "query": "How do you distinguish large cigar-shaped elliptical lesions of Northern Corn Leaf Blight from Gray Leaf Spot?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___Northern_Leaf_Blight",
        "expected_doc_id": "KP-AGRI-MAI-003",
        "category": "symptoms",
    },
    {
        "query_id": "Q-MAI-NLB-02",
        "query": "What is the survival mechanism of Exserohilum turcicum in infected crop residue during winter?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___Northern_Leaf_Blight",
        "expected_doc_id": "KP-AGRI-MAI-003",
        "category": "causes_pathogens",
    },
    {
        "query_id": "Q-MAI-NLB-03",
        "query": "Why are fungicide sprays ineffective after dent stage R5 for Northern Corn Leaf Blight?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___Northern_Leaf_Blight",
        "expected_doc_id": "KP-AGRI-MAI-003",
        "category": "treatment_control",
    },

    # 8. Maize Healthy
    {
        "query_id": "Q-MAI-HLTH-01",
        "query": "Why is planting maize on raised ridges recommended in KP to avoid root waterlogging?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___healthy",
        "expected_doc_id": "KP-AGRI-MAI-004",
        "category": "cultural_practices",
    },
    {
        "query_id": "Q-MAI-HLTH-02",
        "query": "What is the split nitrogen fertilizer application schedule at knee-high and tasseling for maize?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___healthy",
        "expected_doc_id": "KP-AGRI-MAI-004",
        "category": "cultural_practices",
    },
    {
        "query_id": "Q-MAI-HLTH-03",
        "query": "Why is moisture stress during silking R1 particularly critical for maize ear development?",
        "expected_crop": "Corn_(maize)",
        "expected_disease_class": "Corn_(maize)___healthy",
        "expected_doc_id": "KP-AGRI-MAI-004",
        "category": "crop_stage_guidance",
    },

    # 9. Potato Early Blight
    {
        "query_id": "Q-POT-EBLT-01",
        "query": "What are the concentric target-board rings and chlorotic halos characteristic of potato early blight?",
        "expected_crop": "Potato",
        "expected_disease_class": "Potato___Early_blight",
        "expected_doc_id": "KP-AGRI-POT-001",
        "category": "symptoms",
    },
    {
        "query_id": "Q-POT-EBLT-02",
        "query": "How does nitrogen deficiency and plant senescence stress accelerate Alternaria solani infection?",
        "expected_crop": "Potato",
        "expected_disease_class": "Potato___Early_blight",
        "expected_doc_id": "KP-AGRI-POT-001",
        "category": "cultural_practices",
    },
    {
        "query_id": "Q-POT-EBLT-03",
        "query": "What protectant fungicides like mancozeb or chlorothalonil manage early blight during tuber bulking?",
        "expected_crop": "Potato",
        "expected_disease_class": "Potato___Early_blight",
        "expected_doc_id": "KP-AGRI-POT-001",
        "category": "treatment_control",
    },

    # 10. Potato Late Blight
    {
        "query_id": "Q-POT-LBLT-01",
        "query": "What are the emergency symptoms of Phytophthora infestans late blight and white mildew on leaf undersides?",
        "expected_crop": "Potato",
        "expected_disease_class": "Potato___Late_blight",
        "expected_doc_id": "KP-AGRI-POT-002",
        "category": "symptoms",
    },
    {
        "query_id": "Q-POT-LBLT-02",
        "query": "Why is the complete destruction of potato cull piles critical to preventing late blight epidemics in KP?",
        "expected_crop": "Potato",
        "expected_disease_class": "Potato___Late_blight",
        "expected_doc_id": "KP-AGRI-POT-002",
        "category": "prevention",
    },
    {
        "query_id": "Q-POT-LBLT-03",
        "query": "Why must potato vines be dehaulmed 10 to 14 days before harvest to prevent tuber rot?",
        "expected_crop": "Potato",
        "expected_disease_class": "Potato___Late_blight",
        "expected_doc_id": "KP-AGRI-POT-002",
        "category": "crop_stage_guidance",
    },

    # 11. Potato Healthy
    {
        "query_id": "Q-POT-HLTH-01",
        "query": "What certified seed tuber selection and pre-sprouting chitting practices ensure healthy potato crops in KP?",
        "expected_crop": "Potato",
        "expected_disease_class": "Potato___healthy",
        "expected_doc_id": "KP-AGRI-POT-003",
        "category": "cultural_practices",
    },
    {
        "query_id": "Q-POT-HLTH-02",
        "query": "What are the earthing-up and ridging recommendations to prevent potato tuber greening and solanine buildup?",
        "expected_crop": "Potato",
        "expected_disease_class": "Potato___healthy",
        "expected_doc_id": "KP-AGRI-POT-003",
        "category": "cultural_practices",
    },
    {
        "query_id": "Q-POT-HLTH-03",
        "query": "Why is a 3- to 4-year crop rotation with non-solanaceous crops required for potato soil health?",
        "expected_crop": "Potato",
        "expected_disease_class": "Potato___healthy",
        "expected_doc_id": "KP-AGRI-POT-003",
        "category": "prevention",
    },

    # Out-of-Scope Agricultural Queries (Must be safely rejected by Guardrail)
    {
        "query_id": "Q-OUTOFDOMAIN-01",
        "query": "How do you manage cotton leaf curl virus in southern districts?",
        "is_adversarial": True,
        "expected_status": "information_unavailable",
    },
    {
        "query_id": "Q-OUTOFDOMAIN-02",
        "query": "What treatment is recommended for wheat stripe yellow rust in Nowshera?",
        "is_adversarial": True,
        "expected_status": "information_unavailable",
    },
    {
        "query_id": "Q-OUTOFDOMAIN-03",
        "query": "How to control sugarcane red rot in Charsadda fields?",
        "is_adversarial": True,
        "expected_status": "information_unavailable",
    },
    {
        "query_id": "Q-OUTOFDOMAIN-04",
        "query": "What spray cures citrus bacterial canker in Mardan orchards?",
        "is_adversarial": True,
        "expected_status": "information_unavailable",
    },
]


def run_retrieval_evaluation() -> Dict[str, Any]:
    """Execute complete retrieval evaluation benchmark and compute quantitative metrics."""
    logger.info("Initializing AgriculturalRetriever for benchmark evaluation...")
    retriever = AgriculturalRetriever()

    domain_queries = [q for q in BENCHMARK_QUERIES if not q.get("is_adversarial", False)]
    adversarial_queries = [q for q in BENCHMARK_QUERIES if q.get("is_adversarial", False)]

    logger.info(
        f"Evaluating {len(domain_queries)} domain queries across 11 classes + "
        f"{len(adversarial_queries)} safety guardrail adversarial queries."
    )

    query_results = []
    latencies = []

    # Track metrics for top-1, top-3, top-5
    k_values = [1, 3, 5]
    hits = {k: 0 for k in k_values}
    reciprocal_ranks = []
    precisions = {k: [] for k in k_values}
    category_matches = []

    for item in domain_queries:
        qid = item["query_id"]
        q_text = item["query"]
        exp_doc = item["expected_doc_id"]
        exp_class = item["expected_disease_class"]
        exp_cat = item["category"]

        start_time = time.perf_counter()
        resp = retriever.retrieve(q_text, top_k=5)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(elapsed_ms)

        retrieved_docs = [c.doc_id for c in resp.chunks]
        retrieved_classes = [c.disease_class for c in resp.chunks]
        retrieved_categories = [c.category for c in resp.chunks]
        scores = [c.similarity_score for c in resp.chunks]

        # Calculate rank of first relevant document
        rank = 0
        for idx, d_id in enumerate(retrieved_docs):
            if d_id == exp_doc or retrieved_classes[idx] == exp_class:
                rank = idx + 1
                break

        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        # Hits and Precisions at k
        for k in k_values:
            top_k_docs = retrieved_docs[:k]
            top_k_classes = retrieved_classes[:k]
            is_hit = (exp_doc in top_k_docs) or (exp_class in top_k_classes)
            if is_hit:
                hits[k] += 1

            # Precision@k: relevant retrieved / k
            rel_count = sum(
                1 for i in range(min(k, len(retrieved_docs)))
                if retrieved_docs[i] == exp_doc or retrieved_classes[i] == exp_class
            )
            precisions[k].append(rel_count / k)

        # Category / section match in top 3
        cat_match = exp_cat in retrieved_categories[:3]
        category_matches.append(cat_match)

        query_results.append({
            "query_id": qid,
            "query": q_text,
            "expected_disease_class": exp_class,
            "expected_doc_id": exp_doc,
            "expected_category": exp_cat,
            "retrieved_top1_doc": retrieved_docs[0] if retrieved_docs else None,
            "retrieved_top1_score": round(scores[0], 4) if scores else 0.0,
            "rank_of_relevant": rank,
            "reciprocal_rank": round(rr, 4),
            "hit_top1": (rank == 1),
            "hit_top3": (1 <= rank <= 3),
            "hit_top5": (1 <= rank <= 5),
            "category_matched_in_top3": cat_match,
            "latency_ms": round(elapsed_ms, 2),
            "top_citation": resp.chunks[0].citation if resp.chunks else None,
        })

    # Adversarial Guardrail Queries
    adversarial_results = []
    guardrail_rejections = 0

    for item in adversarial_queries:
        qid = item["query_id"]
        q_text = item["query"]

        start_time = time.perf_counter()
        resp = retriever.retrieve(q_text, top_k=3)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        is_properly_rejected = (resp.status == "information_unavailable")
        if is_properly_rejected:
            guardrail_rejections += 1

        adversarial_results.append({
            "query_id": qid,
            "query": q_text,
            "status": resp.status,
            "message": resp.message,
            "properly_rejected": is_properly_rejected,
            "latency_ms": round(elapsed_ms, 2),
        })

    # Summary Statistics
    num_queries = len(domain_queries)
    hit_rate_top1 = hits[1] / num_queries
    hit_rate_top3 = hits[3] / num_queries
    hit_rate_top5 = hits[5] / num_queries
    mrr = float(np.mean(reciprocal_ranks))
    precision_top1 = float(np.mean(precisions[1]))
    precision_top3 = float(np.mean(precisions[3]))
    precision_top5 = float(np.mean(precisions[5]))
    category_match_rate = float(np.mean(category_matches))
    guardrail_accuracy = guardrail_rejections / len(adversarial_queries)
    avg_latency = float(np.mean(latencies))

    summary_metrics = {
        "total_domain_queries": num_queries,
        "total_adversarial_queries": len(adversarial_queries),
        "hit_rate_top1": round(hit_rate_top1, 4),
        "hit_rate_top3": round(hit_rate_top3, 4),
        "hit_rate_top5": round(hit_rate_top5, 4),
        "mean_reciprocal_rank_mrr": round(mrr, 4),
        "precision_top1": round(precision_top1, 4),
        "precision_top3": round(precision_top3, 4),
        "precision_top5": round(precision_top5, 4),
        "category_grounding_accuracy": round(category_match_rate, 4),
        "guardrail_rejection_accuracy": round(guardrail_accuracy, 4),
        "average_retrieval_latency_ms": round(avg_latency, 2),
    }

    report = {
        "milestone": "Milestone 5 - Curated KP Knowledge Base & Source-Grounded RAG Pipeline",
        "benchmark_summary": summary_metrics,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "vector_store_type": "FAISS IndexFlatIP (Cosine Similarity on Normalized Embeddings)",
        "chunking_strategy": {
            "max_chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "contextual_headers": True,
            "provenance_preservation": True,
        },
        "detailed_query_results": query_results,
        "adversarial_guardrail_results": adversarial_results,
    }

    # Save to evaluation directory
    eval_path = config.EVALUATION_DIR / "retrieval_evaluation_report.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 70)
    logger.info("RAG RETRIEVAL EVALUATION BENCHMARK RESULTS")
    logger.info("=" * 70)
    logger.info(f"Hit Rate @ 1:                 {hit_rate_top1 * 100:.2f}%")
    logger.info(f"Hit Rate @ 3:                 {hit_rate_top3 * 100:.2f}%")
    logger.info(f"Hit Rate @ 5:                 {hit_rate_top5 * 100:.2f}%")
    logger.info(f"Mean Reciprocal Rank (MRR):   {mrr:.4f}")
    logger.info(f"Precision @ 1:                {precision_top1 * 100:.2f}%")
    logger.info(f"Precision @ 3:                {precision_top3 * 100:.2f}%")
    logger.info(f"Category Grounding Accuracy:  {category_match_rate * 100:.2f}%")
    logger.info(f"Guardrail Rejection Accuracy: {guardrail_accuracy * 100:.2f}%")
    logger.info(f"Average Retrieval Latency:    {avg_latency:.2f} ms")
    logger.info(f"Report successfully saved to: {eval_path}")
    logger.info("=" * 70)

    return report


if __name__ == "__main__":
    run_retrieval_evaluation()
