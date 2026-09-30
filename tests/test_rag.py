"""Unit and integration tests for Agricultural RAG pipeline.

Tests:
1. Document loading & YAML frontmatter parsing.
2. Chunk metadata preservation and section categorization.
3. Embedding dimension and normalization.
4. Vector index loading and FAISS search.
5. Retrieval response contract and expected source metadata.
6. Crop and disease metadata filtering.
7. Provenance citation formatting.
8. In-domain supported query retrieval.
9. Out-of-scope query rejection (guardrail).
10. Empty query and zero-result fallback behavior.
"""

import pytest
import numpy as np
from pathlib import Path

from src import config
from src.rag.document_loader import (
    load_all_documents,
    parse_markdown_with_frontmatter,
    AgriculturalDocument,
)
from src.rag.chunker import (
    chunk_document,
    chunk_all_documents,
    AgriculturalChunk,
    infer_section_category,
)
from src.rag.embedder import AgriculturalEmbedder
from src.rag.vector_store import FAISSVectorStore
from src.rag.retriever import (
    AgriculturalRetriever,
    RetrievalResponse,
    format_citation,
    detect_out_of_scope_crop,
)


@pytest.fixture(scope="module")
def sample_documents():
    """Load curated documents from knowledge base."""
    docs = load_all_documents(config.KB_DOCUMENTS_DIR)
    assert len(docs) == 11, f"Expected 11 curated documents, found {len(docs)}"
    return docs


@pytest.fixture(scope="module")
def retriever_instance():
    """Instantiate a shared retriever with loaded vector store."""
    return AgriculturalRetriever(min_similarity_threshold=0.45)


def test_document_loading_and_provenance(sample_documents):
    """Verify document loading extracts all required provenance metadata."""
    required_keys = [
        "doc_id", "title", "crop", "disease", "disease_class",
        "organization", "author", "publication_date", "source_url",
        "region", "license", "language"
    ]
    for doc in sample_documents:
        d_dict = doc.to_dict()
        for k in required_keys:
            assert k in d_dict, f"Missing key '{k}' in doc {doc.doc_id}"
            assert d_dict[k] is not None, f"Key '{k}' is None in doc {doc.doc_id}"
        assert len(doc.content) > 500, f"Document content too short in {doc.doc_id}"


def test_chunk_metadata_preservation(sample_documents):
    """Verify chunking preserves section category, headers, and provenance."""
    first_doc = sample_documents[0]
    chunks = chunk_document(first_doc, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 0
    categories = {c.category for c in chunks}
    assert len(categories) >= 3, "Chunks should capture diverse information categories"

    for c in chunks:
        assert c.doc_id == first_doc.doc_id
        assert c.crop == first_doc.crop
        assert c.disease == first_doc.disease
        assert c.disease_class == first_doc.disease_class
        assert c.source_url == first_doc.source_url
        assert len(c.text) > 30
        assert f"[{first_doc.crop}" in c.text, "Chunk text must contain contextual header"


def test_section_category_inference():
    """Verify inference of section categories from markdown headers."""
    assert infer_section_category("Foliar and Fruit Symptoms") == "symptoms"
    assert infer_section_category("Causal Organism and Pathogen Biology") == "causes_pathogens"
    assert infer_section_category("Prevention and Cultural Practices") == "cultural_practices"
    assert infer_section_category("Treatment and Control Recommendations") == "treatment_control"
    assert infer_section_category("Safety, Pre-Harvest Intervals, and Notes") == "safety_usage_notes"
    assert infer_section_category("Relevant Crop-Stage Guidance") == "crop_stage_guidance"


def test_embedding_dimension_and_norm():
    """Verify embedder outputs 384-dimensional L2-normalized float32 vectors."""
    embedder = AgriculturalEmbedder()
    assert embedder.dimension == 384

    texts = ["Management of potato late blight in Swat", "Apple scab symptoms"]
    embeddings = embedder.embed_texts(texts)

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

    # Verify unit L2 norm
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)


def test_vector_index_loading():
    """Verify FAISS vector index loads with exact 164 chunks."""
    store = FAISSVectorStore.load()
    assert store.index.ntotal == 164
    assert len(store.chunks_metadata) == 164
    assert store.dimension == 384


def test_provenance_citation_formatting():
    """Verify citation string format contains doc_id, organization, and URL."""
    sample_meta = {
        "doc_id": "KP-AGRI-POT-002",
        "title": "Potato Late Blight Emergency Advisory",
        "organization": "Potato Research Centre Abbottabad",
        "publication_date": "2023-11",
        "source_url": "https://prc.gov.pk/late-blight",
    }
    citation = format_citation(sample_meta)
    assert "[KP-AGRI-POT-002]" in citation
    assert "Potato Research Centre Abbottabad" in citation
    assert "https://prc.gov.pk/late-blight" in citation


def test_in_domain_supported_query_retrieval(retriever_instance):
    """Verify in-domain queries return relevant chunks with status='success'."""
    query = "What are the leaf symptoms of potato late blight in Swat?"
    response = retriever_instance.retrieve(query, top_k=3)

    assert response.status == "success"
    assert len(response.chunks) == 3
    assert response.chunks[0].crop == "Potato"
    assert "Late_blight" in response.chunks[0].disease_class
    assert response.chunks[0].similarity_score > 0.50
    assert response.chunks[0].citation != ""


def test_crop_metadata_filtering(retriever_instance):
    """Verify crop-filtered retrieval restricts results to target crop."""
    # Query could match rust in either maize or apple
    query = "How to identify rust pustules on foliage?"
    resp_maize = retriever_instance.retrieve(query, top_k=3, filter_crop="Corn_(maize)")
    for chunk in resp_maize.chunks:
        assert chunk.crop in ["Corn_(maize)", "Maize"]

    resp_apple = retriever_instance.retrieve(query, top_k=3, filter_crop="Apple")
    for chunk in resp_apple.chunks:
        assert chunk.crop == "Apple"


def test_disease_class_filtering(retriever_instance):
    """Verify disease_class filtering strictly constrains returned chunks."""
    query = "What preventative fungicide sprays are recommended?"
    resp = retriever_instance.retrieve(
        query, top_k=3, filter_disease_class="Potato___Early_blight"
    )
    for chunk in resp.chunks:
        assert chunk.disease_class == "Potato___Early_blight"
        assert chunk.doc_id == "KP-AGRI-POT-001"


def test_out_of_scope_crop_guardrail_rejection(retriever_instance):
    """Verify queries targeting unsupported crops are safely rejected."""
    out_of_scope_queries = [
        "How do you manage cotton leaf curl virus in southern districts?",
        "What treatment is recommended for wheat stripe yellow rust?",
        "How to control sugarcane red rot in Charsadda fields?",
        "What spray cures citrus bacterial canker in Mardan orchards?",
    ]

    for q in out_of_scope_queries:
        resp = retriever_instance.retrieve(q, top_k=3)
        assert resp.status == "information_unavailable", f"Failed to reject: '{q}'"
        assert len(resp.chunks) == 0
        assert "outside the curated knowledge base" in resp.message


def test_detect_out_of_scope_crop():
    """Verify detection helper recognizes unsupported crops."""
    assert detect_out_of_scope_crop("Wheat rust control") == "Wheat"
    assert detect_out_of_scope_crop("Cotton pest management") == "Cotton"
    assert detect_out_of_scope_crop("Sugarcane red rot") == "Sugarcane"
    # Supported crops should return None
    assert detect_out_of_scope_crop("Apple scab symptoms") is None
    assert detect_out_of_scope_crop("Maize gray leaf spot") is None
    assert detect_out_of_scope_crop("Potato late blight") is None


def test_empty_query_fallback(retriever_instance):
    """Verify empty or whitespace query returns information_unavailable gracefully."""
    resp_empty = retriever_instance.retrieve("", top_k=3)
    assert resp_empty.status == "information_unavailable"
    assert len(resp_empty.chunks) == 0

    resp_space = retriever_instance.retrieve("   ", top_k=3)
    assert resp_space.status == "information_unavailable"
    assert len(resp_space.chunks) == 0
