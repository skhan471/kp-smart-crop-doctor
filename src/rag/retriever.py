"""Agricultural knowledge retriever module for source-grounded RAG.

Provides semantic top-k retrieval, metadata filtering by crop or disease class,
explicit source citation formatting, and guardrails for unsupported or out-of-scope queries.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from src import config
from src.utils import get_logger
from src.rag.embedder import AgriculturalEmbedder
from src.rag.vector_store import FAISSVectorStore

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a single retrieved chunk with similarity score and citation string."""

    chunk_id: str
    doc_id: str
    title: str
    crop: str
    disease: str
    disease_class: str
    section_title: str
    category: str
    text: str
    organization: str
    author: str
    source_url: str
    publication_date: Optional[str]
    similarity_score: float
    citation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "crop": self.crop,
            "disease": self.disease,
            "disease_class": self.disease_class,
            "section_title": self.section_title,
            "category": self.category,
            "text": self.text,
            "organization": self.organization,
            "source_url": self.source_url,
            "similarity_score": round(self.similarity_score, 4),
            "citation": self.citation,
        }


@dataclass
class RetrievalResponse:
    """Represents the complete retrieval output, including status and citations."""

    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    status: str = "success"  # "success" or "information_unavailable"
    message: Optional[str] = None
    applied_crop_filter: Optional[str] = None
    applied_disease_filter: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "message": self.message,
            "applied_crop_filter": self.applied_crop_filter,
            "applied_disease_filter": self.applied_disease_filter,
            "chunks_retrieved": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
        }


def format_citation(meta: Dict[str, Any]) -> str:
    """Format an authoritative and traceable citation string for a retrieved passage."""
    doc_id = meta.get("doc_id", "DOC")
    title = meta.get("title", "Advisory")
    org = meta.get("organization", "KP Agriculture Extension")
    pub_date = meta.get("publication_date", "N/A")
    url = meta.get("source_url", "https://agrikp.gov.pk")
    return f"[{doc_id}] {title}. {org} ({pub_date}). Available at: {url}"


SUPPORTED_CROPS = {"apple", "maize", "corn", "potato"}
SUPPORTED_CROPS_UR = {"سیب", "مکئی", "آلو"}

UNSUPPORTED_CROPS = {
    "wheat", "cotton", "sugarcane", "tobacco", "citrus", "orange",
    "lemon", "mango", "banana", "peach", "plum", "apricot",
    "tomato", "rice", "grape", "chilli", "pepper", "onion", "garlic"
}

UNSUPPORTED_CROPS_UR = {
    "گندم": "Wheat",
    "کپاس": "Cotton",
    "کماد": "Sugarcane",
    "گنا": "Sugarcane",
    "تمباکو": "Tobacco",
    "ترشاوہ": "Citrus",
    "سٹرس": "Citrus",
    "مالٹا": "Citrus",
    "لیموں": "Citrus",
    "آم": "Mango",
    "کیلا": "Banana",
    "آڑو": "Peach",
    "خوبانی": "Apricot",
    "ٹماٹر": "Tomato",
    "چاول": "Rice",
    "دھان": "Rice",
    "انگور": "Grape",
    "مرچ": "Chilli",
    "پیاز": "Onion",
    "لہسن": "Garlic",
}


def detect_out_of_scope_crop(query: str) -> Optional[str]:
    """Detect if a query explicitly targets an unsupported agricultural crop.

    Supports both English and Urdu crop designations.

    Args:
        query: User query string in English or Urdu.

    Returns:
        Name of unsupported crop if detected without supported crops, otherwise None.
    """
    words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    has_supported_en = any(c in words for c in SUPPORTED_CROPS)
    has_supported_ur = any(c in query for c in SUPPORTED_CROPS_UR)

    if not (has_supported_en or has_supported_ur):
        # 1. Check English crop words
        for word in words:
            if word in UNSUPPORTED_CROPS:
                return word.capitalize()
        # 2. Check Urdu crop words
        for ur_crop, en_crop in UNSUPPORTED_CROPS_UR.items():
            if ur_crop in query:
                return en_crop

    return None



class AgriculturalRetriever:
    """Knowledge retriever performing semantic vector search with provenance tracking."""

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedder: Optional[AgriculturalEmbedder] = None,
        min_similarity_threshold: float = 0.45,
    ):
        """Initialize retriever. Loads vector store and embedder if not supplied.

        Args:
            vector_store: Instantiated FAISSVectorStore.
            embedder: Instantiated AgriculturalEmbedder.
            min_similarity_threshold: Minimum cosine similarity score required for relevant results.
        """
        self.embedder = embedder or AgriculturalEmbedder()
        self.vector_store = vector_store or FAISSVectorStore.load()
        self.min_similarity_threshold = min_similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = config.RAG_TOP_K_CHUNKS,
        filter_crop: Optional[str] = None,
        filter_disease_class: Optional[str] = None,
    ) -> RetrievalResponse:
        """Retrieve the top-k most relevant agricultural passages for a given query.

        Args:
            query: User or diagnostic query string.
            top_k: Number of passages to retrieve (default: 3).
            filter_crop: Optional crop constraint ('Apple', 'Corn_(maize)', 'Potato').
            filter_disease_class: Optional disease class constraint.

        Returns:
            RetrievalResponse with retrieved chunks, citations, and status.
        """
        if not query or not query.strip():
            return RetrievalResponse(
                query=query,
                status="information_unavailable",
                message="Query string is empty.",
            )

        # 0. Check out-of-scope crop guardrail
        unsupported_crop = detect_out_of_scope_crop(query)
        if unsupported_crop:
            logger.info(
                f"Query '{query}' asks about unsupported crop '{unsupported_crop}'. "
                "Triggering information_unavailable guardrail."
            )
            return RetrievalResponse(
                query=query,
                chunks=[],
                status="information_unavailable",
                message=(
                    f"The requested agricultural guidance for '{unsupported_crop}' is outside the "
                    "curated knowledge base of KP Smart Crop Doctor (which supports Apple, Maize, and Potato). "
                    "To prevent unsupported field practices, please consult your local KP Agriculture "
                    "Extension Officer or certified agronomist."
                ),
                applied_crop_filter=filter_crop,
                applied_disease_filter=filter_disease_class,
            )

        # 1. Embed query
        query_vec = self.embedder.embed_query(query)

        # 2. Fetch candidates from FAISS (fetch more if filters are active)
        fetch_k = top_k * 4 if (filter_crop or filter_disease_class) else top_k
        raw_results = self.vector_store.search(query_vec, top_k=fetch_k)

        if not raw_results:
            return RetrievalResponse(
                query=query,
                status="information_unavailable",
                message="No matching agricultural documents found in the local knowledge base.",
                applied_crop_filter=filter_crop,
                applied_disease_filter=filter_disease_class,
            )

        # 3. Apply metadata filters if provided
        filtered_results = []
        for meta, score in raw_results:
            if filter_crop:
                # Normalize crop names (e.g. 'Corn_(maize)' vs 'maize' or 'Maize')
                crop_meta = meta.get("crop", "").lower()
                crop_req = filter_crop.lower()
                if crop_req not in crop_meta and crop_meta not in crop_req:
                    continue

            if filter_disease_class:
                if meta.get("disease_class") != filter_disease_class:
                    continue

            filtered_results.append((meta, score))

            if len(filtered_results) >= top_k:
                break

        # 4. Check confidence threshold guardrail
        if not filtered_results or filtered_results[0][1] < self.min_similarity_threshold:
            logger.warning(
                f"Query '{query}' produced low top similarity "
                f"({filtered_results[0][1] if filtered_results else 0.0:.3f} < {self.min_similarity_threshold}). "
                "Triggering information_unavailable guardrail."
            )
            return RetrievalResponse(
                query=query,
                chunks=[],
                status="information_unavailable",
                message=(
                    "The requested agricultural guidance or specific treatment/dosage recommendation "
                    "is not verified in the curated KP agricultural extension knowledge base. "
                    "To prevent hazardous crop management, please consult your local KP Agriculture Extension "
                    "Officer or certified agronomist."
                ),
                applied_crop_filter=filter_crop,
                applied_disease_filter=filter_disease_class,
            )

        # 5. Format retrieved chunks with citations
        retrieved_chunks = []
        for meta, score in filtered_results:
            citation_str = format_citation(meta)
            chunk = RetrievedChunk(
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                title=meta["title"],
                crop=meta["crop"],
                disease=meta["disease"],
                disease_class=meta["disease_class"],
                section_title=meta.get("section_title", "General"),
                category=meta.get("category", "general_guidance"),
                text=meta["text"],
                organization=meta.get("organization", "KP Agriculture Extension"),
                author=meta.get("author", "KP Agronomists"),
                source_url=meta.get("source_url", "https://agrikp.gov.pk"),
                publication_date=meta.get("publication_date"),
                similarity_score=score,
                citation=citation_str,
            )
            retrieved_chunks.append(chunk)

        return RetrievalResponse(
            query=query,
            chunks=retrieved_chunks,
            status="success",
            applied_crop_filter=filter_crop,
            applied_disease_filter=filter_disease_class,
        )
