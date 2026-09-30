"""Retrieval-Augmented Generation (RAG) and multilingual advisory package.

Provides document ingestion, section-aware chunking, dense vector embeddings,
local FAISS vector storage, source-grounded retrieval, terminology glossaries,
and grounded bilingual (English/Urdu) advisory generation.
"""

from src.rag.document_loader import (
    AgriculturalDocument,
    load_all_documents,
    parse_markdown_with_frontmatter,
)
from src.rag.chunker import (
    AgriculturalChunk,
    chunk_document,
    chunk_all_documents,
)
from src.rag.embedder import AgriculturalEmbedder
from src.rag.vector_store import FAISSVectorStore
from src.rag.retriever import (
    AgriculturalRetriever,
    RetrievedChunk,
    RetrievalResponse,
    format_citation,
)
from src.rag.glossary import (
    AgriculturalGlossary,
    get_glossary,
)
from src.rag.translator import (
    AgriculturalTranslator,
)
from src.rag.generator import (
    StructuredAdvisory,
    GroundedAdvisoryGenerator,
    LocalExtractiveSynthesizer,
)

__all__ = [
    "AgriculturalDocument",
    "load_all_documents",
    "parse_markdown_with_frontmatter",
    "AgriculturalChunk",
    "chunk_document",
    "chunk_all_documents",
    "AgriculturalEmbedder",
    "FAISSVectorStore",
    "AgriculturalRetriever",
    "RetrievedChunk",
    "RetrievalResponse",
    "format_citation",
    "AgriculturalGlossary",
    "get_glossary",
    "AgriculturalTranslator",
    "StructuredAdvisory",
    "GroundedAdvisoryGenerator",
    "LocalExtractiveSynthesizer",
]
