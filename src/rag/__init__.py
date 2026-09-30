"""Retrieval-Augmented Generation (RAG) and agricultural knowledge base package.

Provides document ingestion, section-aware chunking, dense vector embeddings,
local FAISS vector storage, and source-grounded retrieval with provenance citations.
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
]
