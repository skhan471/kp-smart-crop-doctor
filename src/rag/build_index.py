"""Knowledge base indexing script.

Ingests curated agricultural documents, segments them into semantic chunks,
computes dense vector embeddings, and builds the persistent local FAISS vector store.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.utils import get_logger
from src.rag.document_loader import load_all_documents
from src.rag.chunker import chunk_all_documents
from src.rag.embedder import AgriculturalEmbedder
from src.rag.vector_store import FAISSVectorStore

logger = get_logger(__name__)


def build_knowledge_base_index() -> FAISSVectorStore:
    """Execute complete ingestion, chunking, embedding, and indexing pipeline."""
    logger.info("=" * 70)
    logger.info("Building KP Smart Crop Doctor Agricultural Knowledge Base Index")
    logger.info("=" * 70)

    # 1. Ingest Documents
    logger.info(f"Ingesting documents from: {config.KB_DOCUMENTS_DIR}")
    documents = load_all_documents(config.KB_DOCUMENTS_DIR)
    if not documents:
        raise RuntimeError("No documents found in knowledge base directory!")

    logger.info(f"Successfully loaded {len(documents)} agricultural documents.")

    # 2. Chunk Documents
    logger.info(
        f"Chunking documents (max_chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})..."
    )
    chunks = chunk_all_documents(
        documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    logger.info(f"Generated {len(chunks)} total semantic chunks.")

    # 3. Generate Embeddings
    embedder = AgriculturalEmbedder()
    chunk_texts = [c.text for c in chunks]
    logger.info(f"Generating dense embeddings for {len(chunk_texts)} passages...")
    embeddings = embedder.embed_texts(chunk_texts, batch_size=32, show_progress=True)
    logger.info(f"Embeddings shape: {embeddings.shape} (dtype: {embeddings.dtype})")

    # 4. Build and Save FAISS Index
    vector_store = FAISSVectorStore(dimension=embedder.dimension)
    metadata_list = [c.to_dict() for c in chunks]
    vector_store.add_embeddings(embeddings, metadata_list)

    # Save to disk
    vector_store.save()
    logger.info("=" * 70)
    logger.info("Knowledge Base Indexing Successfully Completed!")
    logger.info(f"Indexed Documents: {len(documents)}")
    logger.info(f"Indexed Chunks:    {len(chunks)}")
    logger.info(f"Embedding Model:   {config.EMBEDDING_MODEL_NAME}")
    logger.info(f"Vector Store Path: {config.KB_VECTOR_STORE_DIR}")
    logger.info("=" * 70)

    return vector_store


if __name__ == "__main__":
    build_knowledge_base_index()
