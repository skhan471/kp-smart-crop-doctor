"""Local FAISS vector store for semantic similarity search over agricultural text chunks.

Uses IndexFlatIP with normalized vectors for exact Cosine Similarity calculations.
Serializes both the FAISS index binary and chunk metadata JSON.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """FAISS-based vector index supporting cosine similarity and metadata retrieval."""

    def __init__(self, dimension: int = 384):
        """Initialize empty vector store with given embedding dimension."""
        self.dimension = dimension
        # IndexFlatIP computes inner products. With L2-normalized vectors, this equals cosine similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata: List[Dict[str, Any]] = []

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """Add dense vectors and corresponding metadata dictionaries to the index.

        Args:
            embeddings: 2D numpy array of shape (N, dimension) of type float32.
            metadata_list: List of length N containing chunk metadata dictionaries.
        """
        if embeddings.shape[0] != len(metadata_list):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match "
                f"metadata count ({len(metadata_list)})."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match "
                f"index dimension ({self.dimension})."
            )

        # Ensure float32 format
        vectors = embeddings.astype(np.float32)
        self.index.add(vectors)
        self.chunks_metadata.extend(metadata_list)
        logger.info(
            f"Added {len(metadata_list)} vectors to FAISS index. "
            f"Total vectors in index: {self.index.ntotal}"
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search index for the top-k most similar chunks.

        Args:
            query_vector: 1D or 2D numpy array with float32 values.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of (chunk_metadata, cosine_similarity_score) tuples, sorted by score descending.
        """
        if self.index.ntotal == 0:
            logger.warning("Search attempted on an empty FAISS index.")
            return []

        # Ensure 2D shape (1, dimension)
        if query_vector.ndim == 1:
            q_vec = query_vector.reshape(1, -1).astype(np.float32)
        else:
            q_vec = query_vector.astype(np.float32)

        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_vec, actual_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.chunks_metadata[idx]
            # Since vectors are normalized, dist is cosine similarity in [-1.0, 1.0]
            similarity_score = float(dist)
            results.append((meta, similarity_score))

        return results

    def save(
        self,
        index_file: Optional[Path] = None,
        metadata_file: Optional[Path] = None,
    ) -> None:
        """Serialize FAISS index binary and chunk metadata JSON to disk.

        Args:
            index_file: Path for FAISS index binary. Defaults to knowledge_base/vector_store/faiss_index.bin.
            metadata_file: Path for metadata JSON. Defaults to knowledge_base/vector_store/chunks_metadata.json.
        """
        idx_path = index_file or (config.KB_VECTOR_STORE_DIR / "faiss_index.bin")
        meta_path = metadata_file or (config.KB_VECTOR_STORE_DIR / "chunks_metadata.json")

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(idx_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, indent=2)

        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {idx_path}")
        logger.info(f"Saved chunk metadata ({len(self.chunks_metadata)} entries) to {meta_path}")

    @classmethod
    def load(
        cls,
        index_file: Optional[Path] = None,
        metadata_file: Optional[Path] = None,
    ) -> "FAISSVectorStore":
        """Load an existing FAISS index and metadata from disk.

        Args:
            index_file: Path to FAISS index binary.
            metadata_file: Path to metadata JSON.

        Returns:
            Instantiated and populated FAISSVectorStore.
        """
        idx_path = index_file or (config.KB_VECTOR_STORE_DIR / "faiss_index.bin")
        meta_path = metadata_file or (config.KB_VECTOR_STORE_DIR / "chunks_metadata.json")

        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {idx_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Chunks metadata not found at: {meta_path}")

        index = faiss.read_index(str(idx_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        store = cls(dimension=index.d)
        store.index = index
        store.chunks_metadata = metadata
        logger.info(
            f"Loaded FAISS index with {store.index.ntotal} vectors and "
            f"{len(store.chunks_metadata)} metadata records from {idx_path}."
        )
        return store
