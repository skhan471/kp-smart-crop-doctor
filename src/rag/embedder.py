"""Embedding generation module using sentence-transformers for agricultural text chunks.

Generates L2-normalized dense vector representations of queries and passages.
"""

from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


class AgriculturalEmbedder:
    """Wrapper around SentenceTransformer for generating normalized dense embeddings."""

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize embedder with specified sentence-transformers model.

        Args:
            model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2).
            device: 'cpu' or 'cuda'. If None, autodetects.
        """
        self.model_name = model_name
        self.device = str(device) if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Vector dimension: {self.dimension}")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Compute L2-normalized embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding (default: 32).
            show_progress: Whether to display progress bar.

        Returns:
            2D numpy array of shape (N, dimension) with float32 type and unit L2 norm.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization ensures inner product == cosine similarity
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Compute L2-normalized embedding for a single user query.

        Args:
            query: Query string.

        Returns:
            1D numpy array of shape (dimension,) with float32 type and unit L2 norm.
        """
        embedding = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return embedding.astype(np.float32)
