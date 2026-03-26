"""
Embedding generation using Sentence-Transformers.
Supports batch encoding with GPU acceleration when available.
"""
import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from loguru import logger


class Embedder:
    """Generates dense vector embeddings from text chunks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading embedding model '{model_name}' on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to encode per batch.
            show_progress: Whether to display a progress bar.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.info(f"Embeddings generated: shape={embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        Normalized for cosine similarity search.

        Returns:
            np.ndarray of shape (1, embedding_dim).
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding
