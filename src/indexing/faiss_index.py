"""
FAISS index management: build, search, save, and load.
Supports both flat (exact) and IVF (approximate) indexes.
"""
import os
import numpy as np
import faiss
from typing import List, Tuple, Optional
from loguru import logger


class FAISSIndex:
    """Manages a FAISS vector index for similarity search."""

    def __init__(self, embedding_dim: int, index_type: str = "IVFFlat", nlist: int = 100):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.index: Optional[faiss.Index] = None
        self.is_trained = False
        self.chunk_store: List[dict] = []

    def build(self, embeddings: np.ndarray, chunks: List[dict]) -> None:
        """
        Build the FAISS index from embeddings and associate with chunk metadata.

        Args:
            embeddings: np.ndarray of shape (n, embedding_dim).
            chunks: List of chunk dicts with 'text' and 'metadata' keys.
        """
        n = embeddings.shape[0]
        logger.info(f"Building {self.index_type} index with {n} vectors (dim={self.embedding_dim})")

        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IVFFlat":
            effective_nlist = min(self.nlist, n)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, effective_nlist)
            logger.info(f"Training IVF index with nlist={effective_nlist}...")
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        self.index.add(embeddings)
        self.chunk_store = chunks
        self.is_trained = True
        logger.info(f"Index built: {self.index.ntotal} vectors indexed")

    def search(self, query_embedding: np.ndarray, top_k: int = 5, nprobe: int = 10) -> List[Tuple[dict, float]]:
        """
        Search the index for nearest neighbors.

        Args:
            query_embedding: np.ndarray of shape (1, embedding_dim).
            top_k: Number of results to return.
            nprobe: Number of clusters to probe (IVF only).

        Returns:
            List of (chunk_dict, similarity_score) tuples, sorted by relevance.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunk_store[idx], float(score)))

        logger.debug(f"Search returned {len(results)} results (top score: {results[0][1]:.4f})" if results else "No results found")
        return results

    def save(self, index_path: str, chunks_path: str = None) -> None:
        """Persist the FAISS index and chunk store to disk."""
        if self.index is None:
            raise RuntimeError("No index to save.")

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        logger.info(f"Index saved to {index_path}")

        if chunks_path is None:
            chunks_path = index_path.replace(".bin", "_chunks.npy")
        np.save(chunks_path, np.array(self.chunk_store, dtype=object))
        logger.info(f"Chunk store saved to {chunks_path}")

    def load(self, index_path: str, chunks_path: str = None) -> None:
        """Load a previously saved FAISS index and chunk store."""
        self.index = faiss.read_index(index_path)
        self.is_trained = True
        logger.info(f"Index loaded from {index_path} ({self.index.ntotal} vectors)")

        if chunks_path is None:
            chunks_path = index_path.replace(".bin", "_chunks.npy")
        self.chunk_store = np.load(chunks_path, allow_pickle=True).tolist()
        logger.info(f"Chunk store loaded: {len(self.chunk_store)} chunks")
