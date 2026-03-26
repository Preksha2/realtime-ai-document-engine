"""
Retrieval relevance scoring.
Measures how well retrieved chunks match the user's query
using cosine similarity and rank-based metrics.
"""
import numpy as np
from typing import List, Tuple
from loguru import logger

from src.indexing.embedder import Embedder


class RelevanceEvaluator:
    """Evaluates the relevance of retrieved chunks to a query."""

    def __init__(self, embedder: Embedder, threshold: float = 0.7):
        self.embedder = embedder
        self.threshold = threshold

    def score_results(
        self, query: str, results: List[Tuple[dict, float]]
    ) -> dict:
        """
        Compute relevance metrics for a set of retrieved results.

        Args:
            query: The original user query.
            results: List of (chunk_dict, faiss_score) tuples.

        Returns:
            Dict with per-chunk scores and aggregate metrics.
        """
        if not results:
            return {"chunk_scores": [], "mean_score": 0.0, "above_threshold": 0, "precision_at_k": 0.0}

        query_embedding = self.embedder.embed_query(query)
        chunk_texts = [chunk["text"] for chunk, _ in results]
        chunk_embeddings = self.embedder.embed_texts(chunk_texts, show_progress=False)

        # Cosine similarity (embeddings are already normalized)
        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()

        chunk_scores = []
        above_threshold = 0
        for i, (chunk, faiss_score) in enumerate(results):
            cosine_sim = float(similarities[i])
            is_relevant = cosine_sim >= self.threshold
            if is_relevant:
                above_threshold += 1

            chunk_scores.append({
                "chunk_index": chunk["metadata"].get("chunk_index", i),
                "filename": chunk["metadata"].get("filename", "unknown"),
                "faiss_score": round(faiss_score, 4),
                "cosine_similarity": round(cosine_sim, 4),
                "is_relevant": is_relevant,
            })

        mean_score = float(np.mean(similarities))
        precision = above_threshold / len(results) if results else 0.0

        logger.info(
            f"Relevance: mean={mean_score:.4f}, "
            f"precision@{len(results)}={precision:.2f}, "
            f"{above_threshold}/{len(results)} above threshold"
        )

        return {
            "chunk_scores": chunk_scores,
            "mean_score": round(mean_score, 4),
            "above_threshold": above_threshold,
            "precision_at_k": round(precision, 4),
        }
