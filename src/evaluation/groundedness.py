"""
Groundedness evaluation.
Checks whether the LLM-generated response is actually supported
by the retrieved context chunks, detecting hallucination.
"""
import numpy as np
from typing import List, Tuple
from loguru import logger

from src.indexing.embedder import Embedder


class GroundednessEvaluator:
    """Evaluates if the generated answer is grounded in retrieved context."""

    def __init__(self, embedder: Embedder, threshold: float = 0.6):
        self.embedder = embedder
        self.threshold = threshold

    def evaluate(
        self, answer: str, retrieved_chunks: List[Tuple[dict, float]]
    ) -> dict:
        """
        Check how well the answer is supported by the retrieved chunks.

        Approach:
            - Split answer into sentences
            - For each sentence, find max similarity to any chunk
            - Sentence is grounded if similarity exceeds threshold
            - Overall score = fraction of grounded sentences

        Args:
            answer: The LLM-generated response.
            retrieved_chunks: List of (chunk_dict, score) tuples.

        Returns:
            Dict with per-sentence analysis and overall groundedness score.
        """
        if not answer or not retrieved_chunks:
            return {"overall_score": 0.0, "sentence_scores": [], "is_grounded": False}

        sentences = self._split_sentences(answer)
        if not sentences:
            return {"overall_score": 0.0, "sentence_scores": [], "is_grounded": False}

        chunk_texts = [chunk["text"] for chunk, _ in retrieved_chunks]

        # Embed sentences and chunks
        sentence_embeddings = self.embedder.embed_texts(sentences, show_progress=False)
        chunk_embeddings = self.embedder.embed_texts(chunk_texts, show_progress=False)

        # Compute similarity matrix: (num_sentences, num_chunks)
        similarity_matrix = np.dot(sentence_embeddings, chunk_embeddings.T)

        sentence_scores = []
        grounded_count = 0

        for i, sentence in enumerate(sentences):
            max_sim = float(np.max(similarity_matrix[i]))
            best_chunk_idx = int(np.argmax(similarity_matrix[i]))
            is_grounded = max_sim >= self.threshold

            if is_grounded:
                grounded_count += 1

            sentence_scores.append({
                "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                "max_similarity": round(max_sim, 4),
                "best_matching_chunk": best_chunk_idx,
                "is_grounded": is_grounded,
            })

        overall = grounded_count / len(sentences) if sentences else 0.0

        logger.info(
            f"Groundedness: {grounded_count}/{len(sentences)} sentences grounded "
            f"(score={overall:.2f})"
        )

        return {
            "overall_score": round(overall, 4),
            "sentence_scores": sentence_scores,
            "is_grounded": overall >= self.threshold,
            "grounded_sentences": grounded_count,
            "total_sentences": len(sentences),
        }

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Simple sentence splitter."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]
