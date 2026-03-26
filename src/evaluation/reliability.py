"""
Reliability evaluation.
Tests response consistency by running the same query multiple times
and measuring variance across responses.
"""
import numpy as np
from typing import List, Callable
from loguru import logger

from src.indexing.embedder import Embedder


class ReliabilityEvaluator:
    """
    Measures response reliability by checking consistency
    across multiple runs of the same query.
    """

    def __init__(self, embedder: Embedder, num_runs: int = 3):
        self.embedder = embedder
        self.num_runs = num_runs

    def evaluate(self, query_fn: Callable, question: str) -> dict:
        """
        Run the same query multiple times and measure consistency.

        Args:
            query_fn: Callable that takes a question string and returns
                      a dict with an 'answer' field.
            question: The query to test.

        Returns:
            Dict with consistency scores and per-run details.
        """
        logger.info(f"Reliability test: running query {self.num_runs} times")
        responses = []

        for i in range(self.num_runs):
            result = query_fn(question)
            responses.append(result["answer"])
            logger.debug(f"Run {i+1}/{self.num_runs} complete ({len(result['answer'].split())} words)")

        # Embed all responses
        embeddings = self.embedder.embed_texts(responses, show_progress=False)

        # Compute pairwise cosine similarity
        similarity_matrix = np.dot(embeddings, embeddings.T)
        n = len(responses)

        pairwise_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_scores.append(float(similarity_matrix[i][j]))

        mean_consistency = float(np.mean(pairwise_scores)) if pairwise_scores else 0.0
        min_consistency = float(np.min(pairwise_scores)) if pairwise_scores else 0.0
        std_consistency = float(np.std(pairwise_scores)) if pairwise_scores else 0.0

        # Response length variance
        lengths = [len(r.split()) for r in responses]
        length_cv = float(np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 0 else 0.0

        is_reliable = mean_consistency >= 0.75 and length_cv <= 0.5

        logger.info(
            f"Reliability: mean_consistency={mean_consistency:.4f}, "
            f"length_cv={length_cv:.4f}, reliable={is_reliable}"
        )

        return {
            "mean_consistency": round(mean_consistency, 4),
            "min_consistency": round(min_consistency, 4),
            "std_consistency": round(std_consistency, 4),
            "length_coefficient_of_variation": round(length_cv, 4),
            "num_runs": self.num_runs,
            "is_reliable": is_reliable,
            "response_lengths": lengths,
            "pairwise_scores": [round(s, 4) for s in pairwise_scores],
        }
