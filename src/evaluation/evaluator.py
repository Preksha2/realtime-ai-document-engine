"""
Unified evaluation pipeline.
Combines relevance, groundedness, and safety checks into a single report.
"""
from typing import List, Tuple
from loguru import logger

from src.indexing.embedder import Embedder
from src.evaluation.relevance import RelevanceEvaluator
from src.evaluation.groundedness import GroundednessEvaluator
from src.evaluation.safety import SafetyFilter


class ResponseEvaluator:
    """Runs all evaluation checks and produces a unified report."""

    def __init__(
        self,
        embedder: Embedder,
        relevance_threshold: float = 0.7,
        groundedness_threshold: float = 0.6,
        enable_safety: bool = True,
    ):
        self.relevance = RelevanceEvaluator(embedder, threshold=relevance_threshold)
        self.groundedness = GroundednessEvaluator(embedder, threshold=groundedness_threshold)
        self.safety = SafetyFilter(enable_filter=enable_safety)

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Tuple[dict, float]],
    ) -> dict:
        """
        Run full evaluation suite on a RAG response.

        Args:
            query: Original user question.
            answer: LLM-generated response.
            retrieved_chunks: List of (chunk_dict, score) from retrieval.

        Returns:
            Comprehensive evaluation report.
        """
        logger.info("Running evaluation suite...")

        relevance_report = self.relevance.score_results(query, retrieved_chunks)
        groundedness_report = self.groundedness.evaluate(answer, retrieved_chunks)
        safety_report = self.safety.evaluate(answer, query)

        # Overall quality score (weighted average)
        quality_score = (
            0.4 * relevance_report["mean_score"]
            + 0.4 * groundedness_report["overall_score"]
            + 0.2 * (1.0 if safety_report["is_safe"] else 0.0)
        )

        report = {
            "quality_score": round(quality_score, 4),
            "relevance": relevance_report,
            "groundedness": groundedness_report,
            "safety": safety_report,
            "pass": quality_score >= 0.5 and safety_report["is_safe"],
        }

        logger.info(f"Evaluation complete: quality={quality_score:.4f}, pass={report['pass']}")
        return report
