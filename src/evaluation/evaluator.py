"""
Unified evaluation pipeline.
Combines relevance, groundedness, safety, and reliability checks
into a single comprehensive report.
"""
from typing import List, Tuple, Callable, Optional
from loguru import logger

from src.indexing.embedder import Embedder
from src.evaluation.relevance import RelevanceEvaluator
from src.evaluation.groundedness import GroundednessEvaluator
from src.evaluation.safety import SafetyFilter
from src.evaluation.reliability import ReliabilityEvaluator


class ResponseEvaluator:
    """Runs all evaluation checks and produces a unified report."""

    def __init__(
        self,
        embedder: Embedder,
        relevance_threshold: float = 0.7,
        groundedness_threshold: float = 0.6,
        enable_safety: bool = True,
        reliability_runs: int = 3,
    ):
        self.relevance = RelevanceEvaluator(embedder, threshold=relevance_threshold)
        self.groundedness = GroundednessEvaluator(embedder, threshold=groundedness_threshold)
        self.safety = SafetyFilter(enable_filter=enable_safety)
        self.reliability = ReliabilityEvaluator(embedder, num_runs=reliability_runs)

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Tuple[dict, float]],
        query_fn: Optional[Callable] = None,
    ) -> dict:
        """
        Run full evaluation suite on a RAG response.

        Args:
            query: Original user question.
            answer: LLM-generated response.
            retrieved_chunks: List of (chunk_dict, score) from retrieval.
            query_fn: Optional callable for reliability testing.

        Returns:
            Comprehensive evaluation report.
        """
        logger.info("Running evaluation suite...")

        relevance_report = self.relevance.score_results(query, retrieved_chunks)
        groundedness_report = self.groundedness.evaluate(answer, retrieved_chunks)
        safety_report = self.safety.evaluate(answer, query)

        # Reliability (optional — requires query function)
        reliability_report = None
        reliability_score = 1.0  # Default to 1.0 if not tested
        if query_fn is not None:
            reliability_report = self.reliability.evaluate(query_fn, query)
            reliability_score = reliability_report["mean_consistency"]

        # Overall quality score (weighted average)
        quality_score = (
            0.30 * relevance_report["mean_score"]
            + 0.30 * groundedness_report["overall_score"]
            + 0.20 * (1.0 if safety_report["is_safe"] else 0.0)
            + 0.20 * reliability_score
        )

        report = {
            "quality_score": round(quality_score, 4),
            "relevance": relevance_report,
            "groundedness": groundedness_report,
            "safety": safety_report,
            "reliability": reliability_report,
            "pass": quality_score >= 0.5 and safety_report["is_safe"],
        }

        logger.info(f"Evaluation complete: quality={quality_score:.4f}, pass={report['pass']}")
        return report
