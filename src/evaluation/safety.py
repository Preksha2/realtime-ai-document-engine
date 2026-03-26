"""
Safety filter for RAG responses.
Detects potentially harmful, off-topic, or policy-violating content
in generated responses.
"""
import re
from typing import List
from loguru import logger


class SafetyFilter:
    """Filters and flags unsafe or off-topic LLM responses."""

    # Categories of content to flag
    UNSAFE_PATTERNS = {
        "personal_info": [
            r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',       # SSN pattern
            r'\b\d{16}\b',                             # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ],
        "hallucination_signals": [
            r'\b(I think|I believe|I assume|probably|maybe|might be)\b',
            r'\b(as an AI|as a language model|I don\'t have access)\b',
        ],
        "refusal_patterns": [
            r'\b(I cannot|I can\'t|I\'m unable to|I am not able to)\b',
            r'\b(not enough information|context does not|cannot determine)\b',
        ],
    }

    def __init__(self, enable_filter: bool = True):
        self.enable_filter = enable_filter

    def evaluate(self, answer: str, query: str) -> dict:
        """
        Run safety checks on a generated response.

        Args:
            answer: The LLM-generated response.
            query: The original user query.

        Returns:
            Dict with safety verdict and flagged issues.
        """
        if not self.enable_filter:
            return {"is_safe": True, "flags": [], "risk_level": "none"}

        flags = []

        # Check for personal information leakage
        for pattern in self.UNSAFE_PATTERNS["personal_info"]:
            if re.search(pattern, answer):
                flags.append({
                    "category": "personal_info_leak",
                    "severity": "high",
                    "detail": "Response may contain personal identifiable information",
                })

        # Check for hallucination signals
        hedging_count = 0
        for pattern in self.UNSAFE_PATTERNS["hallucination_signals"]:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            hedging_count += len(matches)

        if hedging_count >= 3:
            flags.append({
                "category": "potential_hallucination",
                "severity": "medium",
                "detail": f"Response contains {hedging_count} hedging/uncertainty phrases",
            })

        # Check for refusal patterns
        refusal_count = 0
        for pattern in self.UNSAFE_PATTERNS["refusal_patterns"]:
            if re.search(pattern, answer, re.IGNORECASE):
                refusal_count += 1

        if refusal_count > 0:
            flags.append({
                "category": "partial_refusal",
                "severity": "low",
                "detail": "Response indicates insufficient context for full answer",
            })

        # Check response length anomalies
        if len(answer.split()) < 5:
            flags.append({
                "category": "too_short",
                "severity": "medium",
                "detail": "Response is unusually short, may be incomplete",
            })
        elif len(answer.split()) > 1000:
            flags.append({
                "category": "too_long",
                "severity": "low",
                "detail": "Response is unusually long, may contain filler",
            })

        # Determine overall risk level
        severities = [f["severity"] for f in flags]
        if "high" in severities:
            risk_level = "high"
        elif "medium" in severities:
            risk_level = "medium"
        elif flags:
            risk_level = "low"
        else:
            risk_level = "none"

        is_safe = risk_level in ("none", "low")

        logger.info(f"Safety check: risk={risk_level}, flags={len(flags)}, safe={is_safe}")

        return {
            "is_safe": is_safe,
            "flags": flags,
            "risk_level": risk_level,
            "total_flags": len(flags),
        }
