"""
Unit tests for the evaluation module.
"""
from src.evaluation.safety import SafetyFilter


class TestSafetyFilter:

    def setup_method(self):
        self.filter = SafetyFilter(enable_filter=True)

    def test_safe_response(self):
        result = self.filter.evaluate(
            answer="The quarterly revenue increased by 15% compared to last year.",
            query="What was the revenue growth?"
        )
        assert result["is_safe"] is True
        assert result["risk_level"] in ("none", "low")

    def test_detects_pii_email(self):
        result = self.filter.evaluate(
            answer="Contact john.doe@company.com for more details.",
            query="Who should I contact?"
        )
        pii_flags = [f for f in result["flags"] if f["category"] == "personal_info_leak"]
        assert len(pii_flags) > 0

    def test_detects_hallucination_signals(self):
        result = self.filter.evaluate(
            answer="I think the revenue maybe increased. I believe it probably was around 10%. It might be higher.",
            query="What was the revenue?"
        )
        hallucination_flags = [f for f in result["flags"] if f["category"] == "potential_hallucination"]
        assert len(hallucination_flags) > 0

    def test_detects_short_response(self):
        result = self.filter.evaluate(
            answer="Yes.",
            query="Explain the revenue model in detail."
        )
        short_flags = [f for f in result["flags"] if f["category"] == "too_short"]
        assert len(short_flags) > 0

    def test_disabled_filter(self):
        disabled = SafetyFilter(enable_filter=False)
        result = disabled.evaluate(
            answer="Contact john.doe@company.com",
            query="test"
        )
        assert result["is_safe"] is True
        assert result["flags"] == []

    def test_detects_refusal(self):
        result = self.filter.evaluate(
            answer="I cannot determine the answer because the context does not contain this information.",
            query="What is the profit margin?"
        )
        refusal_flags = [f for f in result["flags"] if f["category"] == "partial_refusal"]
        assert len(refusal_flags) > 0
