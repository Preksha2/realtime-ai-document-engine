"""
LLM client for generating responses via OpenAI-compatible APIs.
Supports configurable model, temperature, and token limits.
"""
import os
from typing import List, Optional
from openai import OpenAI
from loguru import logger


class LLMClient:
    """Handles LLM completions for RAG response generation."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        max_response_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_response_tokens = max_response_tokens
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info(f"LLM client initialized: model={model}, temp={temperature}")

    def generate(self, messages: List[dict]) -> dict:
        """
        Generate a response from the LLM.

        Args:
            messages: Chat-formatted messages (system + user).

        Returns:
            Dict with 'answer', 'usage', and 'model' fields.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_response_tokens,
            )

            answer = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            logger.info(f"LLM response generated ({usage['total_tokens']} tokens)")
            return {
                "answer": answer,
                "usage": usage,
                "model": response.model,
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "usage": {},
                "model": self.model,
            }
