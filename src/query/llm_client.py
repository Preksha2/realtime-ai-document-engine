"""
LLM client supporting multiple backends:
  - HuggingFace transformers (default, free, runs locally)
  - OpenAI API (optional, requires API key)
"""
import os
from typing import List, Optional
from loguru import logger


class LLMClient:
    """Handles LLM completions with pluggable backends."""

    def __init__(
        self,
        backend: str = "huggingface",
        model: str = None,
        temperature: float = 0.2,
        max_response_tokens: int = 512,
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_response_tokens = max_response_tokens
        self._model = None

        if backend == "openai":
            self._init_openai(model or "gpt-3.5-turbo", api_key)
        elif backend == "huggingface":
            self._init_huggingface(model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'huggingface' or 'openai'.")

    def _init_openai(self, model: str, api_key: Optional[str]):
        from openai import OpenAI
        self.model_name = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info(f"LLM backend: OpenAI ({model})")

    def _init_huggingface(self, model: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.model_name = model
        logger.info(f"Loading HuggingFace model: {model} (this may take a moment on first run)")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_response_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        logger.info(f"HuggingFace model loaded: {model}")

    def generate(self, messages: List[dict]) -> dict:
        """Generate a response using the configured backend."""
        if self.backend == "openai":
            return self._generate_openai(messages)
        else:
            return self._generate_huggingface(messages)

    def _generate_openai(self, messages: List[dict]) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
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
            logger.info(f"OpenAI response generated ({usage['total_tokens']} tokens)")
            return {"answer": answer, "usage": usage, "model": response.model}

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return {"answer": f"Error: {str(e)}", "usage": {}, "model": self.model_name}

    def _generate_huggingface(self, messages: List[dict]) -> dict:
        try:
            # Build prompt from messages
            prompt = self._format_chat_prompt(messages)

            output = self.pipe(prompt)
            generated = output[0]["generated_text"]

            # Extract only the new generated text after the prompt
            if generated.startswith(prompt):
                answer = generated[len(prompt):].strip()
            else:
                answer = generated.strip()

            # Clean up common artifacts
            answer = answer.split("\n\n---")[0].strip()
            answer = answer.split("[/INST]")[0].strip()
            answer = answer.split("</s>")[0].strip()

            logger.info(f"HuggingFace response generated ({len(answer.split())} words)")
            return {
                "answer": answer,
                "usage": {"total_tokens": len(self.tokenizer.encode(prompt + answer))},
                "model": self.model_name,
            }

        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            return {"answer": f"Error: {str(e)}", "usage": {}, "model": self.model_name}

    @staticmethod
    def _format_chat_prompt(messages: List[dict]) -> str:
        """Convert chat messages to a single prompt string for local models."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}</s>")
        parts.append("<|assistant|>")
        return "\n".join(parts)
