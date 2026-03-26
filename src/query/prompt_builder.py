"""
Prompt construction for RAG-based LLM querying.
Builds grounded prompts from retrieved context chunks.
"""
from typing import List, Tuple
from loguru import logger


class PromptBuilder:
    """Constructs LLM prompts from retrieved document chunks."""

    SYSTEM_PROMPT = (
        "You are a precise document analysis assistant. Answer the user's question "
        "based ONLY on the provided context. If the context does not contain enough "
        "information to answer, say so clearly. Do not make up information.\n\n"
        "Guidelines:\n"
        "- Cite specific sections when possible\n"
        "- Be concise and direct\n"
        "- If multiple chunks are relevant, synthesize them\n"
        "- Flag any uncertainty or ambiguity"
    )

    def __init__(self, max_context_tokens: int = 2048):
        self.max_context_tokens = max_context_tokens

    def build(self, query: str, retrieved_chunks: List[Tuple[dict, float]]) -> List[dict]:
        """
        Build a chat-formatted prompt with retrieved context.

        Args:
            query: The user's natural language question.
            retrieved_chunks: List of (chunk_dict, score) from FAISS search.

        Returns:
            List of message dicts for the LLM API (system, user).
        """
        context_block = self._format_context(retrieved_chunks)

        user_message = (
            f"Context:\n{context_block}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Answer based on the context above:"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        logger.debug(f"Prompt built with {len(retrieved_chunks)} context chunks")
        return messages

    def _format_context(self, chunks: List[Tuple[dict, float]]) -> str:
        """Format retrieved chunks into a numbered context block."""
        sections = []
        total_tokens = 0

        for i, (chunk, score) in enumerate(chunks, 1):
            token_count = chunk["metadata"].get("token_count", 0)
            if total_tokens + token_count > self.max_context_tokens:
                logger.debug(f"Context truncated at chunk {i} (token limit reached)")
                break

            source = chunk["metadata"].get("filename", "unknown")
            chunk_idx = chunk["metadata"].get("chunk_index", "?")
            sections.append(
                f"[Source: {source} | Chunk {chunk_idx} | Relevance: {score:.3f}]\n"
                f"{chunk['text']}"
            )
            total_tokens += token_count

        return "\n\n".join(sections)
