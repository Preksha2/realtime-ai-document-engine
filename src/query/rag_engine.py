"""
Core RAG engine: ties together retrieval, prompt building, and LLM generation
into a single query interface.
"""
from typing import List, Optional
from loguru import logger

from src.indexing.embedder import Embedder
from src.indexing.faiss_index import FAISSIndex
from src.query.prompt_builder import PromptBuilder
from src.query.llm_client import LLMClient


class RAGEngine:
    """
    End-to-end Retrieval-Augmented Generation engine.

    Flow: query -> embed -> retrieve -> build prompt -> LLM -> response
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        embedder: Embedder,
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        llm_backend: str = "huggingface",
        temperature: float = 0.2,
        top_k: int = 5,
        similarity_threshold: float = 0.65,
        max_context_tokens: int = 2048,
    ):
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.prompt_builder = PromptBuilder(max_context_tokens=max_context_tokens)
        self.llm_client = LLMClient(
            backend=llm_backend,
            model=llm_model,
            temperature=temperature,
        )

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """
        Process a natural language query through the full RAG pipeline.

        Args:
            question: User's question in natural language.
            top_k: Override default number of chunks to retrieve.

        Returns:
            Dict containing answer, sources, scores, and metadata.
        """
        k = top_k or self.top_k
        logger.info(f"Processing query: '{question[:80]}...' (top_k={k})")

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(question)

        # Step 2: Retrieve relevant chunks
        results = self.faiss_index.search(query_embedding, top_k=k)

        # Step 3: Filter by similarity threshold
        filtered = [
            (chunk, score) for chunk, score in results
            if score >= self.similarity_threshold
        ]

        if not filtered:
            logger.warning("No chunks passed similarity threshold")
            return {
                "answer": "I couldn't find sufficiently relevant information in the event logs to answer your question.",
                "sources": [],
                "scores": [],
                "chunks_retrieved": len(results),
                "chunks_used": 0,
            }

        logger.info(f"Retrieved {len(results)} chunks, {len(filtered)} passed threshold ({self.similarity_threshold})")

        # Step 4: Build prompt
        messages = self.prompt_builder.build(question, filtered)

        # Step 5: Generate response
        llm_response = self.llm_client.generate(messages)

        # Step 6: Package response with metadata
        sources = [
            {
                "filename": chunk["metadata"]["filename"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "score": round(score, 4),
                "preview": chunk["text"][:150] + "...",
            }
            for chunk, score in filtered
        ]

        return {
            "answer": llm_response["answer"],
            "sources": sources,
            "scores": [s["score"] for s in sources],
            "chunks_retrieved": len(results),
            "chunks_used": len(filtered),
            "llm_usage": llm_response.get("usage", {}),
            "model": llm_response.get("model", ""),
        }
