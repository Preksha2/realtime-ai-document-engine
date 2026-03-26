"""
Text chunking with configurable size and overlap.
Uses token-aware splitting to stay within embedding model limits.
"""
from typing import List

import tiktoken
from loguru import logger


class TextChunker:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk_document(self, document: dict) -> List[dict]:
        """Split a single document into chunks, preserving metadata."""
        text = document["text"]
        metadata = document["metadata"]

        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "token_count": len(chunk_tokens),
                }
            })

            start += self.chunk_size - self.chunk_overlap

        logger.debug(f"Chunked '{metadata['filename']}' into {len(chunks)} chunks")
        return chunks

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        """Chunk a list of documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
