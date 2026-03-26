"""
End-to-end indexing pipeline: ingest documents, generate embeddings,
and build the FAISS index.
"""
from typing import Optional
from loguru import logger

from src.ingestion import DocumentLoader, TextChunker, TextPreprocessor
from src.indexing.embedder import Embedder
from src.indexing.faiss_index import FAISSIndex


class IndexingPipeline:
    """Orchestrates the full document-to-index pipeline."""

    def __init__(
        self,
        source_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "IVFFlat",
        nlist: int = 100,
    ):
        self.loader = DocumentLoader(source_dir)
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = Embedder(model_name=embedding_model)
        self.faiss_index = FAISSIndex(
            embedding_dim=self.embedder.embedding_dim,
            index_type=index_type,
            nlist=nlist,
        )

    def run(self, save_path: Optional[str] = None) -> FAISSIndex:
        """
        Execute the full indexing pipeline.

        Steps:
            1. Load documents from source directory
            2. Preprocess and clean text
            3. Chunk into overlapping segments
            4. Generate embeddings
            5. Build FAISS index
            6. Optionally save to disk

        Returns:
            The built FAISSIndex instance.
        """
        logger.info("=== Starting Indexing Pipeline ===")

        # Step 1: Load
        documents = self.loader.load_all()
        if not documents:
            raise ValueError("No documents found in source directory.")

        # Step 2: Preprocess
        documents = self.preprocessor.batch_clean(documents)

        # Step 3: Chunk
        chunks = self.chunker.chunk_documents(documents)

        # Step 4: Embed
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Step 5: Build index
        self.faiss_index.build(embeddings, chunks)

        # Step 6: Save
        if save_path:
            self.faiss_index.save(save_path)

        logger.info("=== Indexing Pipeline Complete ===")
        return self.faiss_index
