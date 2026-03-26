"""
Unit tests for the embedding and FAISS indexing module.
"""
import numpy as np
import pytest

from src.indexing.faiss_index import FAISSIndex


class TestFAISSIndex:

    def _make_dummy_data(self, n: int = 100, dim: int = 384):
        embeddings = np.random.randn(n, dim).astype("float32")
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        chunks = [
            {"text": f"Chunk {i} content", "metadata": {"filename": f"doc_{i // 10}.txt", "chunk_index": i}}
            for i in range(n)
        ]
        return embeddings, chunks

    def test_build_flat_index(self):
        embeddings, chunks = self._make_dummy_data(50)
        index = FAISSIndex(embedding_dim=384, index_type="Flat")
        index.build(embeddings, chunks)
        assert index.index.ntotal == 50
        assert index.is_trained

    def test_build_ivf_index(self):
        embeddings, chunks = self._make_dummy_data(200)
        index = FAISSIndex(embedding_dim=384, index_type="IVFFlat", nlist=10)
        index.build(embeddings, chunks)
        assert index.index.ntotal == 200

    def test_search_returns_results(self):
        embeddings, chunks = self._make_dummy_data(100)
        index = FAISSIndex(embedding_dim=384, index_type="Flat")
        index.build(embeddings, chunks)

        query = np.random.randn(1, 384).astype("float32")
        query = query / np.linalg.norm(query)
        results = index.search(query, top_k=5)

        assert len(results) == 5
        assert all(isinstance(r, tuple) for r in results)
        assert all(isinstance(r[0], dict) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_search_respects_top_k(self):
        embeddings, chunks = self._make_dummy_data(50)
        index = FAISSIndex(embedding_dim=384, index_type="Flat")
        index.build(embeddings, chunks)

        query = np.random.randn(1, 384).astype("float32")
        query = query / np.linalg.norm(query)

        results_3 = index.search(query, top_k=3)
        results_10 = index.search(query, top_k=10)
        assert len(results_3) == 3
        assert len(results_10) == 10

    def test_search_without_build_raises(self):
        index = FAISSIndex(embedding_dim=384)
        query = np.random.randn(1, 384).astype("float32")
        with pytest.raises(RuntimeError):
            index.search(query)

    def test_save_and_load(self, tmp_path):
        embeddings, chunks = self._make_dummy_data(50)
        index = FAISSIndex(embedding_dim=384, index_type="Flat")
        index.build(embeddings, chunks)

        index_path = str(tmp_path / "test_index.bin")
        index.save(index_path)

        loaded = FAISSIndex(embedding_dim=384)
        loaded.load(index_path)

        assert loaded.index.ntotal == 50
        assert len(loaded.chunk_store) == 50
