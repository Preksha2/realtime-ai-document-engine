"""
Unit tests for the document ingestion module.
"""
import os
import tempfile
import pytest

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.ingestion.preprocessor import TextPreprocessor


class TestDocumentLoader:

    def test_load_txt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("This is a test document with some content for loading.")

            loader = DocumentLoader(tmpdir)
            docs = loader.load_all()
            assert len(docs) == 1
            assert "test document" in docs[0]["text"]
            assert docs[0]["metadata"]["filename"] == "test.txt"
            assert docs[0]["metadata"]["format"] == ".txt"

    def test_load_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DocumentLoader(tmpdir)
            docs = loader.load_all()
            assert len(docs) == 0

    def test_invalid_directory(self):
        with pytest.raises(FileNotFoundError):
            DocumentLoader("/nonexistent/path")

    def test_skips_unsupported_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "data.csv")
            with open(filepath, "w") as f:
                f.write("col1,col2\nval1,val2")

            loader = DocumentLoader(tmpdir)
            docs = loader.load_all()
            assert len(docs) == 0


class TestTextChunker:

    def test_basic_chunking(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        doc = {
            "text": "This is a fairly long document that should be split into multiple chunks for testing purposes.",
            "metadata": {"filename": "test.txt"}
        }
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        assert all("text" in c for c in chunks)
        assert all("metadata" in c for c in chunks)

    def test_preserves_metadata(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        doc = {
            "text": "Short document.",
            "metadata": {"filename": "report.pdf", "format": ".pdf"}
        }
        chunks = chunker.chunk_document(doc)
        assert chunks[0]["metadata"]["filename"] == "report.pdf"
        assert "chunk_index" in chunks[0]["metadata"]

    def test_chunk_overlap(self):
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        doc = {
            "text": "Word " * 50,
            "metadata": {"filename": "test.txt"}
        }
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_batch_chunking(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        docs = [
            {"text": "First document content here.", "metadata": {"filename": "a.txt"}},
            {"text": "Second document content here.", "metadata": {"filename": "b.txt"}},
        ]
        all_chunks = chunker.chunk_documents(docs)
        filenames = set(c["metadata"]["filename"] for c in all_chunks)
        assert "a.txt" in filenames
        assert "b.txt" in filenames


class TestTextPreprocessor:

    def test_normalize_whitespace(self):
        text = "Hello    world\n\n\n\nNew paragraph"
        result = TextPreprocessor.clean(text)
        assert "    " not in result
        assert "\n\n\n\n" not in result

    def test_remove_page_numbers(self):
        text = "Some content\nPage 5\nMore content\n3 of 10\nEnd"
        result = TextPreprocessor.clean(text)
        assert "Page 5" not in result
        assert "3 of 10" not in result

    def test_fix_encoding(self):
        text = "He said \u201chello\u201d and it\u2019s fine\u2026"
        result = TextPreprocessor.clean(text)
        assert "\u201c" not in result
        assert "\u2019" not in result
        assert '"hello"' in result

    def test_batch_clean(self):
        docs = [
            {"text": "Content\n\n\n\nhere", "metadata": {}},
            {"text": "More\u2019s the pity", "metadata": {}},
        ]
        cleaned = TextPreprocessor.batch_clean(docs)
        assert "\n\n\n\n" not in cleaned[0]["text"]
        assert "\u2019" not in cleaned[1]["text"]
