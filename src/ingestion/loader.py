"""
Document loader supporting PDF, TXT, and DOCX formats.
"""
import os
from pathlib import Path
from typing import List

from pypdf import PdfReader
from docx import Document as DocxDocument
from loguru import logger


class DocumentLoader:
    """Loads raw text from documents in supported formats."""

    SUPPORTED_FORMATS = {".pdf", ".txt", ".docx"}

    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

    def load_all(self) -> List[dict]:
        """Load all supported documents from the source directory."""
        documents = []
        for filepath in self.source_dir.rglob("*"):
            if filepath.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    text = self._load_file(filepath)
                    documents.append({
                        "text": text,
                        "metadata": {
                            "filename": filepath.name,
                            "filepath": str(filepath),
                            "format": filepath.suffix.lower(),
                            "size_bytes": filepath.stat().st_size,
                        }
                    })
                    logger.info(f"Loaded: {filepath.name} ({len(text)} chars)")
                except Exception as e:
                    logger.error(f"Failed to load {filepath.name}: {e}")
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def _load_file(self, filepath: Path) -> str:
        """Dispatch to the appropriate loader based on file extension."""
        ext = filepath.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(filepath)
        elif ext == ".txt":
            return self._load_txt(filepath)
        elif ext == ".docx":
            return self._load_docx(filepath)
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def _load_pdf(self, filepath: Path) -> str:
        reader = PdfReader(str(filepath))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()

    def _load_txt(self, filepath: Path) -> str:
        return filepath.read_text(encoding="utf-8").strip()

    def _load_docx(self, filepath: Path) -> str:
        doc = DocxDocument(str(filepath))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs).strip()
