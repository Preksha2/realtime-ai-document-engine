"""
Text preprocessing utilities for cleaning raw document text
before chunking and embedding.
"""
import re
from typing import List


class TextPreprocessor:
    """Cleans and normalizes raw text extracted from documents."""

    @staticmethod
    def clean(text: str) -> str:
        """Apply all cleaning steps to raw text."""
        text = TextPreprocessor._normalize_whitespace(text)
        text = TextPreprocessor._remove_headers_footers(text)
        text = TextPreprocessor._fix_encoding_artifacts(text)
        return text.strip()

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text

    @staticmethod
    def _remove_headers_footers(text: str) -> str:
        """Remove common page header/footer patterns."""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip page numbers
            if re.match(r'^(Page\s+)?\d+(\s+of\s+\d+)?$', stripped, re.IGNORECASE):
                continue
            # Skip lines that are just dashes or underscores (separators)
            if re.match(r'^[-_=]{5,}$', stripped):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    @staticmethod
    def _fix_encoding_artifacts(text: str) -> str:
        """Replace common encoding issues."""
        replacements = {
            '\u2019': "'",
            '\u2018': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u2013': '-',
            '\u2014': '--',
            '\u2026': '...',
            '\u00a0': ' ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def batch_clean(documents: List[dict]) -> List[dict]:
        """Clean text for a list of documents in place."""
        for doc in documents:
            doc["text"] = TextPreprocessor.clean(doc["text"])
        return documents
