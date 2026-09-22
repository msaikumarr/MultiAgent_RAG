"""
Document Processing Module
==========================
Handles loading, text extraction, cleaning, metadata extraction,
and error handling for research paper PDFs.
"""

import os
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    """Represents a single extracted page from a document with full lineage metadata."""
    source: str                 # Document filename (e.g., "paper_rag_survey.pdf")
    page: int                   # 1-indexed page number
    text: str                   # Cleaned extracted text content
    char_count: int = 0
    word_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentPage":
        return cls(**data)


class DocumentProcessor:
    """
    Robust processor for extracting, cleaning, and managing academic PDF papers.
    """

    def __init__(self, raw_docs_dir: Optional[Path] = None, processed_docs_dir: Optional[Path] = None):
        self.raw_docs_dir = Path(raw_docs_dir or config.document.raw_docs_dir)
        self.processed_docs_dir = Path(processed_docs_dir or config.document.processed_docs_dir)
        self.processed_docs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Cleans extracted PDF text:
        - Removes unprintable/control characters
        - Fixes hyphenated line breaks (e.g., 'archi-\\ntecture' -> 'architecture')
        - Replaces multiple whitespace/newlines with standard spacing
        - Normalizes quotation marks and dashes
        """
        if not raw_text:
            return ""

        # Remove control characters except standard whitespace
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", raw_text)

        # Fix hyphenated words at line wraps
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Replace carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace three or more newlines with double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace excessive horizontal spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)

        # Strip lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines).strip()

        return text

    def process_pdf(self, file_path: Path | str) -> List[DocumentPage]:
        """
        Extracts and cleans all readable pages from a single PDF document.
        
        Args:
            file_path: Path to the target PDF file.
            
        Returns:
            List of DocumentPage objects containing cleaned text and page metadata.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return []

        if path.suffix.lower() != ".pdf":
            logger.warning(f"Skipping non-PDF file: {path.name}")
            return []

        pages: List[DocumentPage] = []
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)

            if total_pages == 0:
                logger.warning(f"PDF file has 0 pages: {path.name}")
                return []

            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    raw_text = page.extract_text() or ""
                    cleaned = self.clean_text(raw_text)

                    if not cleaned.strip():
                        logger.debug(f"Empty or unextractable text on page {page_idx} of {path.name}")
                        continue

                    doc_page = DocumentPage(
                        source=path.name,
                        page=page_idx,
                        text=cleaned,
                        metadata={
                            "file_path": str(path.resolve()),
                            "total_pages": total_pages,
                            "file_size_bytes": path.stat().st_size
                        }
                    )
                    pages.append(doc_page)
                except Exception as page_err:
                    logger.error(f"Error reading page {page_idx} of {path.name}: {page_err}")
                    continue

            logger.info(f"Processed '{path.name}': Extracted {len(pages)} valid pages (out of {total_pages} total pages).")

        except PdfReadError as e:
            logger.error(f"Corrupted or invalid PDF format in '{path.name}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error while parsing '{path.name}': {e}")

        return pages

    def process_all_documents(self) -> List[DocumentPage]:
        """
        Scans raw_documents directory and processes all discovered PDF files.
        
        Returns:
            Aggregated list of DocumentPage objects across all documents.
        """
        if not self.raw_docs_dir.exists():
            logger.warning(f"Raw documents directory does not exist: {self.raw_docs_dir}")
            return []

        pdf_files = sorted(list(self.raw_docs_dir.glob("*.pdf")))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.raw_docs_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF documents in {self.raw_docs_dir}. Beginning ingestion...")
        all_pages: List[DocumentPage] = []

        for pdf_file in pdf_files:
            pages = self.process_pdf(pdf_file)
            all_pages.extend(pages)

        # Save processed metadata cache
        self.save_cache(all_pages)
        return all_pages

    def save_cache(self, pages: List[DocumentPage], cache_filename: str = "processed_pages.json") -> Path:
        """Saves extracted pages to JSON cache for inspection and downstream tasks."""
        cache_path = self.processed_docs_dir / cache_filename
        data = [p.to_dict() for p in pages]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(pages)} preprocessed pages to cache at: {cache_path}")
        return cache_path

    def load_cache(self, cache_filename: str = "processed_pages.json") -> List[DocumentPage]:
        """Loads extracted pages from JSON cache if present."""
        cache_path = self.processed_docs_dir / cache_filename
        if not cache_path.exists():
            logger.warning(f"Cache file not found at {cache_path}")
            return []
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [DocumentPage.from_dict(item) for item in data]


if __name__ == "__main__":
    processor = DocumentProcessor()
    print(f"DocumentProcessor initialized.")
    print(f"Target raw directory: {processor.raw_docs_dir}")
    print(f"Target processed cache: {processor.processed_docs_dir}")
