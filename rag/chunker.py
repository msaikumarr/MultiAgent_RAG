"""
Text Chunking Module
====================
Splits extracted document pages into semantically cohesive,
overlapping chunks with persistent lineage tracking.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

from config import config
from rag.document_processor import DocumentPage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Represents an atomic chunk of text with exact document and page lineage.
    """
    chunk_id: str               # Deterministic ID: e.g. "paper_1.pdf_p1_c0"
    source: str                 # Document filename
    page: int                   # 1-indexed page number
    chunk_index: int            # Ordinal index within the page
    text: str                   # Chunk text content
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
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


class TextChunker:
    """
    Splits document text using a recursive hierarchical boundary splitting algorithm
    (paragraphs -> sentences -> clauses -> words) with sliding window overlap.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_length: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size or config.document.chunk_size
        self.chunk_overlap = chunk_overlap or config.document.chunk_overlap
        self.min_chunk_length = min_chunk_length or config.document.min_chunk_length
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be strictly smaller than chunk_size ({self.chunk_size})")

    def _split_text_recursively(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using the highest-priority separator that fits chunks into chunk_size.
        """
        final_chunks: List[str] = []
        if not text.strip():
            return final_chunks

        # Find the best separator available in text
        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        # Merge splits into chunks with sliding overlap
        good_splits: List[str] = []
        for s in splits:
            if s.strip():
                good_splits.append(s.strip())

        current_chunk: List[str] = []
        current_len = 0

        for split_part in good_splits:
            part_len = len(split_part) + (len(separator) if current_chunk else 0)

            # If a single atomic split piece exceeds chunk_size, split it further
            if part_len > self.chunk_size:
                if current_chunk:
                    doc = separator.join(current_chunk).strip()
                    if doc:
                        final_chunks.append(doc)
                    current_chunk = []
                    current_len = 0

                if new_separators:
                    sub_chunks = self._split_text_recursively(split_part, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    # Character slice fallback
                    for start in range(0, len(split_part), self.chunk_size - self.chunk_overlap):
                        slice_text = split_part[start:start + self.chunk_size].strip()
                        if slice_text:
                            final_chunks.append(slice_text)
                continue

            if current_len + part_len <= self.chunk_size:
                current_chunk.append(split_part)
                current_len += part_len
            else:
                if current_chunk:
                    doc = separator.join(current_chunk).strip()
                    if doc:
                        final_chunks.append(doc)

                # Overlap logic: keep trailing pieces that fit into chunk_overlap
                overlap_chunk: List[str] = []
                overlap_len = 0
                for part in reversed(current_chunk):
                    if overlap_len + len(part) <= self.chunk_overlap:
                        overlap_chunk.insert(0, part)
                        overlap_len += len(part)
                    else:
                        break

                current_chunk = overlap_chunk + [split_part]
                current_len = sum(len(p) for p in current_chunk) + len(separator) * max(0, len(current_chunk) - 1)

        if current_chunk:
            doc = separator.join(current_chunk).strip()
            if doc:
                final_chunks.append(doc)

        return final_chunks

    def chunk_page(self, page: DocumentPage) -> List[DocumentChunk]:
        """
        Splits a single DocumentPage into a list of DocumentChunk objects.
        """
        raw_chunks = self._split_text_recursively(page.text, self.separators)
        chunks: List[DocumentChunk] = []

        chunk_idx = 0
        for text_chunk in raw_chunks:
            cleaned = text_chunk.strip()
            if len(cleaned) < self.min_chunk_length:
                continue

            chunk_id = f"{page.source}_p{page.page}_c{chunk_idx}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                source=page.source,
                page=page.page,
                chunk_index=chunk_idx,
                text=cleaned,
                metadata={
                    **page.metadata,
                    "parent_char_count": page.char_count,
                    "parent_word_count": page.word_count
                }
            )
            chunks.append(chunk)
            chunk_idx += 1

        return chunks

    def chunk_all_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """
        Chunks an entire list of DocumentPages across all documents.
        """
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)

        logger.info(
            f"Chunking complete: Transformed {len(pages)} pages into {len(all_chunks)} semantic chunks "
            f"(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})."
        )
        return all_chunks

    def save_cache(self, chunks: List[DocumentChunk], cache_filename: str = "processed_chunks.json") -> Path:
        """Saves generated chunks to JSON cache for inspection and vector indexing."""
        cache_path = config.document.processed_docs_dir / cache_filename
        data = [c.to_dict() for c in chunks]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to cache at: {cache_path}")
        return cache_path

    def load_cache(self, cache_filename: str = "processed_chunks.json") -> List[DocumentChunk]:
        """Loads chunks from JSON cache if present."""
        cache_path = config.document.processed_docs_dir / cache_filename
        if not cache_path.exists():
            logger.warning(f"Chunk cache file not found at {cache_path}")
            return []
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [DocumentChunk.from_dict(item) for item in data]


if __name__ == "__main__":
    chunker = TextChunker()
    print(f"TextChunker initialized: chunk_size={chunker.chunk_size}, overlap={chunker.chunk_overlap}")
