"""
FAISS Vector Store Module
=========================
Manages FAISS (Facebook AI Similarity Search) index creation, persistence,
metadata alignment, and top-k vector similarity search for RAG.
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from config import config
from rag.chunker import DocumentChunk
from models.embeddings import EmbeddingModel, embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a retrieved chunk with similarity score and ranking."""
    chunk: DocumentChunk
    score: float                # Cosine similarity score (higher is better)
    rank: int                   # 1-based rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "page": self.chunk.page,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata
        }


class FAISSVectorStore:
    """
    High-performance vector database wrapper for FAISS IndexFlatIP (Cosine Similarity).
    Supports persistent disk serialization and atomic metadata alignment.
    """

    def __init__(
        self,
        store_dir: Optional[Path] = None,
        embedder: Optional[EmbeddingModel] = None,
        dimension: Optional[int] = None
    ):
        self.store_dir = Path(store_dir or config.vector_store.store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedder = embedder or embedding_model
        self.dimension = dimension or config.embedding.vector_dimension

        self.index_file = self.store_dir / config.vector_store.index_file_name
        self.metadata_file = self.store_dir / config.vector_store.metadata_file_name

        self.index: Optional[Any] = None
        self.chunks: List[DocumentChunk] = []

    def is_indexed(self) -> bool:
        """Returns True if a persistent FAISS index and metadata exist on disk."""
        return self.index_file.exists() and self.metadata_file.exists()

    def get_total_vectors(self) -> int:
        """Returns the total number of indexed vectors."""
        return self.index.ntotal if self.index is not None else 0

    def build_index(
        self,
        chunks: List[DocumentChunk],
        force_rebuild: bool = False,
        save_after_build: bool = True
    ) -> None:
        """
        Builds a new FAISS IndexFlatIP from document chunks and saves to disk.
        """
        if not chunks:
            logger.warning("No chunks provided to build_index.")
            return

        if not force_rebuild and self.is_indexed():
            logger.info("Found existing FAISS index on disk. Loading without rebuilding...")
            if self.load():
                return

        if faiss is None:
            raise ImportError("faiss-cpu is not installed. Please install it with 'pip install faiss-cpu'.")

        logger.info(f"Building FAISS index for {len(chunks)} chunks (dimension={self.dimension})...")
        t0 = time.time()

        # Step 1: Generate normalized embeddings
        embeddings = self.embedder.embed_chunks(chunks)
        assert embeddings.shape[1] == self.dimension, f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.dimension}"

        # Step 2: Initialize FAISS Inner Product Index (Equivalent to Cosine on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks = chunks

        elapsed = time.time() - t0
        logger.info(f"FAISS index built successfully: {self.get_total_vectors()} vectors indexed in {elapsed:.2f}s.")

        if save_after_build:
            self.save()

    def save(self) -> None:
        """Serializes FAISS binary index and chunks metadata to disk."""
        if self.index is None or not self.chunks:
            logger.warning("Cannot save empty vector store.")
            return

        # Save FAISS binary
        faiss.write_index(self.index, str(self.index_file))
        
        # Save metadata JSON
        metadata = [chunk.to_dict() for chunk in self.chunks]
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved FAISS index to '{self.index_file}' and metadata ({len(self.chunks)} items) to '{self.metadata_file}'.")

    def load(self) -> bool:
        """Loads FAISS binary index and metadata from disk."""
        if not self.is_indexed():
            logger.warning(f"Vector store files not found in {self.store_dir}")
            return False

        if faiss is None:
            raise ImportError("faiss-cpu is not installed.")

        try:
            logger.info(f"Loading FAISS index from {self.index_file}...")
            self.index = faiss.read_index(str(self.index_file))

            with open(self.metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.chunks = [DocumentChunk.from_dict(item) for item in metadata]

            logger.info(f"Loaded FAISS index with {self.get_total_vectors()} vectors and {len(self.chunks)} chunks.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Executes semantic similarity search for a natural language query.
        
        Args:
            query: User search query
            top_k: Number of nearest neighbors to retrieve (default from config)
            score_threshold: Optional minimum cosine similarity cutoff
            
        Returns:
            Ranked list of SearchResult objects
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to similarity_search.")
            return []

        if self.index is None or self.get_total_vectors() == 0:
            if not self.load():
                logger.error("Vector store is not initialized or indexed.")
                return []

        k = min(top_k or config.vector_store.top_k, self.get_total_vectors())
        query_vec = self.embedder.embed_text(query).reshape(1, -1)

        distances, indices = self.index.search(query_vec, k)

        results: List[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            score_val = float(score)
            if score_threshold is not None and score_val < score_threshold:
                continue

            result = SearchResult(
                chunk=self.chunks[idx],
                score=score_val,
                rank=rank
            )
            results.append(result)

        return results


# Singleton Vector Store Instance
vector_store = FAISSVectorStore()


if __name__ == "__main__":
    print(f"FAISSVectorStore initialized. Store dir: {vector_store.store_dir}")
    print(f"Is indexed: {vector_store.is_indexed()}")
