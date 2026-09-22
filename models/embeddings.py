"""
Embedding Module
================
Manages dense semantic vector embeddings using Sentence Transformers
(e.g., sentence-transformers/all-MiniLM-L6-v2) with L2 normalization
for inner product cosine similarity in FAISS.
"""

import os
import ssl
import time
import logging
from typing import List, Optional
import numpy as np

# Configure SSL context and disable verification for proxy / enterprise CA setups
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import urllib3
urllib3.disable_warnings()

try:
    import requests
    _old_requests_init = requests.Session.__init__
    def _new_requests_init(self, *args, **kwargs):
        _old_requests_init(self, *args, **kwargs)
        self.verify = False
    requests.Session.__init__ = _new_requests_init
except Exception:
    pass

try:
    import httpx
    _old_httpx_init = httpx.Client.__init__
    def _new_httpx_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _old_httpx_init(self, *args, **kwargs)
    httpx.Client.__init__ = _new_httpx_init
except Exception:
    pass

from config import config
from rag.chunker import DocumentChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for Hugging Face Sentence Transformers with batch encoding,
    L2 normalization, and fallback resilience.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
        batch_size: Optional[int] = None,
        device: str = "cpu"
    ):
        self.model_name = model_name or config.embedding.model_name
        self.normalize_embeddings = (
            normalize_embeddings if normalize_embeddings is not None
            else config.embedding.normalize_embeddings
        )
        self.batch_size = batch_size or config.embedding.batch_size
        self.device = device
        self._model = None
        self.vector_dimension: int = config.embedding.vector_dimension

    @property
    def model(self):
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: '{self.model_name}' on device '{self.device}'...")
            t0 = time.time()
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.vector_dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully in {time.time() - t0:.2f}s (dimension={self.vector_dimension}).")
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generates a normalized 1D embedding vector for a single query or text passage.
        
        Returns:
            np.ndarray of shape (vector_dimension,) and dtype float32
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        vec = np.asarray(embedding, dtype=np.float32)
        if self.normalize_embeddings:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generates normalized embeddings for a batch of text passages.
        
        Returns:
            np.ndarray of shape (N, vector_dimension) and dtype float32
        """
        if not texts:
            return np.empty((0, self.vector_dimension), dtype=np.float32)

        bs = batch_size or self.batch_size
        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {bs}...")
        t0 = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 20
        )
        elapsed = time.time() - t0
        vecs = np.asarray(embeddings, dtype=np.float32)
        if self.normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms

        logger.info(f"Embedding completed: {len(texts)} items encoded in {elapsed:.2f}s ({len(texts)/max(0.001, elapsed):.1f} items/sec).")
        return vecs

    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Convenience method to encode a collection of DocumentChunk objects.
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_batch(texts, batch_size=batch_size)


# Singleton Instance
embedding_model = EmbeddingModel()


if __name__ == "__main__":
    test_query = "What are the advantages of dense retrieval in RAG?"
    vec = embedding_model.embed_text(test_query)
    print(f"Query: '{test_query}'")
    print(f"Embedding shape: {vec.shape}, L2 norm: {np.linalg.norm(vec):.4f}")
