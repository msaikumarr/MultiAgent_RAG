"""
Phase 4 Test Harness
====================
Validates Sentence Transformers embedding generation, vector dimensionality,
L2 normalization, batch processing throughput, and semantic ranking sanity.
"""

import time
import numpy as np
from rag.chunker import TextChunker
from models.embeddings import EmbeddingModel

def test_embeddings():
    print("1. Loading processed chunks from Phase 3...")
    chunker = TextChunker()
    chunks = chunker.load_cache()
    assert len(chunks) > 0, "No chunks found in cache!"
    print(f"Loaded {len(chunks)} chunks.")

    print("\n2. Initializing EmbeddingModel (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = EmbeddingModel()
    
    # Test single query embedding
    query = "What is the Mean Reciprocal Rank improvement of dense retrieval over BM25?"
    print(f"\n3. Testing Single Query Embedding:")
    print(f"Query: '{query}'")
    
    t0 = time.time()
    query_vec = embedder.embed_text(query)
    q_time = (time.time() - t0) * 1000
    
    print(f"Vector Shape : {query_vec.shape}")
    print(f"Data Type    : {query_vec.dtype}")
    norm = float(np.linalg.norm(query_vec))
    print(f"L2 Norm      : {norm:.4f} (Expected: ~1.0000)")
    print(f"Latency      : {q_time:.2f} ms")
    
    assert query_vec.shape == (384,), f"Expected (384,), got {query_vec.shape}"
    assert abs(norm - 1.0) < 1e-3, f"Vector not L2 normalized: {norm}"

    # Test batch chunk embedding
    print(f"\n4. Testing Batch Embedding on all {len(chunks)} chunks...")
    t0 = time.time()
    chunk_vecs = embedder.embed_chunks(chunks)
    batch_time = time.time() - t0
    
    print(f"Batch Matrix Shape : {chunk_vecs.shape}")
    print(f"Batch Encoding Time: {batch_time:.3f} s ({len(chunks)/batch_time:.1f} chunks/sec)")
    assert chunk_vecs.shape == (len(chunks), 384)

    # Test semantic similarity calculation
    print("\n5. Testing Semantic Cosine Similarity & Ranking Sanity:")
    similarities = np.dot(chunk_vecs, query_vec)
    top_indices = np.argsort(similarities)[::-1][:5]
    
    print(f"Top-5 matching chunks for query: '{query}'\n")
    top_sources = []
    for rank, idx in enumerate(top_indices, start=1):
        c = chunks[idx]
        score = float(similarities[idx])
        top_sources.append(c.source)
        print(f"Rank {rank} [Score: {score:.4f}]: {c.chunk_id} (p.{c.page})")
        print(f"Snippet: {c.text[:140]}...\n")
        assert -1.0 <= score <= 1.0001, f"Cosine score out of bounds: {score}"

    # Verify that the top retrieved chunks come from relevant literature on dense retrieval & BM25
    assert any("paper_1" in src or "paper_4" in src for src in top_sources), "Expected relevant papers in top results"
    print("Semantic ranking sanity test passed!")
    print("\nPhase 4 Verification Passed!")

if __name__ == "__main__":
    test_embeddings()
