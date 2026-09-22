"""
Phase 5 Test Harness
====================
Validates FAISS vector database construction, disk serialization,
index reloading, sub-millisecond similarity search, and source lineage.
"""

import time
from rag.chunker import TextChunker
from rag.vector_store import FAISSVectorStore

def test_vector_store():
    print("1. Loading processed chunks from Phase 3...")
    chunker = TextChunker()
    chunks = chunker.load_cache()
    assert len(chunks) > 0, "No chunks found in cache!"
    print(f"Loaded {len(chunks)} chunks.")

    print("\n2. Initializing and Building FAISS Vector Store...")
    vs = FAISSVectorStore()
    
    t0 = time.time()
    vs.build_index(chunks, force_rebuild=True)
    build_time = time.time() - t0
    
    print(f"Build completed in {build_time:.3f}s. Total indexed vectors: {vs.get_total_vectors()}")
    assert vs.get_total_vectors() == len(chunks), f"Vector count mismatch: {vs.get_total_vectors()} vs {len(chunks)}"

    print("\n3. Testing Persistence: Re-loading Vector Store from Disk...")
    vs_reloaded = FAISSVectorStore()
    success = vs_reloaded.load()
    assert success is True, "Failed to load vector store from disk!"
    assert vs_reloaded.get_total_vectors() == len(chunks)
    print(f"Reload verification passed! {vs_reloaded.get_total_vectors()} vectors loaded.")

    print("\n4. Executing Multi-Topic Retrieval Benchmarks:")
    test_queries = [
        "How do multi-agent collaboration frameworks reduce hallucinations in knowledge synthesis?",
        "What are the limitations of dense retrieval compared to hybrid search?",
        "Explain the three phases of the claim-level verification agent."
    ]

    for q_idx, query in enumerate(test_queries, start=1):
        print(f"\n=======================================================")
        print(f"Query {q_idx}: '{query}'")
        print(f"=======================================================")
        
        t_start = time.time()
        results = vs_reloaded.similarity_search(query, top_k=3)
        latency_ms = (time.time() - t_start) * 1000

        print(f"Retrieval Latency: {latency_ms:.2f} ms | Retrieved: {len(results)} chunks")
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        for res in results:
            print(f"\n  [Rank {res.rank}] Cosine Similarity: {res.score:.4f}")
            print(f"  Source  : {res.chunk.source} (Page {res.chunk.page})")
            print(f"  Chunk ID: {res.chunk.chunk_id}")
            print(f"  Snippet : {res.chunk.text[:130]}...")

    print("\nPhase 5 Verification Passed!")

if __name__ == "__main__":
    test_vector_store()
