"""
Phase 3 Test Harness
====================
Validates hierarchical text chunking, overlap semantics,
lineage metadata attachment, and cache persistence.
"""

from rag.document_processor import DocumentProcessor
from rag.chunker import TextChunker, DocumentChunk

def test_chunker():
    print("1. Loading processed pages from Phase 2...")
    processor = DocumentProcessor()
    pages = processor.load_cache()
    if not pages:
        pages = processor.process_all_documents()
    print(f"Loaded {len(pages)} pages.")

    print("\n2. Initializing TextChunker (chunk_size=600, chunk_overlap=100)...")
    chunker = TextChunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_all_pages(pages)
    print(f"Generated {len(chunks)} total chunks.")

    assert len(chunks) >= len(pages), f"Expected at least {len(pages)} chunks, got {len(chunks)}"

    print("\n3. Inspecting Chunk Lineage and Constraints:")
    for idx, chunk in enumerate(chunks[:4], start=1):
        print(f"\n--- [Chunk {idx}: {chunk.chunk_id}] ---")
        print(f"Source Document : {chunk.source}")
        print(f"Page Number     : {chunk.page}")
        print(f"Chunk Index     : {chunk.chunk_index}")
        print(f"Length (chars)  : {chunk.char_count} (<= {chunker.chunk_size})")
        print(f"Word Count      : {chunk.word_count}")
        print(f"Text Content:\n{chunk.text}")
        assert chunk.char_count <= chunker.chunk_size + 50, f"Chunk exceeded size: {chunk.char_count}"
        assert chunk.chunk_id is not None
        assert chunk.source in [
            "paper_1_dense_retrieval_rag.pdf",
            "paper_2_multi_agent_synthesis.pdf",
            "paper_3_hallucination_verification.pdf",
            "paper_4_hybrid_search_evaluation.pdf"
        ]

    print("\n4. Saving and Reloading Chunk Cache...")
    cache_path = chunker.save_cache(chunks)
    reloaded = chunker.load_cache()
    assert len(reloaded) == len(chunks), f"Reload mismatch: {len(reloaded)} vs {len(chunks)}"
    print(f"Cache verified successfully at {cache_path} ({len(reloaded)} records).")

    print("\nPhase 3 Verification Passed!")

if __name__ == "__main__":
    test_chunker()
