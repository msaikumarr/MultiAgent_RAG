"""
Phase 6 Test Harness
====================
Validates citation-grounded RAG context construction, multi-document evidence
aggregation, and prompt format generation.
"""

from rag.retriever import RAGRetriever, RetrievedContext

def test_retriever():
    retriever = RAGRetriever()

    test_queries = [
        "Compare the RAG techniques and benchmarks used in these papers.",
        "What are the main causes of hallucination in academic synthesis?",
        "Which architectures combine dense embeddings with rerankers?"
    ]

    for q_idx, query in enumerate(test_queries, start=1):
        print(f"\n=======================================================")
        print(f"Query {q_idx}: '{query}'")
        print(f"=======================================================")

        context: RetrievedContext = retriever.retrieve(query, top_k=4)

        print(f"Retrieved Blocks     : {len(context.blocks)}")
        print(f"Unique Source Papers : {context.unique_sources}")
        print(f"Approx Context Words : {context.total_tokens_approx}")

        assert len(context.blocks) > 0, "No context blocks retrieved!"
        assert len(context.unique_sources) > 0, "No source documents identified!"

        print("\n--- Formatted Sources Citation Table ---")
        sources_table = context.get_sources_table()
        for src in sources_table:
            print(f"[{src['citation_index']}] {src['source']} (Page {src['page']}) | Score: {src['score']} | ID: {src['chunk_id']}")

        print("\n--- Formatted LLM Evidence Context (Sample) ---")
        print(context.formatted_context_str[:400] + "\n...[truncated]...\n")

    print("\nPhase 6 Verification Passed!")

if __name__ == "__main__":
    test_retriever()
