"""
Phase 8 Test Harness
====================
Validates RetrievalAgent query intent parsing, semantic evidence retrieval,
trace logging, and LangGraph node output compliance.
"""

from agents.state import SynthesisGraphState
from agents.retrieval_agent import retrieval_node, RetrievalAgent

def test_retrieval_agent():
    print("1. Testing Query Intent Analyzer...")
    agent = RetrievalAgent()
    analysis1 = agent.analyze_query("Compare the RAG methods and hybrid search techniques.")
    assert analysis1["is_comparative"] is True
    print(f"Query 1 Analysis: Comparative={analysis1['is_comparative']}")

    analysis2 = agent.analyze_query("Summarize the main approaches discussed in these papers.")
    assert analysis2["is_summary"] is True
    print(f"Query 2 Analysis: Summary={analysis2['is_summary']}")

    print("\n2. Executing Retrieval Agent Node on State...")
    state: SynthesisGraphState = {
        "user_query": "Compare the advantages and limitations of dense retrieval versus hybrid search.",
        "top_k": 4,
        "execution_trace": []
    }

    result = retrieval_node(state)

    print("\n3. Inspecting Node Outputs:")
    print(f"Processed Query  : '{result['processed_query']}'")
    print(f"Retrieved Blocks : {len(result['retrieved_blocks'])}")
    print(f"Unique Sources   : {result['unique_sources']}")
    print(f"Latency          : {result['retrieval_latency_ms']:.2f} ms")

    assert len(result["retrieved_blocks"]) == 4, f"Expected 4 blocks, got {len(result['retrieved_blocks'])}"
    assert len(result["unique_sources"]) >= 2, f"Expected multiple sources for comparison, got {result['unique_sources']}"
    assert len(result["execution_trace"]) == 1
    assert result["execution_trace"][0]["agent"] == "RetrievalAgent"

    print("\n4. Top Retrieved Evidences Sample:")
    for b in result["retrieved_blocks"][:2]:
        print(f"  [{b['citation_index']}] {b['source']} (p.{b['page']}) [Score: {b['score']:.4f}]")
        print(f"      {b['text'][:120]}...\n")

    print("5. Testing Empty Query Error Handling...")
    err_state: SynthesisGraphState = {"user_query": "", "execution_trace": []}
    err_result = retrieval_node(err_state)
    assert "errors" in err_result and len(err_result["errors"]) > 0
    print(f"Empty query gracefully handled: {err_result['errors']}")

    print("\nPhase 8 Verification Passed!")


if __name__ == "__main__":
    test_retrieval_agent()
