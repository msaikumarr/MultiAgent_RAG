"""
Phase 9 Test Harness
====================
Validates AnalysisAgent per-document analytical extraction, quantitative metrics
parsing, advantage/limitation extraction, and state integration.
"""

from agents.state import SynthesisGraphState
from agents.retrieval_agent import retrieval_node
from agents.analysis_agent import analysis_node, AnalysisAgent

def test_analysis_agent():
    print("1. Running Retrieval Node to obtain state...")
    initial_state: SynthesisGraphState = {
        "user_query": "What are the empirical results, advantages, and limitations of dense retrieval and hybrid search?",
        "top_k": 4,
        "execution_trace": []
    }
    retrieval_output = retrieval_node(initial_state)
    state = {**initial_state, **retrieval_output}
    print(f"Retrieved {len(state['retrieved_blocks'])} blocks from {state['unique_sources']}.")

    print("\n2. Executing Analysis Agent Node...")
    analysis_output = analysis_node(state)
    analyzed_docs = analysis_output["analyzed_documents"]

    print(f"Analyzed {len(analyzed_docs)} documents.")
    assert len(analyzed_docs) > 0, "No documents analyzed!"

    print("\n3. Inspecting Per-Document Analysis Results:")
    for doc in analyzed_docs:
        print(f"\n=======================================================")
        print(f"Document       : {doc['source_paper']} (Pages: {doc['page_numbers']})")
        print(f"Citations      : [Source {', '.join(map(str, doc['citation_indices']))}]")
        print(f"Proposed Method: {doc['proposed_method']}")
        print(f"Empirical Metrics: {doc['empirical_results']}")
        print(f"Key Findings   : {doc['key_findings']}")
        print(f"Advantages     : {doc['advantages']}")
        print(f"Limitations    : {doc['limitations']}")
        print(f"Evidence Quotes: {len(doc['evidence_quotes'])} quotes captured")
        print(f"=======================================================")

        assert len(doc["key_findings"]) > 0
        assert len(doc["advantages"]) > 0
        assert doc["source_paper"] in state["unique_sources"]

    assert len(analysis_output["execution_trace"]) == 2
    assert analysis_output["execution_trace"][1]["agent"] == "AnalysisAgent"
    print("\nPhase 9 Verification Passed!")


if __name__ == "__main__":
    test_analysis_agent()
