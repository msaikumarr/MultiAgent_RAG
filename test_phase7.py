"""
Phase 7 Test Harness
====================
Validates LangGraph State Schema, StateGraph node transitions,
message immutability, and state flow from START to END.
"""

import time
from typing import Dict, Any
from agents.state import SynthesisGraphState, VerifiedClaim
from agents.graph import create_agent_graph

def mock_retrieval(state: SynthesisGraphState) -> Dict[str, Any]:
    print("  -> Executing [Retrieval Agent Node]...")
    return {
        "processed_query": state.get("user_query", "").strip(),
        "retrieved_blocks": [{"source": "paper_1.pdf", "page": 1, "text": "Dense embeddings outperform BM25."}],
        "unique_sources": ["paper_1.pdf"],
        "retrieval_latency_ms": 12.5,
        "execution_trace": [{"node": "retrieval_agent", "timestamp": time.time()}]
    }

def mock_analysis(state: SynthesisGraphState) -> Dict[str, Any]:
    print("  -> Executing [Analysis Agent Node]...")
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "analysis_agent", "timestamp": time.time()})
    return {
        "analyzed_documents": [{"source": "paper_1.pdf", "findings": ["High MRR on dense retrieval"]}],
        "execution_trace": trace
    }

def mock_comparison(state: SynthesisGraphState) -> Dict[str, Any]:
    print("  -> Executing [Comparison Agent Node]...")
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "comparison_agent", "timestamp": time.time()})
    return {
        "comparison_result": {"differences": ["Dense vs Sparse trade-offs"]},
        "execution_trace": trace
    }

def mock_verification(state: SynthesisGraphState) -> Dict[str, Any]:
    print("  -> Executing [Verification Agent Node]...")
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "verification_agent", "timestamp": time.time()})
    
    claim = VerifiedClaim(
        claim_id="C1",
        claim_text="Dense embeddings outperform BM25 by 14.2% MRR.",
        source_paper="paper_1.pdf",
        page=1,
        status="SUPPORTED",
        evidence_quote="achieves a 14.2% improvement in Mean Reciprocal Rank",
        confidence=0.98,
        rationale="Exact textual match in abstract."
    )
    return {
        "extracted_claims": [claim.to_dict()],
        "verification_results": [claim.to_dict()],
        "supported_claims_count": 1,
        "unsupported_claims_count": 0,
        "faithfulness_score": 1.0,
        "execution_trace": trace
    }

def mock_synthesis(state: SynthesisGraphState) -> Dict[str, Any]:
    print("  -> Executing [Synthesis Agent Node]...")
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "synthesis_agent", "timestamp": time.time()})
    return {
        "final_synthesis": "Dense retrieval architectures provide superior semantic capture compared to BM25 [Source 1].",
        "source_citations": [{"citation_index": 1, "source": "paper_1.pdf", "page": 1}],
        "execution_trace": trace
    }


def test_graph_architecture():
    print("1. Compiling Multi-Agent LangGraph StateGraph...")
    app = create_agent_graph(
        retrieval_fn=mock_retrieval,
        analysis_fn=mock_analysis,
        comparison_fn=mock_comparison,
        verification_fn=mock_verification,
        synthesis_fn=mock_synthesis
    )
    assert app is not None, "Failed to compile LangGraph application"

    print("\n2. Executing State Transition Test...")
    initial_state: SynthesisGraphState = {
        "user_query": "Compare dense retrieval with BM25 in RAG.",
        "top_k": 3,
        "execution_trace": []
    }

    t0 = time.time()
    final_output = app.invoke(initial_state)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"\nGraph Execution Finished in {elapsed_ms:.2f} ms")
    print(f"Final State Keys: {list(final_output.keys())}")
    print(f"Nodes executed in trace: {[t['node'] for t in final_output['execution_trace']]}")
    print(f"Final Synthesis Answer: {final_output['final_synthesis']}")
    print(f"Faithfulness Score: {final_output['faithfulness_score']}")

    # Validations
    assert len(final_output["execution_trace"]) == 5, f"Expected 5 trace records, got {len(final_output['execution_trace'])}"
    assert final_output["faithfulness_score"] == 1.0
    assert "paper_1.pdf" in final_output["unique_sources"]
    assert len(final_output["source_citations"]) == 1

    print("\nPhase 7 Verification Passed!")


if __name__ == "__main__":
    test_graph_architecture()
