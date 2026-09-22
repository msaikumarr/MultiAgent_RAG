"""
Phase 12 Test Harness
=====================
Validates End-to-End Multi-Agent Knowledge Synthesis:
Retrieval -> Analysis -> Comparison -> Verification -> Synthesis.
"""

from agents.state import SynthesisGraphState
from agents.graph import create_agent_graph
from agents.retrieval_agent import retrieval_node
from agents.analysis_agent import analysis_node
from agents.comparison_agent import comparison_node
from agents.verification_agent import verification_node
from agents.synthesis_agent import synthesis_node

def test_full_agentic_pipeline():
    print("1. Assembling & Compiling Full LangGraph Pipeline...")
    app = create_agent_graph(
        retrieval_fn=retrieval_node,
        analysis_fn=analysis_node,
        comparison_fn=comparison_node,
        verification_fn=verification_node,
        synthesis_fn=synthesis_node
    )

    query = "Compare the architectures, empirical benchmarks, advantages, and limitations of dense retrieval versus hybrid search."
    print(f"\n2. Executing End-to-End Synthesis on Query:\n   \"{query}\"")

    initial_state: SynthesisGraphState = {
        "user_query": query,
        "top_k": 4,
        "execution_trace": []
    }

    final_output = app.invoke(initial_state)

    print("\n=======================================================")
    print("FINAL EVIDENCE-GROUNDED SYNTHESIS REPORT")
    print("=======================================================\n")
    print(final_output["final_synthesis"])
    print("\n=======================================================")

    # Validations
    report = final_output["final_synthesis"]
    assert "## 1. Executive Summary" in report
    assert "## 2. Detailed Methodological Analysis" in report
    assert "## 3. Cross-Document Comparative Synthesis" in report
    assert "## 4. Grounding & Factual Verification Audit" in report
    assert "## 5. Source Citations & Provenance Registry" in report
    assert "[Source 1]" in report
    assert len(final_output["source_citations"]) == 4
    assert len(final_output["execution_trace"]) == 5

    print(f"\nPipeline Execution Trace ({len(final_output['execution_trace'])} agents executed):")
    for step in final_output["execution_trace"]:
        print(f"  • {step['agent']}: {step['action']} (Latency: {step.get('latency_ms', 0):.2f} ms)")

    print(f"\nFaithfulness Score: {final_output['faithfulness_score']*100:.1f}%")
    print("\nPhase 12 Verification Passed!")


if __name__ == "__main__":
    test_full_agentic_pipeline()
