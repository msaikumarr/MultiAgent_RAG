"""
Phase 10 Test Harness
=====================
Validates ComparisonAgent cross-document synthesis, comparative matrix formulation,
trade-off analysis, and multi-agent pipeline integration.
"""

from agents.state import SynthesisGraphState
from agents.retrieval_agent import retrieval_node
from agents.analysis_agent import analysis_node
from agents.comparison_agent import comparison_node, ComparisonAgent

def test_comparison_agent():
    print("1. Running Multi-Node Pipeline: Retrieval -> Analysis -> Comparison...")
    initial_state: SynthesisGraphState = {
        "user_query": "Compare the architectures, empirical performance, and limitations across dense retrieval and hybrid search papers.",
        "top_k": 4,
        "execution_trace": []
    }

    # Step 1: Retrieval
    retrieval_out = retrieval_node(initial_state)
    state_after_retrieval = {**initial_state, **retrieval_out}

    # Step 2: Analysis
    analysis_out = analysis_node(state_after_retrieval)
    state_after_analysis = {**state_after_retrieval, **analysis_out}

    # Step 3: Comparison
    comparison_out = comparison_node(state_after_analysis)
    final_state = {**state_after_analysis, **comparison_out}

    comp_res = final_state["comparison_result"]
    matrix = final_state["comparative_matrix"]

    print(f"\n2. Comparison Status: Multi-Document = {comp_res['is_multi_document']}")
    print(f"Total Compared Papers: {len(matrix)}")

    print("\n--- [Cross-Document Comparative Matrix] ---")
    for row in matrix:
        print(f"\nPaper       : {row['paper']}")
        print(f"Architecture: {row['architecture']}")
        print(f"Key Metrics : {row['primary_metrics']}")
        print(f"Advantage   : {row['main_advantage']}")
        print(f"Limitation  : {row['main_limitation']}")

    print("\n--- [Synthesized Trade-offs & Differences] ---")
    print("Methodological Differences:")
    for diff in comp_res["methodological_differences"]:
        print(f"  • {diff}")

    print("\nIdentified Engineering Trade-offs:")
    for to in comp_res["tradeoffs"]:
        print(f"  • {to}")

    print("\nIdentified Research Gaps:")
    for rg in comp_res["research_gaps"]:
        print(f"  • {rg}")

    # Validations
    assert len(matrix) >= 2, f"Expected multi-paper matrix, got {len(matrix)}"
    assert len(comp_res["tradeoffs"]) > 0
    assert len(comp_res["common_themes"]) > 0
    assert len(final_state["execution_trace"]) == 3
    assert final_state["execution_trace"][2]["agent"] == "ComparisonAgent"

    print("\nPhase 10 Verification Passed!")


if __name__ == "__main__":
    test_comparison_agent()
