"""
Phase 11 Test Harness
=====================
Validates VerificationAgent claim decomposition, NLI evidence alignment,
hallucination detection, confidence calibration, and faithfulness scoring.
"""

from agents.state import SynthesisGraphState
from agents.retrieval_agent import retrieval_node
from agents.analysis_agent import analysis_node
from agents.comparison_agent import comparison_node
from agents.verification_agent import verification_node, VerificationAgent

def test_verification_agent():
    print("1. Running Multi-Node Pipeline: Retrieval -> Analysis -> Comparison -> Verification...")
    initial_state: SynthesisGraphState = {
        "user_query": "What empirical improvements does hybrid search provide over dense retrieval and BM25?",
        "top_k": 4,
        "execution_trace": []
    }

    # Step 1: Retrieval
    s1 = {**initial_state, **retrieval_node(initial_state)}
    # Step 2: Analysis
    s2 = {**s1, **analysis_node(s1)}
    # Step 3: Comparison
    s3 = {**s2, **comparison_node(s2)}
    # Step 4: Verification
    s4 = {**s3, **verification_node(s3)}

    verified_claims = s4["verification_results"]
    faithfulness = s4["faithfulness_score"]

    print(f"\n2. Verification Summary:")
    print(f"Total Claims Audited : {len(verified_claims)}")
    print(f"Supported Claims     : {s4['supported_claims_count']}")
    print(f"Unsupported Claims   : {s4['unsupported_claims_count']}")
    print(f"Faithfulness Score   : {faithfulness*100:.1f}%")

    assert len(verified_claims) > 0, "No claims were audited!"
    assert faithfulness >= 0.70, f"Expected high faithfulness on genuine extractions, got {faithfulness}"

    print("\n--- [Detailed Claim Verification Audit Sample] ---")
    for vc in verified_claims[:4]:
        print(f"\nClaim ID  : {vc['claim_id']} [{vc['status']}] (Confidence: {vc['confidence']})")
        print(f"Claim Text: {vc['claim_text']}")
        print(f"Source    : {vc['source_paper']} (Page {vc['page']})")
        print(f"Evidence  : \"{vc['evidence_quote']}\"")
        print(f"Rationale : {vc['rationale']}")

    print("\n3. Testing Hallucination Injection Test (Negative Control)...")
    agent = VerificationAgent()
    fake_claim = {
        "claim_id": "FAKE_001",
        "claim_text": "The framework achieved 99.9% accuracy on quantum cryptographic benchmarks using Llama-9.",
        "source_paper": "paper_4_hybrid_search_evaluation.pdf",
        "page": 1,
        "claim_type": "hallucination"
    }
    audit_res = agent.verify_single_claim(fake_claim, s4["retrieved_blocks"])
    print(f"Injected Fake Claim Status: {audit_res.status} (Confidence: {audit_res.confidence})")
    print(f"Rationale: {audit_res.rationale}")
    assert audit_res.status == "UNSUPPORTED", f"Expected fake claim to be flagged UNSUPPORTED, got {audit_res.status}"
    print("Hallucination injection successfully caught and rejected!")

    assert len(s4["execution_trace"]) == 4
    assert s4["execution_trace"][3]["agent"] == "VerificationAgent"

    print("\nPhase 11 Verification Passed!")


if __name__ == "__main__":
    test_verification_agent()
