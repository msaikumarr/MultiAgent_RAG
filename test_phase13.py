"""
Phase 13 Test Harness
=====================
Validates Multi-Model Architecture Integration:
- Model 1: Bi-Encoder Dense Embeddings (Sentence Transformers)
- Model 2: Primary LLM (Generation, Analysis & Synthesis)
- Model 3: Secondary LLM (High-Precision Verification & Auditing)
"""

import json
from models.llm import MultiModelManager, multi_model_manager
from models.embeddings import embedding_model

def test_multi_model_integration():
    print("1. Inspecting Multi-Model Framework Inventory...")
    inventory = multi_model_manager.get_model_inventory()
    print(json.dumps(inventory, indent=2))

    assert "model_1_embeddings" in inventory
    assert "model_2_primary_llm" in inventory
    assert "model_3_verification_llm" in inventory

    print("\n2. Testing Model 1 (Dense Bi-Encoder Semantic Space)...")
    vec = embedding_model.embed_text("Multi-model verification in generative AI.")
    print(f"Model 1 Vector: shape={vec.shape}, dimension={embedding_model.vector_dimension}")
    assert vec.shape == (384,)

    print("\n3. Testing Model 2 (Primary Reasoning & Synthesis LLM)...")
    primary_llm = multi_model_manager.get_primary_llm()
    prompt_test = "Summarize the key advantages of multi-agent decomposition."
    primary_out = primary_llm.generate(prompt_test)
    print(f"Primary Model Class: {type(primary_llm).__name__}")
    print(f"Primary Output Sample: {primary_out[:120]}...")
    assert len(primary_out) > 0

    print("\n4. Testing Model 3 (Secondary Audit & Verification LLM)...")
    verification_llm = multi_model_manager.get_verification_llm()
    audit_prompt = "Audit statement: Dense retrieval improves MRR by 14.2%."
    verification_out = verification_llm.generate(audit_prompt)
    print(f"Verification Model Class: {type(verification_llm).__name__}")
    print(f"Verification Output Sample: {verification_out[:120]}...")
    assert len(verification_out) > 0

    print("\n5. Testing Structured JSON Parsing Interface...")
    json_prompt = "Output a valid JSON test object with keys 'status' and 'model_verified'."
    # Test fallback structured generation
    class MockJsonLLM(type(primary_llm)):
        def generate(self, prompt, **kwargs):
            return "```json\n{\"status\": \"SUCCESS\", \"model_verified\": true, \"models_count\": 3}\n```"
    mock_llm = MockJsonLLM("test")
    parsed = mock_llm.generate_json(json_prompt)
    print(f"Parsed JSON Result: {parsed}")
    assert parsed["status"] == "SUCCESS"
    assert parsed["models_count"] == 3

    print("\nPhase 13 Verification Passed!")


if __name__ == "__main__":
    test_multi_model_integration()
