"""
Multi-Agent Graph State Module
==============================
Defines the strongly-typed state dictionary passed immutably across
LangGraph agent nodes in the synthesis pipeline.
"""

from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass, asdict


@dataclass
class VerifiedClaim:
    """Represents an atomic extracted claim with factual verification status."""
    claim_id: str
    claim_text: str
    source_paper: str
    page: int
    status: str                 # "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"
    evidence_quote: str
    confidence: float           # 0.0 to 1.0
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifiedClaim":
        return cls(**data)


class SynthesisGraphState(TypedDict, total=False):
    """
    Centralized LangGraph state schema for multi-agent synthesis orchestration.
    """
    # 1. Query Parameters
    user_query: str
    processed_query: str
    top_k: int

    # 2. Retrieval State
    retrieved_blocks: List[Dict[str, Any]]
    formatted_context: str
    unique_sources: List[str]
    retrieval_latency_ms: float

    # 3. Analysis Agent Output
    analyzed_documents: List[Dict[str, Any]]
    key_entities: List[str]

    # 4. Comparison Agent Output
    comparison_result: Dict[str, Any]
    comparative_matrix: List[Dict[str, Any]]

    # 5. Verification Agent Output
    extracted_claims: List[Dict[str, Any]]
    verification_results: List[Dict[str, Any]]
    unsupported_claims_count: int
    supported_claims_count: int
    faithfulness_score: float

    # 6. Synthesis Agent Output
    draft_response: str
    final_synthesis: str
    source_citations: List[Dict[str, Any]]

    # 7. Diagnostics & Logging
    execution_trace: List[Dict[str, Any]]
    errors: List[str]
    total_execution_time_ms: float
