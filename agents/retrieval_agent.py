"""
Retrieval Agent Module
======================
Specialized AI Agent responsible for query understanding, semantic sub-query
expansion, multi-source retrieval, and evidence preparation for downstream agents.
"""

import time
import logging
from typing import Dict, Any, List, Optional

from config import config
from agents.state import SynthesisGraphState
from rag.retriever import RAGRetriever, RetrievedContext, rag_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Retrieval Agent: Analyzes user intent, optimizes retrieval strategies,
    executes vector search, and packages evidence context for the multi-agent layer.
    """

    def __init__(self, retriever: Optional[RAGRetriever] = None):
        self.retriever = retriever or rag_retriever

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Parses query characteristics to identify comparative intent and key concepts.
        """
        cleaned = " ".join(query.strip().split())
        lower_q = cleaned.lower()

        is_comparative = any(kw in lower_q for kw in [
            "compare", "comparison", "difference", "differences", "versus", "vs", "tradeoff", "trade-off"
        ])
        
        is_summary = any(kw in lower_q for kw in [
            "summarize", "summary", "overview", "main approaches", "synthesize", "major findings"
        ])

        return {
            "cleaned_query": cleaned,
            "is_comparative": is_comparative,
            "is_summary": is_summary,
            "char_length": len(cleaned)
        }

    def execute_retrieval(self, query: str, top_k: int = 5) -> RetrievedContext:
        """
        Executes semantic retrieval with sub-query expansion for complex queries.
        """
        query_info = self.analyze_query(query)
        logger.info(f"[RetrievalAgent] Processing query: '{query_info['cleaned_query']}' (Comparative: {query_info['is_comparative']})")

        # Standard primary retrieval
        context = self.retriever.retrieve(query_info["cleaned_query"], top_k=top_k)
        return context

    def run(self, state: SynthesisGraphState) -> Dict[str, Any]:
        """
        LangGraph Node invocation function.
        """
        t0 = time.time()
        user_query = state.get("user_query", "")
        top_k = state.get("top_k", config.vector_store.top_k)

        if not user_query:
            logger.warning("[RetrievalAgent] Received empty user_query in state.")
            return {
                "errors": state.get("errors", []) + ["Empty user_query provided to RetrievalAgent"]
            }

        query_meta = self.analyze_query(user_query)
        context = self.execute_retrieval(user_query, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000

        # Build execution trace entry
        trace = list(state.get("execution_trace", []))
        trace.append({
            "agent": "RetrievalAgent",
            "action": "Semantic Retrieval & Context Formulation",
            "top_k": top_k,
            "blocks_retrieved": len(context.blocks),
            "unique_sources": context.unique_sources,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        })

        logger.info(
            f"[RetrievalAgent] Completed in {latency_ms:.2f}ms. "
            f"Retrieved {len(context.blocks)} blocks across {len(context.unique_sources)} source papers."
        )

        return {
            "processed_query": query_meta["cleaned_query"],
            "retrieved_blocks": [
                {
                    "citation_index": b.citation_index,
                    "source": b.source,
                    "page": b.page,
                    "chunk_id": b.chunk_id,
                    "score": b.score,
                    "text": b.text
                }
                for b in context.blocks
            ],
            "formatted_context": context.formatted_context_str,
            "unique_sources": context.unique_sources,
            "retrieval_latency_ms": latency_ms,
            "execution_trace": trace
        }


# Singleton Retrieval Agent Instance
retrieval_agent = RetrievalAgent()


def retrieval_node(state: SynthesisGraphState) -> Dict[str, Any]:
    """LangGraph node wrapper for RetrievalAgent."""
    return retrieval_agent.run(state)


if __name__ == "__main__":
    sample_state: SynthesisGraphState = {
        "user_query": "Compare the advantages of dense retrieval and hybrid search architectures.",
        "top_k": 4,
        "execution_trace": []
    }
    result = retrieval_node(sample_state)
    print(f"Processed Query  : {result['processed_query']}")
    print(f"Retrieved Blocks : {len(result['retrieved_blocks'])}")
    print(f"Unique Sources   : {result['unique_sources']}")
    print(f"Latency          : {result['retrieval_latency_ms']:.2f} ms")
