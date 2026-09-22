"""
LangGraph Multi-Agent Orchestration Graph
=========================================
Assembles the complete state machine coordinating the Retrieval, Analysis,
Comparison, Verification, and Synthesis agents.
"""

import time
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from agents.state import SynthesisGraphState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def create_agent_graph(
    retrieval_fn,
    analysis_fn,
    comparison_fn,
    verification_fn,
    synthesis_fn
) -> Any:
    """
    Constructs and compiles the multi-agent LangGraph workflow.
    
    Args:
        retrieval_fn: Callable for Retrieval Agent node
        analysis_fn: Callable for Analysis Agent node
        comparison_fn: Callable for Comparison Agent node
        verification_fn: Callable for Verification Agent node
        synthesis_fn: Callable for Synthesis Agent node
        
    Returns:
        Compiled LangGraph instance
    """
    workflow = StateGraph(SynthesisGraphState)

    # 1. Define Nodes
    workflow.add_node("retrieval_agent", retrieval_fn)
    workflow.add_node("analysis_agent", analysis_fn)
    workflow.add_node("comparison_agent", comparison_fn)
    workflow.add_node("verification_agent", verification_fn)
    workflow.add_node("synthesis_agent", synthesis_fn)

    # 2. Define Sequential Graph Flow
    workflow.add_edge(START, "retrieval_agent")
    workflow.add_edge("retrieval_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "comparison_agent")
    workflow.add_edge("comparison_agent", "verification_agent")
    workflow.add_edge("verification_agent", "synthesis_agent")
    workflow.add_edge("synthesis_agent", END)

    # 3. Compile Graph
    compiled_app = workflow.compile()
    logger.info("Compiled Multi-Agent LangGraph Knowledge Synthesis StateGraph.")
    return compiled_app
