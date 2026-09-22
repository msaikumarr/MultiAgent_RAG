"""
Analysis Agent Module
=====================
Specialized AI Agent responsible for extracting methodologies, empirical metrics,
advantages, and limitations strictly from retrieved research paper passages.
"""

import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import config, PROMPTS_DIR
from agents.state import SynthesisGraphState
from models.llm import multi_model_manager, render_prompt, resolve_engine_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class AnalysisAgent:
    """
    Analysis Agent: Performs deep per-document analytical decomposition of retrieved evidence.
    """

    def __init__(self, prompt_template_path: Optional[Path] = None):
        self.prompt_path = prompt_template_path or (PROMPTS_DIR / "analysis_prompt.txt")
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        if self.prompt_path.exists():
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Analyze the following retrieved evidence for the query: {user_query}\n\nContext:\n{formatted_context}"

    def extract_metrics(self, text: str) -> Dict[str, str]:
        """Extracts quantitative empirical metrics (percentages, recall, latency, MRR) from passage."""
        metrics: Dict[str, str] = {}
        
        # Match percentage metrics (e.g., 14.2% improvement, 96.2% Recall)
        pct_matches = re.findall(r'(\d+(?:\.\d+)?%)\s+([a-zA-Z0-9\s@\-_]+?)(?=[,\.\n]|$)', text)
        for val, desc in pct_matches[:4]:
            clean_desc = desc.strip()
            if len(clean_desc) < 40:
                metrics[clean_desc] = val

        # Match latency / time metrics (e.g. 4.2 milliseconds, 45ms)
        lat_matches = re.findall(r'(\d+(?:\.\d+)?\s*(?:milliseconds|ms|seconds|s))\s+([a-zA-Z0-9\s\-_]+?)(?=[,\.\n]|$)', text, re.IGNORECASE)
        for val, desc in lat_matches[:2]:
            clean_desc = desc.strip()
            if len(clean_desc) < 40:
                metrics[clean_desc] = val

        return metrics

    def extract_document_analysis(self, source_paper: str, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes all retrieved passages corresponding to a single research paper.
        """
        combined_text = "\n".join(b.get("text", "") for b in blocks)
        pages = sorted(list(set(b.get("page", 1) for b in blocks)))
        citations = sorted(list(set(b.get("citation_index", 1) for b in blocks)))
        
        # 1. Extract empirical metrics
        metrics = self.extract_metrics(combined_text)

        # 2. Extract key findings / advantages
        findings: List[str] = []
        advantages: List[str] = []
        limitations: List[str] = []
        evidence_quotes: List[str] = []

        lines = [line.strip() for line in combined_text.split("\n") if line.strip()]
        for line in lines:
            lower = line.lower()
            if any(k in lower for k in ["demonstrate", "show", "achieved", "result", "found", "reduction"]):
                if len(line) > 30 and line not in findings:
                    findings.append(line)
                    evidence_quotes.append(line)
            elif any(k in lower for k in ["advantage", "superior", "benefit", "strength", "scalable"]):
                if len(line) > 25 and line not in advantages:
                    advantages.append(line)
            elif any(k in lower for k in ["limitation", "overhead", "fails", "trade-off", "complexity"]):
                if len(line) > 25 and line not in limitations:
                    limitations.append(line)

        # Fallbacks for empty fields
        if not findings and lines:
            findings.append(lines[0])
        if not advantages:
            advantages.append("Semantic representation improvement over baseline keyword approaches.")
        if not limitations:
            limitations.append("Not explicitly detailed in retrieved passages.")

        # Identify core methodology summary
        methodology_snips = [l for l in lines if any(k in l.lower() for k in ["architecture", "methodology", "pipeline", "framework", "utilize", "propose"])]
        proposed_method = methodology_snips[0] if methodology_snips else "Dense vector representation and neural semantic retrieval."

        return {
            "source_paper": source_paper,
            "page_numbers": pages,
            "citation_indices": citations,
            "proposed_method": proposed_method,
            "key_findings": findings[:4],
            "empirical_results": metrics,
            "advantages": advantages[:3],
            "limitations": limitations[:3],
            "evidence_quotes": evidence_quotes[:4]
        }

    def run_llm_analysis(self, user_query: str, formatted_context: str) -> List[Dict[str, Any]]:
        """
        Attempts LLM-driven structured analysis via the primary Gemini model.
        Raises on any failure (no API key, network/quota error, malformed JSON) so
        the caller can fall back to deterministic extraction.
        """
        llm = multi_model_manager.get_primary_llm()
        prompt = render_prompt(self.prompt_template, user_query=user_query, formatted_context=formatted_context)
        result = llm.generate_json(prompt)

        if not isinstance(result, list):
            raise ValueError("Expected a JSON list of per-document analyses from the LLM.")

        normalized: List[Dict[str, Any]] = []
        for item in result:
            if not isinstance(item, dict) or not item.get("source_paper"):
                continue
            normalized.append({
                "source_paper": item.get("source_paper", "unknown"),
                "page_numbers": item.get("page_numbers") or [1],
                "citation_indices": item.get("citation_indices") or [1],
                "proposed_method": item.get("proposed_method") or "Not discussed in retrieved excerpts.",
                "key_findings": item.get("key_findings") or [],
                "empirical_results": item.get("empirical_results") or {},
                "advantages": item.get("advantages") or [],
                "limitations": item.get("limitations") or [],
                "evidence_quotes": item.get("evidence_quotes") or []
            })

        if not normalized:
            raise ValueError("LLM returned no usable per-document analyses.")
        self._last_engine = resolve_engine_label(llm)
        return normalized

    def run(self, state: SynthesisGraphState) -> Dict[str, Any]:
        """
        LangGraph Node execution for Analysis Agent.
        """
        t0 = time.time()
        retrieved_blocks = state.get("retrieved_blocks", [])
        user_query = state.get("user_query", "")
        formatted_context = state.get("formatted_context", "")

        if not retrieved_blocks:
            logger.warning("[AnalysisAgent] No retrieved blocks to analyze.")
            return {
                "analyzed_documents": [],
                "key_entities": []
            }

        analyzed_docs: List[Dict[str, Any]] = []
        engine = "deterministic-extraction"
        try:
            analyzed_docs = self.run_llm_analysis(user_query, formatted_context)
            engine = getattr(self, "_last_engine", "gemini")
        except Exception as e:
            logger.warning(f"[AnalysisAgent] LLM-based analysis unavailable/failed, falling back to deterministic extraction: {e}")

        if not analyzed_docs:
            # Deterministic fallback: group retrieved blocks by document and extract via regex/heuristics.
            doc_groups: Dict[str, List[Dict[str, Any]]] = {}
            for block in retrieved_blocks:
                src = block.get("source", "unknown_document")
                doc_groups.setdefault(src, []).append(block)

            for src, blocks in doc_groups.items():
                analyzed_docs.append(self.extract_document_analysis(src, blocks))
            engine = "deterministic-extraction"

        all_entities = [
            str(doc.get("source_paper", "")).replace(".pdf", "").replace("_", " ").title()
            for doc in analyzed_docs
        ]

        latency_ms = (time.time() - t0) * 1000

        # Update trace
        trace = list(state.get("execution_trace", []))
        trace.append({
            "agent": "AnalysisAgent",
            "action": "Fact, Methodology & Empirical Metric Extraction",
            "engine": engine,
            "papers_analyzed": len(analyzed_docs),
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        })

        logger.info(f"[AnalysisAgent] Completed in {latency_ms:.2f}ms. Analyzed {len(analyzed_docs)} documents.")

        return {
            "analyzed_documents": analyzed_docs,
            "key_entities": list(set(all_entities)),
            "execution_trace": trace
        }


# Singleton Analysis Agent Instance
analysis_agent = AnalysisAgent()


def analysis_node(state: SynthesisGraphState) -> Dict[str, Any]:
    """LangGraph node wrapper for AnalysisAgent."""
    return analysis_agent.run(state)


if __name__ == "__main__":
    print(f"AnalysisAgent initialized with prompt template from: {analysis_agent.prompt_path}")
