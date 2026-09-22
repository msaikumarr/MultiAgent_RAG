"""
Synthesis Agent Module
======================
Master AI Agent responsible for generating the final evidence-grounded academic
synthesis, embedding formal source citations, filtering unverified claims,
and presenting a transparent verification audit.
"""

import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from config import config, PROMPTS_DIR, OUTPUTS_DIR
from agents.state import SynthesisGraphState
from models.llm import multi_model_manager, render_prompt, resolve_engine_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class SynthesisAgent:
    """
    Synthesis Agent: Combines verified multi-agent findings into a coherent,
    audited academic knowledge synthesis document.
    """

    def __init__(self, prompt_template_path: Optional[Path] = None):
        self.prompt_path = prompt_template_path or (PROMPTS_DIR / "synthesis_prompt.txt")
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        if self.prompt_path.exists():
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Synthesize verified findings for query: {user_query}\n\nContext:\n{formatted_context}"

    def build_synthesis_report(self, state: SynthesisGraphState) -> str:
        """
        Synthesizes verified claims, analytical breakdowns, and comparative matrices
        into an evidence-grounded academic report.
        """
        user_query = state.get("user_query", "Academic Inquiry")
        analyzed_docs = state.get("analyzed_documents", [])
        comparison_res = state.get("comparison_result", {})
        verified_claims = state.get("verification_results", [])
        retrieved_blocks = state.get("retrieved_blocks", [])
        faithfulness = state.get("faithfulness_score", 1.0)

        # 1. Filter out UNSUPPORTED claims
        accepted_claims = [c for c in verified_claims if c.get("status") in ["SUPPORTED", "PARTIALLY_SUPPORTED"]]
        unsupported_claims = [c for c in verified_claims if c.get("status") == "UNSUPPORTED"]

        # 2. Extract Document Citations mapping
        citations_map: Dict[str, List[int]] = {}
        for block in retrieved_blocks:
            src = block.get("source", "doc")
            idx = block.get("citation_index", 1)
            citations_map.setdefault(src, []).append(idx)

        # Build Section 1: Executive Summary
        exec_summary_lines = [
            "## 1. Executive Summary",
            f"This academic synthesis addresses the query: *\"{user_query}\"* based on dynamic multi-agent "
            f"retrieval and verification across **{len(analyzed_docs)} peer-reviewed research papers**."
        ]
        if comparison_res.get("common_themes"):
            exec_summary_lines.append(f"**Consensus Finding:** {comparison_res['common_themes'][0]}")

        # Build Section 2: Methodological Analysis & Key Findings
        findings_lines = ["\n## 2. Detailed Methodological Analysis & Empirical Findings"]
        for doc in analyzed_docs:
            src = doc.get("source_paper", "Paper")
            c_indices = sorted(list(set(citations_map.get(src, [1]))))
            c_str = ", ".join(f"[Source {i}]" for i in c_indices)

            findings_lines.append(f"\n### {src} ({c_str})")
            findings_lines.append(f"- **Proposed Methodology:** {doc.get('proposed_method', 'N/A')}")
            
            if doc.get("empirical_results"):
                metrics_fmt = ", ".join(f"{k}: **{v}**" for k, v in doc["empirical_results"].items())
                findings_lines.append(f"- **Key Empirical Metrics:** {metrics_fmt}")
            
            if doc.get("key_findings"):
                findings_lines.append("- **Core Findings:**")
                for f in doc["key_findings"][:3]:
                    findings_lines.append(f"  • {f} ({c_str})")
            
            if doc.get("advantages"):
                findings_lines.append(f"- **Demonstrated Advantage:** {doc['advantages'][0]}")

        # Build Section 3: Comparative Analysis & Trade-offs
        comp_lines = ["\n## 3. Cross-Document Comparative Synthesis & Engineering Trade-offs"]
        matrix = comparison_res.get("comparative_matrix", [])
        if matrix:
            comp_lines.append("\n| Research Paper | Proposed Architecture | Key Metrics | Core Advantage | Main Limitation |")
            comp_lines.append("|---|---|---|---|---|")
            for row in matrix:
                p_name = row.get("paper", "")
                arch = row.get("architecture", "")[:35]
                metrics = row.get("primary_metrics", "Unspecified")[:30]
                adv = row.get("main_advantage", "")[:40]
                lim = row.get("main_limitation", "")[:40]
                comp_lines.append(f"| `{p_name}` | {arch} | {metrics} | {adv} | {lim} |")

        if comparison_res.get("tradeoffs"):
            comp_lines.append("\n**Identified Engineering Trade-offs:**")
            for to in comparison_res["tradeoffs"]:
                comp_lines.append(f"- {to}")

        if comparison_res.get("research_gaps"):
            comp_lines.append("\n**Open Research Gaps:**")
            for rg in comparison_res["research_gaps"]:
                comp_lines.append(f"- {rg}")

        # Build Section 4: Verification Audit
        audit_lines = [
            "\n## 4. Grounding & Factual Verification Audit",
            f"- **System Faithfulness Score:** `{faithfulness*100:.1f}%`",
            f"- **Verified Grounded Claims:** `{len(accepted_claims)}` accepted",
            f"- **Hallucinated / Unsupported Claims:** `{len(unsupported_claims)}` rejected"
        ]
        if unsupported_claims:
            audit_lines.append("\n> [!WARNING]\n> **Filtered Unsupported Claims:**")
            for uc in unsupported_claims:
                audit_lines.append(f"> - *\"{uc.get('claim_text')}\"* (Reason: {uc.get('rationale')})")

        # Build Section 5: Citations Registry
        sources_lines = ["\n## 5. Source Citations & Provenance Registry"]
        for block in retrieved_blocks:
            sources_lines.append(
                f"- **[Source {block.get('citation_index', 1)}]** Paper: `{block.get('source')}` | "
                f"Page: `{block.get('page')}` | Similarity Score: `{block.get('score', 0.0):.4f}` | "
                f"Chunk ID: `{block.get('chunk_id')}`"
            )

        report = "\n".join(
            exec_summary_lines +
            findings_lines +
            comp_lines +
            audit_lines +
            sources_lines
        )
        return report

    def run_llm_synthesis(self, state: SynthesisGraphState) -> str:
        """
        Attempts LLM-driven final report generation via the primary Gemini model,
        composing verified claims and comparative findings into free-form academic prose.
        Raises on any failure so the caller can fall back to the deterministic template builder.
        """
        user_query = state.get("user_query", "Academic Inquiry")
        comparison_res = state.get("comparison_result", {})
        formatted_context = state.get("formatted_context", "")
        verified_claims = [
            c for c in state.get("verification_results", [])
            if c.get("status") in ("SUPPORTED", "PARTIALLY_SUPPORTED")
        ]

        if not verified_claims:
            raise ValueError("No verified (SUPPORTED/PARTIALLY_SUPPORTED) claims available for LLM synthesis.")

        llm = multi_model_manager.get_primary_llm()
        prompt = render_prompt(
            self.prompt_template,
            user_query=user_query,
            verified_claims_json=json.dumps(verified_claims, indent=2),
            comparison_json=json.dumps(comparison_res, indent=2),
            formatted_context=formatted_context
        )
        text = llm.generate(prompt, temperature=config.llm.temperature)
        if not text or len(text.strip()) < 50:
            raise ValueError("LLM synthesis returned an empty or implausibly short response.")
        self._last_engine = resolve_engine_label(llm)
        return text.strip()

    def run(self, state: SynthesisGraphState) -> Dict[str, Any]:
        """
        LangGraph Node execution for Synthesis Agent.
        """
        t0 = time.time()
        final_text = ""
        engine = "deterministic-template"
        try:
            final_text = self.run_llm_synthesis(state)
            engine = getattr(self, "_last_engine", "gemini")
        except Exception as e:
            logger.warning(f"[SynthesisAgent] LLM-based synthesis unavailable/failed, falling back to deterministic report builder: {e}")

        if not final_text:
            final_text = self.build_synthesis_report(state)
            engine = "deterministic-template"

        latency_ms = (time.time() - t0) * 1000

        # Build source citations list
        retrieved_blocks = state.get("retrieved_blocks", [])
        source_citations = [
            {
                "citation_index": b.get("citation_index", 1),
                "source": b.get("source", ""),
                "page": b.get("page", 1),
                "chunk_id": b.get("chunk_id", ""),
                "score": b.get("score", 0.0)
            }
            for b in retrieved_blocks
        ]

        # Save to outputs directory
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = OUTPUTS_DIR / f"synthesis_{timestamp_str}.md"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(final_text)
            logger.info(f"[SynthesisAgent] Saved synthesis artifact to: {report_file}")
        except Exception as e:
            logger.warning(f"[SynthesisAgent] Could not write report file: {e}")

        # Update trace
        trace = list(state.get("execution_trace", []))
        trace.append({
            "agent": "SynthesisAgent",
            "action": "Final Evidence Synthesis & Source Citation Generation",
            "engine": engine,
            "report_length_chars": len(final_text),
            "citations_count": len(source_citations),
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        })

        logger.info(f"[SynthesisAgent] Completed in {latency_ms:.2f}ms. Generated {len(final_text)} chars of grounded synthesis.")

        return {
            "final_synthesis": final_text,
            "source_citations": source_citations,
            "execution_trace": trace
        }


# Singleton Synthesis Agent Instance
synthesis_agent = SynthesisAgent()


def synthesis_node(state: SynthesisGraphState) -> Dict[str, Any]:
    """LangGraph node wrapper for SynthesisAgent."""
    return synthesis_agent.run(state)


if __name__ == "__main__":
    print(f"SynthesisAgent initialized with prompt template from: {synthesis_agent.prompt_path}")
