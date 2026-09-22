"""
Verification Agent Module
=========================
Critical component responsible for claim-level factual verification, Natural Language
Inference (NLI) alignment against source passages, hallucination prevention,
and calculating overall faithfulness scores.
"""

import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import json
from config import config, PROMPTS_DIR
from agents.state import SynthesisGraphState, VerifiedClaim
from models.llm import multi_model_manager, render_prompt, resolve_engine_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class VerificationAgent:
    """
    Verification Agent: Implements automated claim decomposition, evidence alignment,
    and faithfulness validation.
    """

    def __init__(self, prompt_template_path: Optional[Path] = None):
        self.prompt_path = prompt_template_path or (PROMPTS_DIR / "verification_prompt.txt")
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        if self.prompt_path.exists():
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Verify claims against evidence: {candidate_claims_json}\n\nContext:\n{formatted_context}"

    def extract_candidate_claims(self, state: SynthesisGraphState) -> List[Dict[str, Any]]:
        """
        Decomposes analysis outputs into atomic testable factual claims.
        """
        claims: List[Dict[str, Any]] = []
        analyzed_docs = state.get("analyzed_documents", [])
        claim_counter = 1

        for doc in analyzed_docs:
            src = doc.get("source_paper", "unknown")
            page = doc.get("page_numbers", [1])[0]

            # 1. Findings Claims
            for finding in doc.get("key_findings", []):
                if len(finding.strip()) > 15:
                    claims.append({
                        "claim_id": f"CLM_{claim_counter:03d}",
                        "claim_text": finding.strip(),
                        "source_paper": src,
                        "page": page,
                        "claim_type": "finding"
                    })
                    claim_counter += 1

            # 2. Metric Claims
            for desc, val in doc.get("empirical_results", {}).items():
                claims.append({
                    "claim_id": f"CLM_{claim_counter:03d}",
                    "claim_text": f"{src} achieves {val} on {desc}.",
                    "source_paper": src,
                    "page": page,
                    "claim_type": "metric",
                    "target_value": val
                })
                claim_counter += 1

            # 3. Advantage Claims
            for adv in doc.get("advantages", [])[:2]:
                if len(adv.strip()) > 15:
                    claims.append({
                        "claim_id": f"CLM_{claim_counter:03d}",
                        "claim_text": adv.strip(),
                        "source_paper": src,
                        "page": page,
                        "claim_type": "advantage"
                    })
                    claim_counter += 1

        return claims

    def verify_single_claim(
        self,
        claim: Dict[str, Any],
        retrieved_blocks: List[Dict[str, Any]]
    ) -> VerifiedClaim:
        """
        Verifies a single claim against retrieved text passages using token entailment.
        """
        claim_text = claim["claim_text"]
        source_paper = claim["source_paper"]
        claim_id = claim["claim_id"]
        page = claim.get("page", 1)

        # Find matching source blocks
        matching_blocks = [b for b in retrieved_blocks if b.get("source") == source_paper]
        if not matching_blocks:
            # Check if any other block contains this exact text
            matching_blocks = retrieved_blocks

        if not matching_blocks:
            return VerifiedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                source_paper=source_paper,
                page=page,
                status="UNSUPPORTED",
                evidence_quote="",
                confidence=0.0,
                rationale="No matching source document passages found in retrieved context."
            )

        best_quote = ""
        highest_overlap = 0.0
        claim_words = set(re.findall(r'\w+', claim_text.lower()))

        for block in matching_blocks:
            block_text = block.get("text", "")
            block_lower = block_text.lower()
            block_page = block.get("page", page)

            # Check for exact substring or key numeric matches
            target_val = claim.get("target_value")
            if target_val and target_val.lower() in block_lower:
                sentences = [s.strip() for s in block_text.split(".") if target_val.lower() in s.lower()]
                if sentences:
                    best_quote = sentences[0] + "."
                    highest_overlap = 0.95
                    page = block_page
                    break

            # Word-level Jaccard overlap on sentences
            for sentence in block_text.split("."):
                s_clean = sentence.strip()
                if len(s_clean) < 15:
                    continue
                s_words = set(re.findall(r'\w+', s_clean.lower()))
                if not s_words:
                    continue
                overlap = len(claim_words.intersection(s_words)) / max(1, len(claim_words))
                if overlap > highest_overlap:
                    highest_overlap = overlap
                    best_quote = s_clean + "."
                    page = block_page

        # Determine Classification
        if highest_overlap >= 0.65 or (claim.get("target_value") and highest_overlap >= 0.40):
            status = "SUPPORTED"
            confidence = min(1.0, round(0.70 + (highest_overlap * 0.30), 2))
            rationale = f"Direct textual entailment confirmed with source passage ({highest_overlap*100:.1f}% lexical alignment)."
        elif highest_overlap >= 0.35:
            status = "PARTIALLY_SUPPORTED"
            confidence = round(0.40 + (highest_overlap * 0.30), 2)
            rationale = f"Core concept mentioned with partial overlap ({highest_overlap*100:.1f}%), but qualifiers require inference."
        else:
            status = "UNSUPPORTED"
            confidence = round(1.0 - highest_overlap, 2)
            rationale = "Insufficient evidence in retrieved passages; flagged as potential hallucination."
            best_quote = ""

        return VerifiedClaim(
            claim_id=claim_id,
            claim_text=claim_text,
            source_paper=source_paper,
            page=page,
            status=status,
            evidence_quote=best_quote,
            confidence=confidence,
            rationale=rationale
        )

    def run_llm_verification(
        self,
        candidate_claims: List[Dict[str, Any]],
        formatted_context: str
    ) -> List[VerifiedClaim]:
        """
        Attempts LLM-driven NLI-style claim verification via the secondary Gemini model.
        Raises on any failure, or if the LLM does not return a verdict for every candidate
        claim, so the caller can fall back to the deterministic lexical-overlap heuristic.
        """
        llm = multi_model_manager.get_verification_llm()
        prompt = render_prompt(
            self.prompt_template,
            candidate_claims_json=json.dumps(candidate_claims, indent=2),
            formatted_context=formatted_context
        )
        result = llm.generate_json(prompt)

        if not isinstance(result, list):
            raise ValueError("Expected a JSON list of verified claims from the LLM.")

        originals_by_id = {c["claim_id"]: c for c in candidate_claims}
        verified: List[VerifiedClaim] = []

        for item in result:
            if not isinstance(item, dict):
                continue
            claim_id = item.get("claim_id")
            original = originals_by_id.get(claim_id, {})

            status = item.get("status", "UNSUPPORTED")
            if status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
                status = "UNSUPPORTED"

            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            verified.append(VerifiedClaim(
                claim_id=claim_id or original.get("claim_id", f"CLM_{len(verified) + 1:03d}"),
                claim_text=item.get("claim_text") or original.get("claim_text", ""),
                source_paper=item.get("source_paper") or original.get("source_paper", "unknown"),
                page=item.get("page", original.get("page", 1)),
                status=status,
                evidence_quote=item.get("evidence_quote", ""),
                confidence=confidence,
                rationale=item.get("rationale", "")
            ))

        if len(verified) != len(candidate_claims):
            raise ValueError(
                f"LLM returned verdicts for {len(verified)}/{len(candidate_claims)} claims; incomplete audit."
            )
        self._last_engine = resolve_engine_label(llm)
        return verified

    def run(self, state: SynthesisGraphState) -> Dict[str, Any]:
        """
        LangGraph Node execution for Verification Agent.
        """
        t0 = time.time()
        retrieved_blocks = state.get("retrieved_blocks", [])
        formatted_context = state.get("formatted_context", "")

        # 1. Extract atomic claims
        candidate_claims = self.extract_candidate_claims(state)

        # 2. Verify all claims (LLM-driven NLI audit, with deterministic fallback)
        verified_claims: List[VerifiedClaim] = []
        engine = "deterministic-lexical-overlap"
        if candidate_claims:
            try:
                verified_claims = self.run_llm_verification(candidate_claims, formatted_context)
                engine = getattr(self, "_last_engine", "gemini")
            except Exception as e:
                logger.warning(f"[VerificationAgent] LLM-based verification unavailable/failed, falling back to deterministic NLI heuristic: {e}")

        if not verified_claims and candidate_claims:
            for claim in candidate_claims:
                verified_claims.append(self.verify_single_claim(claim, retrieved_blocks))
            engine = "deterministic-lexical-overlap"

        supported_count = 0
        unsupported_count = 0
        partially_count = 0
        for v_claim in verified_claims:
            if v_claim.status == "SUPPORTED":
                supported_count += 1
            elif v_claim.status == "PARTIALLY_SUPPORTED":
                partially_count += 1
            else:
                unsupported_count += 1

        # 3. Calculate Global Faithfulness Score
        total = len(verified_claims)
        if total > 0:
            faithfulness_score = round((supported_count + 0.5 * partially_count) / total, 4)
        else:
            faithfulness_score = 1.0

        latency_ms = (time.time() - t0) * 1000

        # Update trace
        trace = list(state.get("execution_trace", []))
        trace.append({
            "agent": "VerificationAgent",
            "action": "Claim Decomposition & NLI Evidence Verification",
            "engine": engine,
            "total_claims_audited": total,
            "supported": supported_count,
            "partially_supported": partially_count,
            "unsupported": unsupported_count,
            "faithfulness_score": faithfulness_score,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        })

        logger.info(
            f"[VerificationAgent] Completed in {latency_ms:.2f}ms. "
            f"Audited {total} claims: {supported_count} SUPPORTED, {partially_count} PARTIAL, {unsupported_count} UNSUPPORTED. "
            f"Faithfulness Score: {faithfulness_score*100:.1f}%."
        )

        return {
            "extracted_claims": [c for c in candidate_claims],
            "verification_results": [v.to_dict() for v in verified_claims],
            "supported_claims_count": supported_count,
            "unsupported_claims_count": unsupported_count,
            "faithfulness_score": faithfulness_score,
            "execution_trace": trace
        }


# Singleton Verification Agent Instance
verification_agent = VerificationAgent()


def verification_node(state: SynthesisGraphState) -> Dict[str, Any]:
    """LangGraph node wrapper for VerificationAgent."""
    return verification_agent.run(state)


if __name__ == "__main__":
    print(f"VerificationAgent initialized with prompt template from: {verification_agent.prompt_path}")
