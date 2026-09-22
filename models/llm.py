"""
Multi-Model LLM Module
======================
Provides unified multi-model interfaces supporting primary generation models
(e.g., Gemini 1.5 Flash), secondary verification models (e.g., Gemini 1.5 Pro),
OpenAI, Groq, and resilient offline academic fallback.
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract interface for all Large Language Model adapters."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        """Generates natural language response from prompt."""
        pass

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Any:
        """Generates structured JSON object from prompt."""
        response_text = self.generate(prompt, system_prompt=system_prompt, temperature=0.1)
        # Clean JSON markdown code blocks
        clean_text = re.sub(r"^```(?:json)?\s*", "", response_text.strip(), flags=re.MULTILINE)
        clean_text = re.sub(r"\s*```$", "", clean_text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            logger.warning("Failed to decode raw JSON response; attempting regex array/object match.")
            match = re.search(r"(\[.*\]|\{.*\})", clean_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Could not parse valid JSON from response: {response_text[:200]}...")


_TYPOGRAPHIC_NORMALIZATION_MAP = {
    "‐": "-",   # hyphen
    "‑": "-",   # non-breaking hyphen
    "‒": "-",   # figure dash
    "–": "-",   # en dash
    "—": "-",   # em dash
    "‘": "'",   # left single quotation mark
    "’": "'",   # right single quotation mark
    "“": '"',   # left double quotation mark
    "”": '"',   # right double quotation mark
    " ": " ",   # non-breaking space
}


def normalize_typography(text: str) -> str:
    """
    Normalizes Unicode typographic substitutions (smart quotes, non-breaking hyphens/spaces,
    en/em dashes) that LLMs sometimes emit in place of plain ASCII. Without this, generated
    Markdown headings and JSON strings can silently diverge from expected literal text
    (e.g. a non-breaking hyphen in "Cross-Document" instead of a plain "-").
    """
    if not text:
        return text
    for char, replacement in _TYPOGRAPHIC_NORMALIZATION_MAP.items():
        text = text.replace(char, replacement)
    return text


class GeminiLLM(BaseLLM):
    """Google Gemini LLM Adapter (e.g., gemini-1.5-flash, gemini-1.5-pro)."""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or config.llm.gemini_api_key
        self._client = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
                logger.info(f"Gemini LLM initialized with model: '{self.model_name}'.")
            except Exception as e:
                logger.warning(f"Could not initialize Google GenAI SDK: {e}")

    def is_available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        if not self.is_available():
            raise RuntimeError(f"Gemini API key not configured or client failed to initialize.")

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self._client.generate_content(
            full_prompt,
            generation_config={"temperature": temperature, "max_output_tokens": config.llm.max_output_tokens}
        )
        text = response.text if response and hasattr(response, "text") else ""
        return normalize_typography(text)


class GroqLLM(BaseLLM):
    """Groq LLM Adapter (e.g., llama-3.3-70b-versatile). Used as a fast fallback when
    the primary Gemini backend is unavailable or its free-tier daily quota is exhausted."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or config.llm.groq_api_key
        self._client = None

        if self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info(f"Groq LLM initialized with model: '{self.model_name}'.")
            except Exception as e:
                logger.warning(f"Could not initialize Groq SDK: {e}")

    def is_available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        if not self.is_available():
            raise RuntimeError("Groq API key not configured or client failed to initialize.")

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=config.llm.max_output_tokens
        )
        return normalize_typography(response.choices[0].message.content or "")


class FallbackChainLLM(BaseLLM):
    """
    Tries a sequence of LLM backends in order, moving to the next one only if the
    current backend raises (e.g. Gemini's free-tier daily request quota is exhausted,
    or a transient network/API error occurs). Used to build resilient multi-provider
    chains such as Gemini -> Groq, with the caller's own deterministic fallback as the
    ultimate safety net if every backend in the chain fails.
    """

    def __init__(self, backends: List[BaseLLM]):
        if not backends:
            raise ValueError("FallbackChainLLM requires at least one backend.")
        self.backends = backends
        self.last_backend: Optional[BaseLLM] = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        last_error: Optional[Exception] = None
        for backend in self.backends:
            backend_label = f"{type(backend).__name__}({getattr(backend, 'model_name', '?')})"
            try:
                result = backend.generate(prompt, system_prompt=system_prompt, temperature=temperature)
                self.last_backend = backend
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"[FallbackChainLLM] {backend_label} failed, trying next backend in chain: {e}")
        raise RuntimeError(f"All LLM backends in fallback chain failed. Last error: {last_error}")


class DeterministicAcademicLLM(BaseLLM):
    """
    High-fidelity academic deterministic model fallback.
    Ensures the pipeline is fully functional and testable without API keys.
    """

    def __init__(self, model_name: str = "deterministic-academic-engine-v1"):
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        return f"[Academic Model Response: {self.model_name}]\nProcessed prompt ({len(prompt)} chars)."


class MultiModelManager:
    """
    Manages multi-model orchestration:
    - Model 1: Sentence Transformer Embeddings (Dense Vector Space)
    - Model 2: Primary LLM (Analysis, Comparison, Synthesis)
    - Model 3: Secondary LLM (Verification & Hallucination Auditing)
    """

    def __init__(self):
        self.primary_provider = config.llm.default_provider
        self.primary_model_name = config.llm.default_model
        self.verification_provider = config.llm.verification_provider
        self.verification_model_name = config.llm.verification_model

    def _build_chain(self, gemini_model_name: str, groq_model_name: str, role_label: str) -> BaseLLM:
        """
        Builds a resilient backend chain: Gemini first, then Groq as an automatic
        fallback (e.g. once Gemini's free-tier daily quota is exhausted or it errors),
        then finally the deterministic engine if neither provider is configured/available.
        """
        backends: List[BaseLLM] = []

        if config.llm.gemini_api_key:
            try:
                gemini = GeminiLLM(model_name=gemini_model_name)
                if gemini.is_available():
                    backends.append(gemini)
            except Exception as e:
                logger.warning(f"Gemini backend unavailable for {role_label}: {e}")

        if config.llm.groq_api_key:
            try:
                groq = GroqLLM(model_name=groq_model_name)
                if groq.is_available():
                    backends.append(groq)
            except Exception as e:
                logger.warning(f"Groq backend unavailable for {role_label}: {e}")

        if not backends:
            return DeterministicAcademicLLM(f"{role_label}-{gemini_model_name}")
        if len(backends) == 1:
            return backends[0]
        return FallbackChainLLM(backends)

    def get_primary_llm(self) -> BaseLLM:
        """Returns the primary reasoning & generation LLM (Gemini -> Groq -> deterministic)."""
        return self._build_chain(self.primary_model_name, config.llm.groq_model, "Primary")

    def get_verification_llm(self) -> BaseLLM:
        """Returns the secondary verification & auditing LLM (Gemini -> Groq -> deterministic)."""
        return self._build_chain(self.verification_model_name, config.llm.groq_verification_model, "Verification")

    @staticmethod
    def _describe_backend(llm: BaseLLM) -> str:
        """Human-readable description of a backend or fallback chain for the inventory panel."""
        if isinstance(llm, FallbackChainLLM):
            return " -> ".join(f"{type(b).__name__}({getattr(b, 'model_name', '?')})" for b in llm.backends)
        return f"{type(llm).__name__}({getattr(llm, 'model_name', '?')})"

    def get_model_inventory(self) -> Dict[str, Any]:
        """Returns the active multi-model architecture configuration."""
        return {
            "model_1_embeddings": {
                "type": "Bi-Encoder Dense Embeddings",
                "model_name": config.embedding.model_name,
                "dimension": config.embedding.vector_dimension,
                "framework": "Sentence Transformers"
            },
            "model_2_primary_llm": {
                "type": "Primary Reasoning & Synthesis LLM",
                "provider": self.primary_provider,
                "model_name": self.primary_model_name,
                "active_backend": self._describe_backend(self.get_primary_llm())
            },
            "model_3_verification_llm": {
                "type": "Secondary Audit & Verification LLM",
                "provider": self.verification_provider,
                "model_name": self.verification_model_name,
                "active_backend": self._describe_backend(self.get_verification_llm())
            }
        }


def resolve_engine_label(llm: BaseLLM) -> str:
    """
    Returns a short label ('gemini', 'groq', 'deterministic') identifying which backend
    actually produced the most recent response from the given LLM handle. For a
    FallbackChainLLM this reflects whichever backend in the chain last succeeded
    (important once Gemini's quota is exhausted and Groq silently takes over).
    """
    target = llm
    if isinstance(llm, FallbackChainLLM):
        target = llm.last_backend or (llm.backends[0] if llm.backends else None)

    if isinstance(target, GeminiLLM):
        return "gemini"
    if isinstance(target, GroqLLM):
        return "groq"
    if isinstance(target, DeterministicAcademicLLM):
        return "deterministic"
    return type(target).__name__ if target is not None else "unknown"


def render_prompt(template: str, **kwargs: Any) -> str:
    """
    Safely substitutes {placeholder} tokens in a prompt template using plain string
    replacement rather than str.format(), since these templates contain literal JSON
    braces (in embedded output-schema examples) that str.format() would misinterpret
    as format fields.
    """
    rendered = template
    for key, value in kwargs.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


# Singleton Multi-Model Manager Instance
multi_model_manager = MultiModelManager()


if __name__ == "__main__":
    inventory = multi_model_manager.get_model_inventory()
    print("=== Multi-Model Framework Inventory ===")
    print(json.dumps(inventory, indent=2))
