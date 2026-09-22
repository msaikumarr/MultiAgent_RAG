"""
Central Configuration Module
=============================
Project: A Multi-Agent and Multi-Model Framework for Intelligent Knowledge Synthesis using Generative AI
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
PROCESSED_DOCS_DIR = DATA_DIR / "processed_documents"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
PROMPTS_DIR = BASE_DIR / "prompts"
EVALUATION_DIR = BASE_DIR / "evaluation"
OUTPUTS_DIR = BASE_DIR / "outputs"


@dataclass
class DocumentConfig:
    """Document processing and chunking configurations."""
    raw_docs_dir: Path = RAW_DOCS_DIR
    processed_docs_dir: Path = PROCESSED_DOCS_DIR
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    min_chunk_length: int = 50


@dataclass
class EmbeddingConfig:
    """Embedding model parameters."""
    model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_dimension: int = 384
    normalize_embeddings: bool = True
    batch_size: int = 32


@dataclass
class VectorStoreConfig:
    """FAISS vector store configurations."""
    store_dir: Path = VECTOR_STORE_DIR
    index_file_name: str = "faiss_index.bin"
    metadata_file_name: str = "chunks_metadata.json"
    top_k: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))


@dataclass
class LLMConfig:
    """LLM Provider and Multi-Model settings."""
    default_provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    default_model: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")

    # Secondary LLM for Multi-Model Verification / Comparison
    verification_provider: str = os.getenv("VERIFICATION_LLM_PROVIDER", "gemini")
    verification_model: str = os.getenv("VERIFICATION_LLM_MODEL", "gemini-3.5-flash-lite")

    # Groq fallback models: used automatically once Gemini's free-tier daily quota is
    # exhausted or it otherwise errors, before falling back to the deterministic engine.
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_verification_model: str = os.getenv("GROQ_VERIFICATION_MODEL", "openai/gpt-oss-120b")

    temperature: float = 0.2
    # Gemini 2.5+ models spend part of this budget on internal "thinking" tokens before
    # emitting visible output, so this must be generous enough to avoid MAX_TOKENS truncation
    # of the visible JSON/report text.
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
    
    # API Keys
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )


@dataclass
class AppConfig:
    """Global aggregate configuration."""
    document: DocumentConfig = field(default_factory=DocumentConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    def ensure_directories(self) -> None:
        """Ensure all required project directories exist on disk."""
        for directory in [
            self.document.raw_docs_dir,
            self.document.processed_docs_dir,
            self.vector_store.store_dir,
            PROMPTS_DIR,
            EVALUATION_DIR,
            OUTPUTS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Default Singleton Configuration Instance
config = AppConfig()
config.ensure_directories()

if __name__ == "__main__":
    print("=== Configuration Module Verification ===")
    print(f"Base Directory       : {BASE_DIR}")
    print(f"Raw Documents Dir    : {config.document.raw_docs_dir}")
    print(f"Vector Store Dir     : {config.vector_store.store_dir}")
    print(f"Embedding Model      : {config.embedding.model_name}")
    print(f"Default LLM Provider : {config.llm.default_provider} ({config.llm.default_model})")
    print(f"Verification Model   : {config.llm.verification_provider} ({config.llm.verification_model})")
    print(f"Chunk Size / Overlap : {config.document.chunk_size} / {config.document.chunk_overlap}")
    print(f"Top-K Retrieval      : {config.vector_store.top_k}")
    print("All project directories successfully verified.")
