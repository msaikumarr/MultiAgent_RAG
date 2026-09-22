"""
RAG Retriever and Context Formatting Module
===========================================
Translates vector database search results into structured,
citation-grounded context blocks for LLM injection.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from config import config
from rag.vector_store import FAISSVectorStore, SearchResult, vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ContextBlock:
    """Represents a single evidence passage formatted for LLM consumption."""
    citation_index: int         # 1-indexed (e.g., [Source 1])
    source: str                 # Document filename (e.g., "paper_1.pdf")
    page: int                   # Page number
    chunk_id: str               # Full unique chunk identifier
    score: float                # Semantic relevance score
    text: str                   # Passage text

    def to_citation_header(self) -> str:
        return f"[Source {self.citation_index}] Paper: {self.source} | Page: {self.page} | Score: {self.score:.4f}"

    def to_full_block(self) -> str:
        return f"{self.to_citation_header()}\n{self.text}\n"


@dataclass
class RetrievedContext:
    """Encapsulates the complete retrieved evidence package for a query."""
    query: str
    blocks: List[ContextBlock] = field(default_factory=list)
    formatted_context_str: str = ""
    unique_sources: List[str] = field(default_factory=list)
    total_tokens_approx: int = 0

    def get_sources_table(self) -> List[Dict[str, Any]]:
        """Returns structured metadata list for citations and audit reporting."""
        return [
            {
                "citation_index": b.citation_index,
                "source": b.source,
                "page": b.page,
                "chunk_id": b.chunk_id,
                "score": b.score,
                "char_length": len(b.text),
                "snippet": b.text[:120] + "..." if len(b.text) > 120 else b.text
            }
            for b in self.blocks
        ]


class RAGRetriever:
    """
    Coordinates semantic retrieval and formats context windows with explicit citations.
    """

    def __init__(self, vs: Optional[FAISSVectorStore] = None):
        self.vector_store = vs or vector_store

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> RetrievedContext:
        """
        Executes query retrieval and generates structured evidence context.
        """
        k = top_k or config.vector_store.top_k
        search_results = self.vector_store.similarity_search(
            query=query,
            top_k=k,
            score_threshold=score_threshold
        )

        blocks: List[ContextBlock] = []
        formatted_parts: List[str] = []
        unique_sources = set()

        for idx, res in enumerate(search_results, start=1):
            block = ContextBlock(
                citation_index=idx,
                source=res.chunk.source,
                page=res.chunk.page,
                chunk_id=res.chunk.chunk_id,
                score=res.score,
                text=res.chunk.text
            )
            blocks.append(block)
            formatted_parts.append(block.to_full_block())
            unique_sources.add(res.chunk.source)

        formatted_context_str = "\n---\n".join(formatted_parts)
        approx_tokens = sum(len(b.text.split()) for b in blocks)

        return RetrievedContext(
            query=query,
            blocks=blocks,
            formatted_context_str=formatted_context_str,
            unique_sources=sorted(list(unique_sources)),
            total_tokens_approx=approx_tokens
        )


# Singleton Instance
rag_retriever = RAGRetriever()


if __name__ == "__main__":
    context = rag_retriever.retrieve("How does dense retrieval compare with BM25?", top_k=3)
    print(f"Retrieved {len(context.blocks)} blocks from sources: {context.unique_sources}")
    print("\nFormatted Context Output:\n")
    print(context.formatted_context_str)
