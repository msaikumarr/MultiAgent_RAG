"""
Sample Academic Research Papers Generator
=========================================
Generates multi-page academic research paper PDFs with realistic sections,
formal terminology, methodology, equations/tables, advantages, and limitations.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

PAPERS_DIR = Path(__file__).resolve().parent / "raw_documents"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_DATA = [
    {
        "filename": "paper_1_dense_retrieval_rag.pdf",
        "title": "Advancing Dense Retrieval Architectures in Retrieval-Augmented Generation",
        "authors": "Dr. Elena Vance, Dr. Marcus Thorne, Prof. Alistair Finch",
        "pages": [
            # Page 1
            [
                ("Title", "Advancing Dense Retrieval Architectures in Retrieval-Augmented Generation"),
                ("Authors", "Elena Vance, Marcus Thorne, Alistair Finch — Department of AI Systems, Horizon University"),
                ("Heading", "Abstract"),
                ("Body", "Retrieval-Augmented Generation (RAG) has emerged as the premier paradigm to ground Large Language Models (LLMs) on non-parametric knowledge. However, traditional sparse lexical matching frequently suffers from vocabulary mismatch when encountering nuanced academic queries. In this work, we investigate dense bi-encoder architectures paired with Sentence-Transformer embeddings and FAISS vector indexing. We evaluate multi-stage dense retrieval over 150,000 academic passages and demonstrate that cosine similarity in normalized 384-dimensional dense semantic space achieves a 14.2% improvement in Mean Reciprocal Rank (MRR@10) over BM25."),
                ("Heading", "1. Introduction"),
                ("Body", "Parametric LLM memory is prone to knowledge cutoff and catastrophic hallucination. RAG mitigates this limitation by retrieving authoritative external passages dynamically. The core challenge lies in the semantic retrieval phase: how to transform heterogeneous user queries into dense numerical vectors that accurately align with dense document representations in vector databases."),
                ("Heading", "2. Dense Embedding Methodology"),
                ("Body", "Our proposed framework utilizes a dual-encoder transformer architecture (e.g., all-MiniLM-L6-v2) trained with contrastive infoNCE loss. Given a query q and document chunk d, semantic similarity is computed as S(q, d) = <E(q), E(d)> / (||E(q)|| * ||E(d)||). To enable sub-millisecond retrieval across millions of chunks, we employ FAISS with Inverted File Indexing (IVF) and Hierarchical Navigable Small World (HNSW) graphs.")
            ],
            # Page 2
            [
                ("Heading", "3. Experimental Evaluation and Results"),
                ("Body", "We conducted comparative retrieval benchmarks across four benchmark datasets (MS-MARCO, SciFact, HotpotQA, and BioASQ). Our dense FAISS pipeline attained Recall@5 of 88.4% and Recall@10 of 94.1%. The query latency averaged 4.2 milliseconds per search on a single CPU core, proving that lightweight dense representations offer an optimal balance of throughput and accuracy."),
                ("Heading", "4. Advantages and Strengths"),
                ("Body", "1. Superior semantic capture: Dense embeddings capture conceptual synonyms and paraphrase variations where keyword search fails completely.\n2. Scalable indexing: FAISS vector indices allow near-linear scaling to millions of embeddings with low memory footprint.\n3. Modular integration: Embeddings can be refreshed without retraining the underlying language generation model."),
                ("Heading", "5. Limitations and Future Scope"),
                ("Body", "Despite strong semantic recall, pure dense retrieval occasionally fails on exact alphanumeric identifiers, version codes, and chemical formulas. Future research should explore hybrid sparse-dense ensembles and dynamic context compression mechanisms."),
                ("Heading", "6. Conclusion"),
                ("Body", "Dense retrieval using Sentence Transformers and FAISS provides an efficient, robust, and mathematically grounded foundation for multi-document RAG pipelines.")
            ]
        ]
    },
    {
        "filename": "paper_2_multi_agent_synthesis.pdf",
        "title": "Multi-Agent Collaboration Frameworks for Complex Knowledge Synthesis",
        "authors": "Dr. Sophia Chen, Dr. Ryan Patel, Dr. Kenji Takahashi",
        "pages": [
            # Page 1
            [
                ("Title", "Multi-Agent Collaboration Frameworks for Complex Knowledge Synthesis"),
                ("Authors", "Sophia Chen, Ryan Patel, Kenji Takahashi — Institute for Advanced Intelligent Systems"),
                ("Heading", "Abstract"),
                ("Body", "Single-prompt LLMs fail to perform comprehensive comparative reasoning across long, multi-document research corpora. In this paper, we propose a modular Multi-Agent Knowledge Synthesis framework utilizing state-machine orchestration via LangGraph. Our architecture decomposes synthesis into specialized collaborative agents: Retrieval Agent, Analysis Agent, Comparison Agent, Verification Agent, and Synthesis Agent. Empirical tests show an 82% reduction in comparative synthesis omissions compared to monolithic LLM prompts."),
                ("Heading", "1. Introduction and Architectural Motivation"),
                ("Body", "Synthesizing information from multiple research papers requires multiple distinct cognitive operations: information retrieval, critical fact extraction, cross-document contrastive analysis, factual verification, and coherent synthesis. Forcing a single LLM call to perform all operations results in context overload, missed subtleties, and increased hallucination rates."),
                ("Heading", "2. Multi-Agent Role Specification"),
                ("Body", "Our LangGraph state architecture coordinates five discrete agentic roles:\n• Retrieval Agent: Dispatches targeted sub-queries to vector stores.\n• Analysis Agent: Extracts isolated facts, methodology parameters, and key empirical metrics per paper.\n• Comparison Agent: Builds structured comparative matrices contrasting methodologies, pros, and cons.\n• Verification Agent: Cross-validates generated statements against source chunk text.\n• Synthesis Agent: Compiles verified claims into an evidence-grounded final report.")
            ],
            # Page 2
            [
                ("Heading", "3. Agent Coordination and State Management"),
                ("Body", "The multi-agent system uses a centralized TypedState dictionary passed immutably across nodes. Each agent inspects state inputs, invokes specialized system prompts, and appends structured outputs. If the Verification Agent identifies inconsistencies, it flags the claim for filtering prior to final synthesis."),
                ("Heading", "4. Advantages of Agentic Decomposition"),
                ("Body", "1. Traceable Reasoning: Every intermediate deduction (analysis, comparison, verification) is visible in execution logs.\n2. Modular Extensibility: Individual agents can use different specialized models (e.g., faster models for retrieval parsing, larger models for synthesis).\n3. Drastic reduction in cognitive load: Each agent operates under focused, single-purpose prompt instructions."),
                ("Heading", "5. Limitations and Trade-offs"),
                ("Body", "Multi-agent execution incurs increased token usage and latency (averaging 3.5x higher latency than single-turn RAG). Furthermore, compounding errors can arise if the upstream retrieval agent extracts irrelevant context."),
                ("Heading", "6. Conclusion"),
                ("Body", "Specialized multi-agent orchestration provides significant qualitative and grounded superiority for academic multi-document synthesis tasks.")
            ]
        ]
    },
    {
        "filename": "paper_3_hallucination_verification.pdf",
        "title": "Faithfulness Verification and Hallucination Mitigation in LLM Generation",
        "authors": "Prof. David Miller, Dr. Maya Lin, Dr. Arthur Pendelton",
        "pages": [
            # Page 1
            [
                ("Title", "Faithfulness Verification and Hallucination Mitigation in LLM Generation"),
                ("Authors", "David Miller, Maya Lin, Arthur Pendelton — Center for Reliable AI Research"),
                ("Heading", "Abstract"),
                ("Body", "Large Language Models often generate plausible-sounding but factually ungrounded statements, commonly known as hallucinations. In academic research synthesis, unverified claims can lead to severe misinformation. We introduce an automated claim-level Verification Agent that breaks candidate answers into atomic propositional claims and validates each claim against retrieved source chunks using Natural Language Inference (NLI) classification: SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED."),
                ("Heading", "1. The Hallucination Problem in Academic RAG"),
                ("Body", "Even when authoritative documents are injected into the LLM context window, generative models often hallucinate by extrapolating beyond the evidence, attributing findings to the wrong authors, or fusing contradictory claims. Strict verification is therefore non-negotiable for academic reliability."),
                ("Heading", "2. Verification Pipeline Architecture"),
                ("Body", "Our verification framework operates in three sequential phases:\n1. Claim Decomposition: Extracts atomic factual statements from the draft response.\n2. Evidence Alignment: Matches each claim to the exact source chunk containing relevant tokens.\n3. Entailment Verification: Classifies whether the chunk logically entails the claim, assigning a confidence score [0.0 to 1.0].")
            ],
            # Page 2
            [
                ("Heading", "3. Experimental Results on Faithfulness Benchmarks"),
                ("Body", "Evaluating our verification module against 500 expert-annotated research claims yielded an F1-score of 91.4% in detecting factual hallucinations. Filtering out UNSUPPORTED claims before user presentation improved user trust ratings by 68% on a 5-point Likert scale."),
                ("Heading", "4. Strengths and Practical Benefits"),
                ("Body", "• Strict Grounding: Ensures no fabricated facts or synthetic numbers reach the final answer.\n• Transparent Citations: Every accepted claim links directly to its parent document and page index.\n• Graceful Uncertainty Handling: If evidence is absent, the system explicitly declares insufficient evidence rather than guessing."),
                ("Heading", "5. Limitations"),
                ("Body", "The verification agent relies on the quality of the retrieved context. If the retrieval module omits key context, valid true claims may be mistakenly classified as UNSUPPORTED due to context deficiency."),
                ("Heading", "6. Conclusion"),
                ("Body", "Post-generation verification is essential to achieve factual integrity in GenAI research synthesis systems.")
            ]
        ]
    },
    {
        "filename": "paper_4_hybrid_search_evaluation.pdf",
        "title": "Benchmarking Hybrid Dense-Sparse Retrieval with Reranking for Academic Literature",
        "authors": "Dr. Tariq Al-Mansoor, Dr. Rachel Green, Prof. Samuel Vance",
        "pages": [
            # Page 1
            [
                ("Title", "Benchmarking Hybrid Dense-Sparse Retrieval with Reranking for Academic Literature"),
                ("Authors", "Tariq Al-Mansoor, Rachel Green, Samuel Vance — Data Intelligence Laboratory"),
                ("Heading", "Abstract"),
                ("Body", "Dense vector embeddings excel at understanding semantic themes, while sparse lexical models (BM25) excel at exact keyword matching. In this paper, we conduct a comprehensive benchmark comparing pure dense retrieval, sparse BM25 retrieval, and Reciprocal Rank Fusion (RRF) hybrid search combined with Cross-Encoder reranking across 20 academic domains. Our findings show that hybrid retrieval improves Recall@20 by 18.7% over pure dense search."),
                ("Heading", "1. Motivation"),
                ("Body", "Academic literature contains domain-specific nomenclature, math notation, and chemical names that out-of-the-box dense embedding models map poorly. Integrating sparse term frequency signals alongside dense cosine distance remedies this blind spot."),
                ("Heading", "2. Proposed Hybrid Architecture"),
                ("Body", "Our hybrid retrieval pipeline executes parallel searches: FAISS index retrieves top-50 dense candidates using sentence-transformers/all-MiniLM-L6-v2, while BM25 retrieves top-50 lexical candidates. Scores are merged via Reciprocal Rank Fusion: RRF_score(d) = sum(1 / (k + rank_i(d))). A cross-encoder reranker then scores the top-20 merged candidates to yield the final top-k context.")
            ],
            # Page 2
            [
                ("Heading", "3. Results and Computational Overhead"),
                ("Body", "Hybrid search achieved 96.2% Recall@10, significantly outperforming standalone BM25 (76.8%) and standalone Dense FAISS (88.4%). However, cross-encoder reranking added 45ms of inference latency per query, representing an engineering trade-off between speed and top-1 precision."),
                ("Heading", "4. Advantages and Key Findings"),
                ("Body", "1. Robustness: High retrieval quality across both keyword-heavy and conceptually ambiguous queries.\n2. Reranking Accuracy: Cross-encoders capture deep token-level cross-attention between queries and retrieved passages.\n3. Explainability: Transparent fusion of lexical token overlap and vector similarity."),
                ("Heading", "5. Limitations"),
                ("Body", "Increased computational complexity and memory requirements from maintaining dual indices (FAISS vector index + inverted BM25 index)."),
                ("Heading", "6. Conclusion"),
                ("Body", "Hybrid search with reranking provides state-of-the-art retrieval accuracy for enterprise and academic knowledge synthesis pipelines.")
            ]
        ]
    }
]


def create_pdf(paper_info: dict):
    filepath = PAPERS_DIR / paper_info["filename"]
    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1a365d'),
        alignment=1, # Center
        spaceAfter=8
    )
    
    author_style = ParagraphStyle(
        'DocAuthor',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#4a5568'),
        alignment=1,
        spaceAfter=14
    )
    
    heading_style = ParagraphStyle(
        'DocHeading',
        parent=styles['Heading2'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=10,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=6
    )

    story = []

    for page_idx, page_elements in enumerate(paper_info["pages"]):
        if page_idx > 0:
            story.append(PageBreak())
        
        for element_type, text in page_elements:
            if element_type == "Title":
                story.append(Paragraph(text, title_style))
            elif element_type == "Authors":
                story.append(Paragraph(text, author_style))
                story.append(Spacer(1, 4))
            elif element_type == "Heading":
                story.append(Paragraph(text, heading_style))
            elif element_type == "Body":
                formatted_text = text.replace("\n", "<br/>")
                story.append(Paragraph(formatted_text, body_style))
            story.append(Spacer(1, 3))

    doc.build(story)
    print(f"Generated academic paper: {filepath.name} ({len(paper_info['pages'])} pages)")


def main():
    print(f"Generating academic PDF dataset in: {PAPERS_DIR}")
    for paper in PAPERS_DATA:
        create_pdf(paper)
    print("All sample academic PDF research papers created successfully.")


if __name__ == "__main__":
    main()
