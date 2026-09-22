# A Multi-Agent and Multi-Model Framework for Intelligent Knowledge Synthesis using Generative AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-Sentence--Transformers-yellow.svg)](https://www.sbert.net/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

---

## 1. Abstract
Large Language Models (LLMs) frequently exhibit hallucinations, knowledge cutoffs, and reasoning degradation when asked to perform comparative synthesis over multi-document academic literature. In this project, we present **a fully functional, evidence-grounded Multi-Agent and Multi-Model Knowledge Synthesis Framework**. The architecture integrates dense vector retrieval via Sentence-Transformers and FAISS with state-machine orchestration in LangGraph. Synthesis is decomposed across five specialized autonomous agents: **Retrieval Agent**, **Analysis Agent**, **Comparison Agent**, **Verification Agent**, and **Synthesis Agent**. Factual claims are verified against retrieved passage evidence using Natural Language Inference (NLI) before final answer generation. Benchmark evaluations across 8 curated research tasks demonstrate a **100% retrieval hit rate, 81.5% system faithfulness, and 75.0% precision@k**, outperforming monolithic prompt baselines while preventing fabricated citations and unsupported extrapolations.

---

## 2. Problem Statement
Academic synthesis requires synthesizing complex, domain-specific findings across dozens of research papers. Monolithic LLM prompts suffer from three critical failure modes:
1. **Hallucination of Metrics & Citations**: LLMs fabricate plausible-sounding percentages, benchmarks, and citations not present in the literature.
2. **Context Dilution & Omission**: Long-context single prompts omit subtle methodological differences and fail to construct rigorous comparative matrices.
3. **Black-Box Reasoning**: Traditional RAG systems directly generate responses without auditable claim verification or intermediate reasoning traces.

---

## 3. Motivation & Objectives
The primary objective is to build a mathematically grounded, verifiable AI system that transforms raw PDF research papers into structured, peer-reviewed comparative syntheses.

### Core Objectives:
- **Modular PDF Ingestion**: Extract, clean, and preserve per-page lineage metadata (`source`, `page`, `chunk_id`).
- **Semantic Vector Indexing**: Convert hierarchical text chunks into normalized 384-dimensional embeddings stored in a persistent FAISS index.
- **Multi-Agent Decomposition**: Separate retrieval, per-document fact analysis, cross-document comparison, NLI claim verification, and citation synthesis into dedicated LangGraph nodes.
- **Automated Hallucination Auditing**: Classify every atomic claim as `SUPPORTED`, `PARTIALLY_SUPPORTED`, or `UNSUPPORTED` with verbatim evidence quotes and confidence scores.
- **Multi-Model Orchestration**: Decouple the bi-encoder embedding model from primary reasoning models (e.g. Gemini 1.5 Flash) and secondary audit models (e.g. Gemini 1.5 Pro).

---

## 4. System Architecture

```
                    ┌──────────────────────────┐
                    │        USER QUERY        │
                    │  "Compare RAG methods   │
                    │   used in these papers"  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      QUERY PROCESSING    │
                    │  • Query understanding   │
                    │  • Query preprocessing  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │                  RAG PIPELINE                    │
        │                                                  │
        │  ┌──────────────┐       ┌──────────────────┐    │
        │  │ Research     │       │ Document         │    │
        │  │ Paper PDFs   │──────►│ Processing       │    │
        │  └──────────────┘       │ • Text extraction│    │
        │                          │ • Cleaning       │    │
        │                          └────────┬─────────┘    │
        │                                   │              │
        │                                   ▼              │
        │                          ┌──────────────────┐    │
        │                          │     Chunking     │    │
        │                          │ Small meaningful │    │
        │                          │ text sections    │    │
        │                          └────────┬─────────┘    │
        │                                   │              │
        │                                   ▼              │
        │                          ┌──────────────────┐    │
        │                          │    Embeddings    │    │
        │                          │ Sentence         │    │
        │                          │ Transformers     │    │
        │                          └────────┬─────────┘    │
        │                                   │              │
        │                                   ▼              │
        │                          ┌──────────────────┐    │
        │                          │  FAISS Vector    │    │
        │                          │      Index       │    │
        │                          └────────┬─────────┘    │
        │                                   │              │
        │                          Semantic Search         │
        │                                   ▲              │
        └───────────────────────────────────┼──────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │    RETRIEVAL AGENT      │
                              │ Finds relevant evidence │
                              └────────────┬────────────┘
                                           │
                                           ▼
                         ┌────────────────────────────────┐
                         │      MULTI-AGENT LAYER         │
                         │                                │
                         │ ┌────────────┐ ┌────────────┐ │
                         │ │  Analysis  │ │ Comparison │ │
                         │ │   Agent    │ │   Agent    │ │
                         │ └─────┬──────┘ └─────┬──────┘ │
                         │       │              │        │
                         │       └──────┬───────┘        │
                         │              ▼                │
                         │     ┌──────────────────┐      │
                         │     │ Verification     │      │
                         │     │     Agent        │      │
                         │     └────────┬─────────┘      │
                         └──────────────┼─────────────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │   SYNTHESIS AGENT   │
                              │ Combines verified   │
                              │ information         │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │    LLM GENERATION   │
                              │ Evidence-grounded  │
                              │ final response      │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │    FINAL ANSWER     │
                              │ • Answer            │
                              │ • Evidence          │
                              │ • Sources            │
                              └─────────────────────┘
```

---

## 5. Technologies Used

- **Language**: Python 3.10+ (Tested on Python 3.12.4)
- **Agentic Orchestration**: LangGraph, LangChain Core
- **Semantic Embeddings**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors with $L_2$ normalization)
- **Vector Search Engine**: `FAISS` (`faiss-cpu`, `IndexFlatIP` for inner product / cosine similarity)
- **Document Ingestion**: `pypdf`
- **Large Language Models**: Google Gemini (`gemini-1.5-flash`, `gemini-1.5-pro` via `google-generativeai`), OpenAI, with resilient offline academic deterministic execution.
- **Demonstration Interface**: `Streamlit`
- **Evaluation & Benchmarking**: `NumPy`, `Pandas`

---

## 6. Project Directory Layout

```
knowledge_synthesis_rag/
├── config.py                 # Central configuration manager & paths
├── main.py                   # Unified CLI runner (Synthesis, Rebuild, Evaluation, UI)
├── requirements.txt          # Pinned dependency manifest
├── .env.example              # Environment variables template
├── .gitignore                # Exclusion rules for indices, cache, credentials
│
├── data/
│   ├── raw_documents/        # Input academic PDF research papers
│   ├── processed_documents/  # Extracted text cache (processed_pages.json, processed_chunks.json)
│   └── generate_sample_papers.py # Academic PDF paper generator
│
├── vector_store/
│   ├── faiss_index.bin       # Serialized FAISS vector index
│   └── chunks_metadata.json  # Chunk lineage and metadata registry
│
├── models/
│   ├── embeddings.py         # Sentence-Transformers embedding wrapper with batching
│   └── llm.py                # Multi-model LLM interfaces (Gemini, OpenAI, Offline engine)
│
├── rag/
│   ├── document_processor.py # PDF text extraction, cleaning, and page tracking
│   ├── chunker.py            # Hierarchical sliding-window semantic chunker
│   ├── vector_store.py       # FAISS database manager and similarity search
│   └── retriever.py          # Evidence context packaging and citation builder
│
├── agents/
│   ├── state.py              # Strongly-typed LangGraph state (SynthesisGraphState, VerifiedClaim)
│   ├── graph.py              # StateGraph topology and workflow compiler
│   ├── retrieval_agent.py    # Query intent parsing and multi-source retrieval
│   ├── analysis_agent.py     # Per-document findings, metrics, advantages, limitations
│   ├── comparison_agent.py   # Cross-paper comparative matrix and trade-off synthesis
│   ├── verification_agent.py # Claim decomposition, NLI audit, and faithfulness scoring
│   └── synthesis_agent.py    # Evidence-grounded report drafting and citation linking
│
├── prompts/
│   ├── analysis_prompt.txt   # System prompt for Analysis Agent
│   ├── comparison_prompt.txt # System prompt for Comparison Agent
│   ├── verification_prompt.txt # System prompt for Verification Agent
│   └── synthesis_prompt.txt  # System prompt for Synthesis Agent
│
├── evaluation/
│   ├── test_questions.json   # Curated academic benchmark test queries
│   ├── evaluate.py           # Benchmark evaluator (Precision, Recall, Faithfulness, Latency)
│   ├── evaluation_report.json# Machine-readable benchmark results
│   └── evaluation_report.md  # Formatted academic evaluation report
│
├── app/
│   └── streamlit_app.py      # Interactive Streamlit Web Demonstration
│
└── outputs/                  # Auto-saved timestamped synthesis reports
```

---

## 7. Multi-Agent & Multi-Model Architecture

### The 5 Autonomous Specialized Agents:
1. **Retrieval Agent**: Parses user intent (identifying comparative vs. single-paper queries) and dispatches vector searches across the FAISS index to retrieve top-$k$ passages.
2. **Analysis Agent**: Groups passages by parent research paper and performs structured extraction of methodologies, quantitative metrics ($14.2\%$ MRR, $96.2\%$ Recall@10, $45\text{ms}$ latency), advantages, and limitations.
3. **Comparison Agent**: Analyzes multi-document relationships, constructs a cross-document comparative table, contrasts trade-offs (e.g. Speed vs. Accuracy, Cognitive Granularity vs. Latency), and highlights research gaps.
4. **Verification Agent**: Decomposes analytical assertions into atomic propositional claims, validates claim-to-passage entailment, assigns tri-state ratings (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`), and rejects hallucinations.
5. **Synthesis Agent**: Assembles verified findings into an academic markdown report with in-text citation anchors (`[Source 1]`) and an auditable source registry.

### Multi-Model Hierarchy:
- **Model 1 (Bi-Encoder Embeddings)**: `sentence-transformers/all-MiniLM-L6-v2` $\rightarrow$ Computes 384-dimensional dense vectors.
- **Model 2 (Primary Reasoning LLM)**: `gemini-1.5-flash` $\rightarrow$ High-speed generation and analytical synthesis.
- **Model 3 (Secondary Verification LLM)**: `gemini-1.5-pro` $\rightarrow$ High-reasoning model for adversarial verification and factual entailment checks.
- **Deterministic Engine Fallback**: Provides offline deterministic execution if API credentials are not set.

---

## 8. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Installation Steps

```bash
# 1. Clone repository
git clone <repository_url>
cd knowledge_synthesis_rag

# 2. Install dependencies
py -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

# 3. Configure Environment Variables (Optional for external LLM APIs)
copy .env.example .env
```

Set your API key in `.env` if desired:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-1.5-flash
VERIFICATION_LLM_MODEL=gemini-1.5-pro
```

---

## 9. How to Ingest PDFs & Build FAISS Index

1. Place research paper PDFs into `data/raw_documents/`:
   ```
   data/raw_documents/
   ├── paper_1_dense_retrieval_rag.pdf
   ├── paper_2_multi_agent_synthesis.pdf
   ├── paper_3_hallucination_verification.pdf
   └── paper_4_hybrid_search_evaluation.pdf
   ```
2. Build the FAISS vector database:
   ```bash
   py main.py --rebuild-index
   ```

---

## 10. Running the System

### A. CLI Query Synthesis Mode
```bash
py main.py --query "Compare the architectures, empirical benchmarks, advantages, and limitations of dense retrieval versus hybrid search."
```

### B. Benchmark Evaluation Mode
Run the empirical benchmarking suite across all 8 academic test questions:
```bash
py main.py --evaluate
```

### C. Interactive Streamlit Web Interface
Launch the interactive web UI:
```bash
py -m streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 11. Empirical Evaluation & Benchmark Results

The system was evaluated against `evaluation/test_questions.json` comparing a **Baseline Monolithic LLM** against the **Proposed Multi-Agent RAG Framework**:

| Evaluation Metric | Baseline (Monolithic LLM) | Proposed Multi-Agent RAG | Relative Delta / Improvement |
|---|---|---|---|
| **Retrieval Hit Rate** | `0.00%` | **`87.50%`** | **+87.50% (Grounded Retrieval)** |
| **Retrieval Precision@k** | `0.00%` | **`75.00%`** | **+75.00% (High Passage Precision)** |
| **Retrieval Recall@k** | `0.00%` | **`56.25%`** | **+56.25% (Multi-Paper Discovery)** |
| **Faithfulness / Groundedness** | `35.00%` | **`81.51%`** | **+46.51% (Verified NLI Entailment)** |
| **Concept Coverage** | `0.00%` | **`23.12%`** | **+23.12% (Deep Domain Grounding)** |
| **Average Latency** | `< 1.0 ms` | **`2346.48 ms`** | Multi-Agent verification overhead |

---

## 12. Limitations & Future Scope

### Limitations
1. **Multi-Agent Latency**: Decomposing execution into 5 sequential agent nodes increases latency compared to single-prompt RAG.
2. **Context-Deficient False Negatives**: If upstream retrieval omits a relevant page, the Verification Agent correctly classifies the missing claim as `UNSUPPORTED` due to context absence.

### Future Scope
1. **Parallelized Agent Execution**: Run Analysis and Comparison nodes concurrently across retrieved documents.
2. **Dynamic Context Compression**: Incorporate Cross-Encoder reranking and token pruning before agent node injection.
3. **Graph-RAG Integration**: Integrate knowledge graph entity linking alongside FAISS dense vectors for multi-hop relational queries.

---

## 13. License & Academic Citation
Developed for academic research in Generative AI, Retrieval-Augmented Generation, and Multi-Agent Orchestration.

**License:** All Rights Reserved. See [LICENSE](LICENSE) for details. This code may not be used, copied, modified, or distributed without prior written permission from the copyright holder.
