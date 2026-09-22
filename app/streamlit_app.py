"""
Streamlit Web Demonstration Application
=======================================
Interactive UI for "A Multi-Agent and Multi-Model Framework for
Intelligent Knowledge Synthesis using Generative AI"
"""

import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from config import config, RAW_DOCS_DIR
from rag.document_processor import DocumentProcessor
from rag.chunker import TextChunker
from rag.vector_store import FAISSVectorStore, vector_store
from agents.state import SynthesisGraphState
from agents.graph import create_agent_graph
from agents.retrieval_agent import retrieval_node
from agents.analysis_agent import analysis_node
from agents.comparison_agent import comparison_node
from agents.verification_agent import verification_node
from agents.synthesis_agent import synthesis_node
from models.llm import multi_model_manager

# Page Configuration
st.set_page_config(
    page_title="Multi-Agent Knowledge Synthesis Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #F3F4F6;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
    }
    .status-supported {
        color: #059669;
        font-weight: bold;
    }
    .status-partial {
        color: #D97706;
        font-weight: bold;
    }
    .status-unsupported {
        color: #DC2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Initializes and caches the compiled LangGraph pipeline."""
    return create_agent_graph(
        retrieval_fn=retrieval_node,
        analysis_fn=analysis_node,
        comparison_fn=comparison_node,
        verification_fn=verification_node,
        synthesis_fn=synthesis_node
    )


def rebuild_knowledge_base():
    """Processes raw PDFs, chunks text, and rebuilds the FAISS vector index."""
    with st.spinner("1/3 Extracting and cleaning PDF text..."):
        processor = DocumentProcessor()
        pages = processor.process_all_documents()

    with st.spinner(f"2/3 Splitting {len(pages)} pages into semantic chunks..."):
        chunker = TextChunker()
        chunks = chunker.chunk_all_pages(pages)
        chunker.save_cache(chunks)

    with st.spinner(f"3/3 Generating embeddings and building FAISS index for {len(chunks)} chunks..."):
        vs = FAISSVectorStore()
        vs.build_index(chunks, force_rebuild=True)

    return len(pages), len(chunks), vs.get_total_vectors()


# -------------------------------------------------------------
# SIDEBAR: Knowledge Base & Multi-Model Config
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("### 📚 Knowledge Base Manager")
    
    # Check indexed status
    is_indexed = vector_store.is_indexed()
    doc_count = len(list(RAW_DOCS_DIR.glob("*.pdf")))
    
    st.info(f"**PDF Papers on Disk:** {doc_count}\n\n**FAISS Status:** {'✅ Indexed' if is_indexed else '⚠️ Index Missing'}")
    
    if st.button("🔄 Build / Refresh Index", use_container_width=True):
        pages_n, chunks_n, vecs_n = rebuild_knowledge_base()
        st.success(f"Indexed {doc_count} papers ({pages_n} pages, {chunks_n} chunks) into FAISS!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Multi-Model Inventory")
    inventory = multi_model_manager.get_model_inventory()
    st.caption(f"**Model 1 (Embeddings):** {inventory['model_1_embeddings']['model_name']}")
    st.caption(f"**Model 2 (Primary LLM):** {inventory['model_2_primary_llm']['model_name']} ({inventory['model_2_primary_llm']['active_backend']})")
    st.caption(f"**Model 3 (Verification LLM):** {inventory['model_3_verification_llm']['model_name']}")

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Parameters")
    top_k = st.slider("Retrieval Depth (Top-K Chunks)", min_value=2, max_value=8, value=4, step=1)
    chunk_size = st.number_input("Chunk Size (Chars)", value=config.document.chunk_size, disabled=True)
    chunk_overlap = st.number_input("Chunk Overlap (Chars)", value=config.document.chunk_overlap, disabled=True)


# -------------------------------------------------------------
# MAIN VIEW: Header & Interactive Query Console
# -------------------------------------------------------------
st.markdown("<div class='main-header'>A Multi-Agent & Multi-Model Framework for Intelligent Knowledge Synthesis</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Evidence-Grounded Scientific Multi-Document Synthesis with Automated Verification & Hallucination Mitigation</div>", unsafe_allow_html=True)

# Preset Query Selector
preset_queries = [
    "Select or type a custom question below...",
    "Compare the architectures, empirical benchmarks, advantages, and limitations of dense retrieval versus hybrid search.",
    "What methods are used to reduce hallucinations in generative synthesis?",
    "Which papers discuss multi-agent architectures and LangGraph orchestration?",
    "Summarize the common research gaps and engineering trade-offs across these papers.",
    "Synthesize the major findings and reported metrics from these documents."
]

selected_preset = st.selectbox("🎯 Quick Academic Presets:", preset_queries)
default_text = "" if selected_preset.startswith("Select") else selected_preset

query_input = st.text_area(
    "💬 Enter Academic Research Query:",
    value=default_text,
    placeholder="e.g., Compare the RAG techniques and benchmark results across the papers.",
    height=90
)

col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 5])
with col_btn1:
    submit_clicked = st.button("🚀 Run Multi-Agent Synthesis", type="primary", use_container_width=True)
with col_btn2:
    eval_clicked = st.button("📊 View Benchmark Evaluation", use_container_width=True)

# -------------------------------------------------------------
# EXECUTION & RESULTS PRESENTATION
# -------------------------------------------------------------
if submit_clicked and query_input.strip():
    pipeline = get_pipeline()

    with st.status("🧠 Multi-Agent Pipeline Executing...", expanded=True) as status_box:
        t_start = time.time()
        
        st.write("🔍 **Retrieval Agent**: Formulating semantic queries & searching FAISS index...")
        state_input: SynthesisGraphState = {
            "user_query": query_input.strip(),
            "top_k": top_k,
            "execution_trace": []
        }

        output = pipeline.invoke(state_input)
        total_time = time.time() - t_start
        status_box.update(label=f"✅ Multi-Agent Synthesis Completed in {total_time:.2f}s!", state="complete", expanded=False)

    # 1. Top Metrics Bar
    faithfulness = output.get("faithfulness_score", 0.0)
    sources = output.get("unique_sources", [])
    claims = output.get("verification_results", [])
    sup_count = output.get("supported_claims_count", 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System Faithfulness", f"{faithfulness*100:.1f}%", delta="NLI Grounded")
    with col2:
        st.metric("Source Papers Cited", len(sources))
    with col3:
        st.metric("Audited Claims", f"{len(claims)}", f"{sup_count} Supported")
    with col4:
        st.metric("Total Latency", f"{total_time*1000:.1f} ms")

    st.markdown("---")

    # Tabs for Structured Exploration
    tab_report, tab_claims, tab_matrix, tab_trace, tab_sources = st.tabs([
        "📄 Final Synthesis Report",
        "🛡️ Verification Audit (Claims)",
        "📊 Comparative Matrix",
        "⏱️ Multi-Agent Trace",
        "📚 Retrieved Passages"
    ])

    # Tab 1: Final Synthesis Report
    with tab_report:
        st.markdown(output.get("final_synthesis", "No synthesis generated."))

    # Tab 2: Verification Audit
    with tab_claims:
        st.markdown("### 🛡️ Claim-Level Grounding & Hallucination Audit")
        st.caption("Every extracted claim is evaluated against raw passages using Natural Language Inference (NLI).")
        
        if claims:
            table_data = []
            for c in claims:
                status_icon = "🟢" if c["status"] == "SUPPORTED" else ("🟡" if c["status"] == "PARTIALLY_SUPPORTED" else "🔴")
                table_data.append({
                    "Status": f"{status_icon} {c['status']}",
                    "Claim ID": c.get("claim_id"),
                    "Claim Statement": c.get("claim_text"),
                    "Source Paper": f"{c.get('source_paper')} (p.{c.get('page')})",
                    "Confidence": f"{c.get('confidence', 0.0):.2f}",
                    "Verbatim Evidence Quote": f"\"{c.get('evidence_quote', '')}\"",
                    "Rationale": c.get("rationale")
                })
            df_claims = pd.DataFrame(table_data)
            st.dataframe(df_claims, use_container_width=True)
        else:
            st.info("No candidate claims required verification.")

    # Tab 3: Comparative Matrix
    with tab_matrix:
        st.markdown("### 📊 Cross-Document Comparative Matrix")
        matrix = output.get("comparative_matrix", [])
        if matrix:
            df_matrix = pd.DataFrame(matrix)
            st.dataframe(df_matrix, use_container_width=True)
        else:
            st.info("Single-document query or comparative matrix not generated.")

    # Tab 4: Multi-Agent Execution Trace
    with tab_trace:
        st.markdown("### ⏱️ Step-by-Step Multi-Agent Execution Trace")
        trace = output.get("execution_trace", [])
        if trace:
            for step in trace:
                st.markdown(
                    f"**{step.get('agent')}** — *{step.get('action')}*  \n"
                    f"⏱️ **Latency:** `{step.get('latency_ms', 0):.2f} ms` | **Details:** `{json.dumps({k: v for k, v in step.items() if k not in ['agent', 'action', 'latency_ms', 'timestamp']})}`"
                )
        else:
            st.info("Execution trace empty.")

    # Tab 5: Retrieved Sources
    with tab_sources:
        st.markdown("### 📚 Retrieved Evidence Passages")
        blocks = output.get("retrieved_blocks", [])
        for b in blocks:
            with st.expander(f"[Source {b.get('citation_index')}] {b.get('source')} (Page {b.get('page')}) — Cosine Similarity: {b.get('score', 0.0):.4f}"):
                st.markdown(f"**Chunk ID:** `{b.get('chunk_id')}`")
                st.markdown(f"**Passage Text:**\n\n{b.get('text')}")

elif eval_clicked:
    st.markdown("## 📊 Academic Benchmark Evaluation Report")
    eval_report_path = config.document.processed_docs_dir.parent / "evaluation" / "evaluation_report.md"
    if eval_report_path.exists():
        with open(eval_report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Evaluation report not found. Run 'py evaluation/evaluate.py' to generate benchmarks.")
