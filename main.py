"""
Main Execution and Interactive Console Entrypoint
=================================================
Project: A Multi-Agent and Multi-Model Framework for Intelligent Knowledge Synthesis using Generative AI
"""

import sys
import os
import shutil
import time
import argparse
import urllib.parse
from pathlib import Path

# Avoid UnicodeEncodeError on consoles with legacy code pages (e.g. Windows cp1252)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from evaluation.evaluate import BenchmarkEvaluator


def build_pipeline():
    """Compiles the 5-Agent LangGraph Knowledge Synthesis StateGraph."""
    return create_agent_graph(
        retrieval_fn=retrieval_node,
        analysis_fn=analysis_node,
        comparison_fn=comparison_node,
        verification_fn=verification_node,
        synthesis_fn=synthesis_node
    )


def rebuild_index(quiet: bool = False):
    """Ingests raw PDFs from data/raw_documents, creates chunks, and rebuilds FAISS vector store."""
    if not quiet:
        print("\n=======================================================")
        print("        SYNCHRONIZING KNOWLEDGE BASE & FAISS INDEX     ")
        print("=======================================================")
    
    t0 = time.time()
    processor = DocumentProcessor()
    pages = processor.process_all_documents()
    if not pages:
        print(f"[!] Warning: No readable PDF documents found in {RAW_DOCS_DIR}")
        return 0

    chunker = TextChunker()
    chunks = chunker.chunk_all_pages(pages)
    chunker.save_cache(chunks)

    vs = FAISSVectorStore()
    vs.build_index(chunks, force_rebuild=True)
    
    # Reload global vector store instance in memory
    vector_store.load()
    
    if not quiet:
        print(f"[✓] Extracted {len(pages)} pages across documents.")
        print(f"[✓] Created {len(chunks)} semantic chunks.")
        print(f"[✓] FAISS index synchronized with {vs.get_total_vectors()} vectors in {time.time() - t0:.2f}s.\n")
    return len(chunks)


def _resolve_file_path(raw_path: str) -> Path:
    """
    Normalizes a user-supplied path that may be a plain filesystem path or a
    'file://' URI (commonly produced by browsers, file managers, or 'copy link'
    actions), including the '/C:/...' form Windows tools sometimes emit.
    """
    clean_path = raw_path.strip().strip("'").strip('"')
    if clean_path.lower().startswith("file:"):
        parsed = urllib.parse.urlparse(clean_path)
        clean_path = urllib.parse.unquote(parsed.path)
        # A URI like file:///C:/foo.pdf parses to a path starting with "/C:/foo.pdf" on
        # Windows; strip the leading slash so Path() treats it as a drive-rooted path.
        if len(clean_path) > 2 and clean_path[0] == "/" and clean_path[2] == ":":
            clean_path = clean_path[1:]
    return Path(clean_path).resolve()


def ingest_document_file(file_path: str, replace_all: bool = False) -> bool:
    """
    Ingests a specific PDF document provided by path.
    Copies it to RAW_DOCS_DIR and rebuilds the FAISS index.
    """
    p = _resolve_file_path(file_path)
    
    if not p.exists():
        print(f"[X] Error: File does not exist at '{p}'")
        return False

    if p.suffix.lower() != ".pdf":
        print(f"[X] Error: File must be a PDF document (.pdf), received: '{p.name}'")
        return False

    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    if replace_all:
        for existing in RAW_DOCS_DIR.glob("*.pdf"):
            try:
                existing.unlink()
            except Exception:
                pass

    target_path = RAW_DOCS_DIR / p.name
    try:
        shutil.copy2(p, target_path)
        print(f"[✓] Added document to knowledge repository: '{p.name}'")
    except shutil.SameFileError:
        print(f"[i] Document '{p.name}' is already in the repository.")
    except Exception as e:
        print(f"[X] Failed to copy file: {e}")
        return False

    rebuild_index(quiet=False)
    return True


def run_synthesis(query: str, top_k: int = 4, show_header: bool = True):
    """Executes end-to-end multi-agent synthesis on a user query."""
    if show_header:
        print("\n=======================================================")
        print(f"QUERY: \"{query}\"")
        print("=======================================================\n")

    if not vector_store.is_indexed():
        print("[!] Notice: FAISS index missing. Initializing auto-indexing from raw documents...")
        rebuild_index(quiet=False)

    app = build_pipeline()
    initial_state: SynthesisGraphState = {
        "user_query": query,
        "top_k": top_k,
        "execution_trace": []
    }

    t0 = time.time()
    print("⏳ Running Multi-Agent RAG Pipeline (Retrieval -> Analysis -> Comparison -> Verification -> Synthesis)...")
    output = app.invoke(initial_state)
    elapsed = time.time() - t0

    print("\n" + "="*60)
    print("                RAG SYNTHESIS REPORT                    ")
    print("="*60 + "\n")
    print(output.get("final_synthesis", "No synthesis output generated."))
    print("\n-------------------------------------------------------")
    print(f"Total Multi-Agent Latency : {elapsed*1000:.2f} ms")
    print(f"System Faithfulness Score : {output.get('faithfulness_score', 0.0)*100:.1f}%")
    print(f"Source Citations Cited    : {len(output.get('source_citations', []))}")
    print("-------------------------------------------------------\n")


def interactive_console(initial_file: str = None, top_k: int = 4):
    """Interactive Console Session for Document Uploading & Q&A Synthesis."""
    print("=" * 65)
    print("   INTELLIGENT MULTI-AGENT KNOWLEDGE SYNTHESIS CONSOLE")
    print("=" * 65)
    print("Upload a document, ask questions, and receive grounded RAG summaries.")
    print("Commands: 'upload <path>' | 'docs' | 'rebuild' | 'exit' or 'q'")
    print("-" * 65)

    if initial_file:
        ingest_document_file(initial_file)
    else:
        # Check current documents
        current_docs = list(RAW_DOCS_DIR.glob("*.pdf")) if RAW_DOCS_DIR.exists() else []
        if current_docs:
            print(f"\n📂 Currently Indexed Documents ({len(current_docs)}):")
            for i, doc in enumerate(current_docs, start=1):
                print(f"   [{i}] {doc.name}")
        else:
            print("\n📂 No documents currently indexed.")

        doc_input = input("\n👉 Enter path to a PDF document to upload (or press [Enter] to use existing docs): ").strip()
        if doc_input:
            ingest_document_file(doc_input)
        elif not vector_store.is_indexed():
            rebuild_index()

    print("\n" + "=" * 65)
    print(" 🚀 READY! Type your question below to ask the RAG model.")
    print(" Type 'exit' or 'q' to quit at any time.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("💬 Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not query:
            continue

        cmd = query.lower()
        if cmd in ["exit", "quit", "q"]:
            print("Session ended. Goodbye!")
            break
        elif cmd.startswith("upload "):
            new_path = query[7:].strip()
            ingest_document_file(new_path)
            continue
        elif cmd.rstrip("'\"").endswith(".pdf"):
            # Safety net: a bare PDF path typed without the "upload " prefix (e.g. pasted
            # from a file manager) would otherwise be silently misinterpreted as a
            # nonsense question. Treat it as an upload instead.
            print(f"[i] That looks like a PDF file path, not a question — uploading it instead of querying.")
            ingest_document_file(query)
            continue
        elif cmd in ["docs", "list", "files"]:
            docs = list(RAW_DOCS_DIR.glob("*.pdf")) if RAW_DOCS_DIR.exists() else []
            print(f"\n📂 Indexed Documents ({len(docs)}):")
            for i, d in enumerate(docs, start=1):
                print(f"   [{i}] {d.name}")
            print()
            continue
        elif cmd in ["rebuild", "reindex"]:
            rebuild_index()
            continue
        elif cmd in ["help", "?"]:
            print("\nCommands:")
            print("  upload <path> : Add and index a new PDF document")
            print("  docs          : List currently loaded PDF documents")
            print("  rebuild       : Re-chunk and re-index all loaded documents")
            print("  exit / q      : Exit the interactive console\n")
            continue

        # Execute synthesis
        run_synthesis(query, top_k=top_k, show_header=False)


def main():
    parser = argparse.ArgumentParser(
        description="A Multi-Agent and Multi-Model Framework for Intelligent Knowledge Synthesis using Generative AI"
    )
    parser.add_argument("--file", "-f", type=str, help="Path to a PDF document to upload and index immediately")
    parser.add_argument("--query", "-q", type=str, help="Direct research question to synthesize (one-shot mode)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive document Q&A console")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the FAISS vector database from PDFs")
    parser.add_argument("--evaluate", action="store_true", help="Run the benchmark evaluation across all test queries")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--serve-ui", action="store_true", help="Launch the interactive Streamlit Web UI")

    args = parser.parse_args()

    if args.rebuild_index:
        rebuild_index()
    elif args.evaluate:
        evaluator = BenchmarkEvaluator()
        evaluator.run_benchmark(top_k=args.top_k)
    elif args.serve_ui:
        import subprocess
        print("Launching Streamlit Web Application...")
        subprocess.run(["streamlit", "run", str(PROJECT_ROOT / "app" / "streamlit_app.py")])
    elif args.query:
        if args.file:
            ingest_document_file(args.file)
        run_synthesis(args.query, top_k=args.top_k)
    else:
        # Default behavior: Interactive Console Session for easy document upload and Q&A
        interactive_console(initial_file=args.file, top_k=args.top_k)


if __name__ == "__main__":
    main()
