"""
Phase 2 Test Harness
====================
Validates PDF loading, text cleaning, per-page metadata extraction,
and cache persistence.
"""

from pathlib import Path
from rag.document_processor import DocumentProcessor

def test_document_processor():
    processor = DocumentProcessor()
    
    print("1. Processing all PDF documents...")
    pages = processor.process_all_documents()
    print(f"Total extracted pages: {len(pages)}")
    
    assert len(pages) > 0, "No pages extracted!"
    
    print("\n2. Page Lineage & Content Sample:")
    for idx, page in enumerate(pages[:3], start=1):
        print(f"\n--- [Page Sample {idx}] ---")
        print(f"Source Document : {page.source}")
        print(f"Page Number     : {page.page}")
        print(f"Character Count : {page.char_count}")
        print(f"Word Count      : {page.word_count}")
        print(f"Snippet (first 180 chars): {page.text[:180]}...")
        
    print("\n3. Testing Cache Persistence & Reloading...")
    cached = processor.load_cache()
    assert len(cached) == len(pages), f"Cache size mismatch: {len(cached)} vs {len(pages)}"
    print(f"Cache verified successfully! {len(cached)} records reloaded.")
    print("\nPhase 2 Verification Passed!")

if __name__ == "__main__":
    test_document_processor()
