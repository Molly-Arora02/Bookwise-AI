"""
BookWise AI — Ingestion CLI Script.

Executes end-to-end document processing:
1. Validates and loads source PDF page-by-page.
2. Normalizes text and maps printed page numbers.
3. Generates semantically cohesive chunks with complete metadata.
4. Generates embeddings and persists vectors in ChromaDB.
5. Verifies indexing integrity and prevents duplicate indexing.
"""

import argparse
import sys
import time
from pathlib import Path

from config import settings
from src.bookwise.pdf_loader import PDFLoader
from src.bookwise.chunking import Chunker
from src.bookwise.vector_store import VectorStore


def run_ingestion(pdf_path: Path, force_rebuild: bool = False) -> bool:
    """Run full ingestion pipeline from PDF to persistent ChromaDB."""
    print("=" * 60)
    print("  BOOKWISE AI — DOCUMENT INGESTION PIPELINE")
    print("  'Find answers in your documents. Verify every source.'")
    print("=" * 60)
    start_time = time.time()

    if not pdf_path.exists():
        print(f"[-] ERROR: Source PDF file not found at: {pdf_path.resolve()}")
        print("    Please place 'History_2024-25.pdf' inside the 'data/' directory.")
        return False

    # 1. Document Extraction
    print(f"\n[1/4] Inspecting and loading PDF: {pdf_path.name}")
    try:
        loader = PDFLoader(pdf_path)
        summary = loader.get_document_summary()
        print(f"      File Size       : {summary['file_size_bytes'] / (1024*1024):.2f} MB")
        print(f"      SHA-256 Checksum: {summary['file_hash']}")
        print(f"      Total Pages     : {summary['total_pages']}")
        print(f"      Non-Empty Pages : {summary['non_empty_pages']}")
        print(f"      Total Characters: {summary['total_characters']:,}")
        print(f"      Average Chars/Pg: {summary['average_chars_per_page']:.1f}")
        print(f"      Requires OCR    : {summary['requires_ocr']}")

        pages = loader.load()
    except Exception as e:
        print(f"[-] ERROR during PDF extraction: {e}")
        return False

    # 2. Check if already indexed
    vector_store = VectorStore()
    if not force_rebuild and vector_store.is_indexed_and_current(loader.file_hash):
        print("\n[+] Source document has not changed and index is current.")
        stats = vector_store.get_stats()
        print(f"    Current Chunks in DB: {stats['total_chunks']}")
        print(f"    Collection Name     : {stats['collection_name']}")
        print("    Use '--force' if you wish to rebuild the index from scratch.")
        print(f"\n[✓] Ingestion verified in {time.time() - start_time:.2f}s.")
        return True

    # 3. Semantic Chunking
    print(f"\n[2/4] Chunking document pages (Size: {settings.chunk_size}, Overlap: {settings.chunk_overlap})...")
    chunker = Chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = chunker.chunk_pages(pages)
    print(f"      Generated {len(chunks)} cohesive text chunks across {len(pages)} pages.")

    # 4. Vector Storage Ingestion
    print(f"\n[3/4] Generating embeddings and persisting to ChromaDB ({vector_store.collection_name})...")
    try:
        result = vector_store.index_chunks(
            chunks=chunks,
            source_file_hash=loader.file_hash,
            source_filename=pdf_path.name,
            force_rebuild=force_rebuild,
        )
        print(f"      Status        : {result['status']}")
        print(f"      Chunks Indexed: {result['chunks_indexed']}")
    except Exception as e:
        print(f"[-] ERROR during vector indexing: {e}")
        return False

    # 5. Verification
    print("\n[4/4] Verifying vector store persistence...")
    stats = vector_store.get_stats()
    print(f"      Total Chunks Persisted: {stats['total_chunks']}")
    print(f"      Persist Directory     : {stats['persist_dir']}")
    print(f"      Status                : {'READY' if stats['is_indexed'] else 'FAILED'}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[✓] Document indexing completed successfully in {elapsed:.2f} seconds.")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="BookWise AI Ingestion Pipeline")
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(settings.pdf_path),
        help=f"Path to PDF file (default: {settings.pdf_path})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if source file is unchanged",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display vector store statistics and exit",
    )

    args = parser.parse_args()

    if args.stats:
        vs = VectorStore()
        import pprint
        pprint.pprint(vs.get_stats())
        return

    success = run_ingestion(Path(args.pdf), force_rebuild=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
