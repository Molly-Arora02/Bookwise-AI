"""
BookWise AI — Automated Tests for PDF Loading, Extraction, Chunking, and Ingestion.
"""

import pytest
from pathlib import Path
from config import settings
from src.bookwise.pdf_loader import PDFLoader, compute_file_sha256, normalize_page_text, determine_printed_page
from src.bookwise.chunking import Chunker, TextChunk


@pytest.fixture
def pdf_loader():
    return PDFLoader(settings.pdf_path)


def test_pdf_exists():
    """Verify source PDF exists on disk."""
    assert settings.pdf_path.exists(), f"Source PDF missing at {settings.pdf_path}"


def test_pdf_metadata_and_pages(pdf_loader):
    """Verify PDF total pages and extraction integrity."""
    summary = pdf_loader.get_document_summary()
    assert summary["total_pages"] == 91, f"Expected 91 pages, got {summary['total_pages']}"
    assert summary["non_empty_pages"] >= 90, "At least 90 pages must contain selectable text"
    assert summary["total_characters"] > 400000, "Catalogue should contain > 400,000 characters"
    assert summary["requires_ocr"] is False, "Native text available, OCR not required"


def test_sha256_checksum(pdf_loader):
    """Verify file SHA-256 calculation."""
    checksum = compute_file_sha256(settings.pdf_path)
    assert len(checksum) == 64
    assert checksum == pdf_loader.file_hash


def test_printed_page_mapping():
    """Verify printed page mapping logic."""
    # PDF Page 1 is Cover
    num1, str1 = determine_printed_page(1, "")
    assert num1 is None and str1 == "Cover"

    # PDF Page 2 is printed page 1 (Front Matter)
    num2, str2 = determine_printed_page(2, "Orient BlackSwan is one of India's best known...")
    assert num2 == 1

    # PDF Page 3 is Contents
    num3, str3 = determine_printed_page(3, "CONTENTS")
    assert num3 is None and str3 == "Contents"

    # PDF Page 4 is printed page 1
    num4, str4 = determine_printed_page(4, "HISTORY 1")
    assert num4 == 1


def test_chunking_generation(pdf_loader):
    """Verify chunk generation, metadata attachment, and stable IDs."""
    pages = pdf_loader.load()
    chunker = Chunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_pages(pages)

    assert len(chunks) > 500, f"Expected > 500 chunks, got {len(chunks)}"

    # Check properties of chunk
    c = chunks[0]
    assert isinstance(c, TextChunk)
    assert c.chunk_id.startswith("bw_")
    assert c.pdf_page == 2  # First non-empty text page
    assert c.printed_page == 1
    assert c.global_chunk_index == 0
    assert len(c.text) > 0
    assert c.content_hash == c.metadata["content_hash"]
