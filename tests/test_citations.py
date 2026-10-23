"""
BookWise AI — Automated Tests for Citation Provenance and Verification.
"""

import pytest
from src.bookwise.citations import CitationVerifier, Citation


@pytest.fixture
def verifier():
    return CitationVerifier(total_document_pages=91)


def test_page_bounds(verifier):
    """Verify that citations can only reference existent PDF pages (1 to 91)."""
    assert verifier.verify_page_exists(1) is True
    assert verifier.verify_page_exists(2) is True
    assert verifier.verify_page_exists(91) is True
    assert verifier.verify_page_exists(0) is False
    assert verifier.verify_page_exists(92) is False
    assert verifier.verify_page_exists(-5) is False


def test_evidence_verification_on_page(verifier):
    """Verify that evidence snippet is matched against real page content."""
    page_text = "Orient BlackSwan was incorporated in 1948. Prices updated on 20-01-2025."
    
    assert verifier.verify_evidence_on_page("incorporated in 1948", page_text) is True
    assert verifier.verify_evidence_on_page("Prices updated on 20-01-2025", page_text) is True
    assert verifier.verify_evidence_on_page("fabricated random claim not in page", page_text) is False


def test_citation_construction(verifier):
    """Verify construction of Citation objects with metadata formatting."""
    page_text = "Prices updated on 20-01-2025. These prices are subject to change without notice."
    snippet = "Prices updated on 20-01-2025"

    citation = verifier.build_verified_citation(
        pdf_page=2,
        printed_page=1,
        printed_page_str="Page 1 (Front Matter)",
        source_file="History_2024-25.pdf",
        evidence_snippet=snippet,
        page_text=page_text,
    )

    assert citation is not None
    assert citation.pdf_page == 2
    assert citation.printed_page == 1
    assert citation.verified is True
    assert citation.format_citation() == "PDF Page 2 (Catalogue Page 1 (Front Matter))"


def test_rejection_of_unsupported_citation(verifier):
    """Verify that ungrounded citations are rejected and return None."""
    page_text = "Some random text without matching evidence."
    snippet = "This text does not exist anywhere on this page."

    citation = verifier.build_verified_citation(
        pdf_page=5,
        printed_page=2,
        printed_page_str="Page 2",
        source_file="History_2024-25.pdf",
        evidence_snippet=snippet,
        page_text=page_text,
    )

    assert citation is None
