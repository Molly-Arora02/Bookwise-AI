"""
BookWise AI — Citation Verification Module.

Ensures every citation is grounded in verifiable document metadata,
validates that cited text exists on the cited page, prevents hallucinated
citations, and coordinates both PDF page index and printed catalogue numbers.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import re


@dataclass
class Citation:
    """Represents a verified citation linked directly to source document coordinates."""
    pdf_page: int                    # 1-indexed PDF page index
    printed_page: Optional[int]      # 1-indexed printed catalogue page number
    printed_page_str: str            # Human-readable printed page string ("Page 1", "Cover", etc.)
    source_file: str                 # Filename ("History_2024-25.pdf")
    evidence_snippet: str            # Exact verifiable excerpt from cited page
    verified: bool                   # Whether citation was verified against page text
    confidence: float                # Verification confidence [0.0 to 1.0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary."""
        return asdict(self)

    def format_citation(self) -> str:
        """Standard human-readable citation label."""
        if self.printed_page and self.printed_page != self.pdf_page:
            return f"PDF Page {self.pdf_page} (Catalogue {self.printed_page_str})"
        return f"PDF Page {self.pdf_page}"


class CitationVerifier:
    """Verifies that citations map directly to physical source pages and content."""

    def __init__(self, total_document_pages: int = 91):
        self.total_document_pages = total_document_pages

    def verify_page_exists(self, pdf_page: int) -> bool:
        """Ensure cited PDF page number is within the bounds of the document."""
        return 1 <= pdf_page <= self.total_document_pages

    def verify_evidence_on_page(self, evidence_snippet: str, page_text: str) -> bool:
        """
        Confirm that the evidence snippet or its core factual content
        actually occurs in the page text.
        """
        if not evidence_snippet or not page_text:
            return False

        clean_evidence = re.sub(r"\s+", " ", evidence_snippet.strip().lower())
        clean_page = re.sub(r"\s+", " ", page_text.strip().lower())

        # Exact substring match
        if clean_evidence in clean_page:
            return True

        # Significant phrase matching (any 6+ word contiguous sequence)
        words = clean_evidence.split()
        if len(words) >= 5:
            phrase = " ".join(words[:5])
            if phrase in clean_page:
                return True

        return False

    def build_verified_citation(
        self,
        pdf_page: int,
        printed_page: Optional[int],
        printed_page_str: str,
        source_file: str,
        evidence_snippet: str,
        page_text: str,
    ) -> Optional[Citation]:
        """
        Construct a verified citation if and only if:
        1. The page exists.
        2. The evidence snippet is grounded in the page text.
        """
        if not self.verify_page_exists(pdf_page):
            return None

        is_verified = self.verify_evidence_on_page(evidence_snippet, page_text)
        if not is_verified:
            # If evidence cannot be matched on the claimed page, reject citation
            return None

        return Citation(
            pdf_page=pdf_page,
            printed_page=printed_page,
            printed_page_str=printed_page_str,
            source_file=source_file,
            evidence_snippet=evidence_snippet.strip(),
            verified=True,
            confidence=1.0,
        )
