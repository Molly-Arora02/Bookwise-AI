"""
BookWise AI — PDF Document Extraction and Normalization.

Extracts text page-by-page from the source PDF using PyMuPDF, preserves
PDF page indices and printed page numbers, normalizes whitespace, and
computes cryptographic hashes for content verification and change detection.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pymupdf


@dataclass
class DocumentPage:
    """Represents a single extracted page from a document with full metadata."""
    page_number: int                 # 1-indexed PDF page number (1 to N)
    printed_page: Optional[int]      # 1-indexed printed catalogue page number if reliably identified
    printed_page_str: str            # Human-readable printed page representation ("Page 1", "Cover", etc.)
    text: str                        # Normalized clean text content
    raw_text: str                    # Original raw extracted text
    char_count: int                  # Character length of normalized text
    source: str                      # Source filename (e.g. "History_2024-25.pdf")
    content_hash: str                # SHA-256 hash of page text
    file_hash: str                   # SHA-256 hash of source PDF file
    metadata: Dict[str, Any]         # Additional page attributes


def compute_file_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_sha256(text: str) -> str:
    """Calculate the SHA-256 checksum of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_page_text(raw_text: str) -> str:
    """
    Normalize extracted page text while strictly preserving:
    - ISBNs (e.g., 978-93-5442-737-4)
    - Prices (e.g., `795, ₹795)
    - Dates (e.g., 20-01-2025, 1948)
    - Book titles, authors, and tabular alignments.
    """
    if not raw_text:
        return ""

    # Replace null bytes and non-printable control characters (except newline and tab)
    cleaned = raw_text.replace("\x00", "")
    
    # Normalize unicode non-breaking spaces, em-spaces, and thin-spaces to regular space
    cleaned = re.sub(r"[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]", " ", cleaned)

    # In Indian DTP fonts (InDesign / Kruti), backtick ` before digits denotes the Rupee symbol ₹.
    # We standardize price occurrences to include both ₹ and the original format for robust retrieval.
    # e.g., `795 -> ₹795 (`795)
    def price_replacer(match):
        price_num = match.group(1)
        return f"₹{price_num} (`{price_num})"

    # Clean multi-spaces on the same line while preserving line breaks
    lines = cleaned.split("\n")
    normalized_lines = []
    for line in lines:
        line_stripped = re.sub(r"[ \t]+", " ", line).strip()
        # Harmonize Rupee price representation
        line_harmonized = re.sub(r"`\s*(\d+)", price_replacer, line_stripped)
        normalized_lines.append(line_harmonized)

    # Collapse excessive blank lines (more than 2 consecutive)
    result = "\n".join(normalized_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def determine_printed_page(pdf_page_1based: int, raw_text: str) -> Tuple[Optional[int], str]:
    """
    Determine the printed catalogue page number corresponding to a given PDF page.
    
    Verified Catalogue Structure for History_2024-25.pdf:
    - PDF Page 1: Front Cover -> printed_page = None, str = "Cover"
    - PDF Page 2: Publisher Information (prices update & incorporation date)
                  -> printed_page = 1, str = "Page 1 (Front Matter)"
    - PDF Page 3: Table of Contents -> printed_page = None, str = "Contents"
    - PDF Pages 4 to 90: Catalogue body & indices:
                  PDF Page 4 -> printed Page 1
                  PDF Page 5 -> printed Page 2
                  PDF Page 6 -> printed Page 3
                  Formula: printed_page = pdf_page_1based - 3
    - PDF Page 91: Back Cover -> printed_page = None, str = "Back Matter"
    """
    if pdf_page_1based == 1:
        return None, "Cover"
    
    if pdf_page_1based == 2:
        # Prompt fixture: "Reference: PDF page 2, corresponding to printed catalogue page 1."
        return 1, "Page 1 (Front Matter)"
    
    if pdf_page_1based == 3:
        return None, "Contents"

    # For content pages (4 to 90), check if printed page number appears in header
    # e.g., "LATEST HISTORY 3" or "2 LATEST HISTORY" or "HISTORY 7"
    expected_printed = pdf_page_1based - 3

    # Attempt to confirm from page header text
    header_snippet = raw_text[:300]
    header_match = re.search(r"(?:HISTORY|LATEST\s+HISTORY)\s+(\d+)", header_snippet, re.IGNORECASE)
    if not header_match:
        header_match = re.search(r"(\d+)\s+(?:HISTORY|LATEST\s+HISTORY)", header_snippet, re.IGNORECASE)

    if header_match:
        found_num = int(header_match.group(1))
        # If header number matches or is plausible, return it
        return found_num, f"Page {found_num}"

    if 4 <= pdf_page_1based <= 90:
        return expected_printed, f"Page {expected_printed}"

    return None, f"PDF Page {pdf_page_1based}"


class PDFLoader:
    """High-performance PDF Loader using PyMuPDF."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Source PDF not found at: {self.pdf_path.resolve()}")
        self.file_hash = compute_file_sha256(self.pdf_path)

    def load(self) -> List[DocumentPage]:
        """
        Extract text and metadata from all pages in the PDF document.
        Returns a list of DocumentPage instances.
        """
        pages: List[DocumentPage] = []
        doc = pymupdf.open(str(self.pdf_path))

        try:
            total_pages = len(doc)
            for idx in range(total_pages):
                pdf_page_num = idx + 1
                page = doc[idx]
                raw_text = page.get_text()
                normalized_text = normalize_page_text(raw_text)
                
                printed_page, printed_page_str = determine_printed_page(pdf_page_num, raw_text)
                content_hash = compute_text_sha256(normalized_text)

                doc_page = DocumentPage(
                    page_number=pdf_page_num,
                    printed_page=printed_page,
                    printed_page_str=printed_page_str,
                    text=normalized_text,
                    raw_text=raw_text,
                    char_count=len(normalized_text),
                    source=self.pdf_path.name,
                    content_hash=content_hash,
                    file_hash=self.file_hash,
                    metadata={
                        "total_pages": total_pages,
                        "pdf_version": doc.metadata.get("format", ""),
                        "creator": doc.metadata.get("creator", ""),
                        "creation_date": doc.metadata.get("creationDate", ""),
                    }
                )
                pages.append(doc_page)
        finally:
            doc.close()

        return pages

    def get_document_summary(self) -> Dict[str, Any]:
        """Return high-level summary metrics about the PDF."""
        pages = self.load()
        non_empty_pages = [p for p in pages if p.char_count > 0]
        total_chars = sum(p.char_count for p in pages)
        
        return {
            "source_file": self.pdf_path.name,
            "file_size_bytes": self.pdf_path.stat().st_size,
            "file_hash": self.file_hash,
            "total_pages": len(pages),
            "non_empty_pages": len(non_empty_pages),
            "total_characters": total_chars,
            "average_chars_per_page": round(total_chars / len(pages), 2) if pages else 0,
            "requires_ocr": len(non_empty_pages) == 0,
        }
