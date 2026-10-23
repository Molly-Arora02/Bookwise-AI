"""
BookWise AI — Semantic Document Chunking Module.

Splits document pages into semantically cohesive chunks using LangChain's
RecursiveCharacterTextSplitter, preserving page-level and document-level
metadata, stable identifiers, content hashes, and sequential global indices
to support neighboring chunk retrieval.
"""

import hashlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.bookwise.pdf_loader import DocumentPage


@dataclass
class TextChunk:
    """Represents a discrete semantic chunk of text with complete provenance metadata."""
    chunk_id: str                    # Stable unique chunk identifier
    text: str                        # Chunk text content
    pdf_page: int                    # 1-indexed PDF page number
    printed_page: Optional[int]      # Catalogue printed page number
    printed_page_str: str            # Human-readable printed page representation
    chunk_index_in_page: int         # 0-indexed position within the page
    total_page_chunks: int           # Total chunks generated for this page
    global_chunk_index: int          # Sequential index across entire document (0 to N-1)
    source: str                      # Source filename (e.g. "History_2024-25.pdf")
    content_hash: str                # SHA-256 checksum of chunk text
    file_hash: str                   # SHA-256 checksum of source PDF
    char_count: int                  # Character length of chunk
    metadata: Dict[str, Any]         # Flattened dictionary for ChromaDB storage

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to serializable dictionary."""
        return asdict(self)


class Chunker:
    """Configurable Semantic Chunker for Document Pages."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            keep_separator=True,
            length_function=len,
        )

    def chunk_page(
        self,
        page: DocumentPage,
        starting_global_index: int = 0
    ) -> List[TextChunk]:
        """Split a single DocumentPage into semantic TextChunks."""
        if not page.text.strip():
            return []

        raw_splits = self.splitter.split_text(page.text)
        total_chunks = len(raw_splits)
        chunks: List[TextChunk] = []

        for idx, split_text in enumerate(raw_splits):
            cleaned_text = split_text.strip()
            if not cleaned_text:
                continue

            chunk_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
            # Stable unique identifier: {file_hash_prefix}_p{pdf_page}_c{idx}_{chunk_hash_prefix}
            chunk_id = f"bw_{page.file_hash[:8]}_p{page.page_number:03d}_c{idx:02d}_{chunk_hash[:8]}"
            global_idx = starting_global_index + len(chunks)

            metadata: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "pdf_page": page.page_number,
                "printed_page": page.printed_page if page.printed_page is not None else -1,
                "printed_page_str": page.printed_page_str,
                "chunk_index_in_page": idx,
                "total_page_chunks": total_chunks,
                "global_chunk_index": global_idx,
                "source": page.source,
                "content_hash": chunk_hash,
                "file_hash": page.file_hash,
                "char_count": len(cleaned_text),
            }

            chunk = TextChunk(
                chunk_id=chunk_id,
                text=cleaned_text,
                pdf_page=page.page_number,
                printed_page=page.printed_page,
                printed_page_str=page.printed_page_str,
                chunk_index_in_page=idx,
                total_page_chunks=total_chunks,
                global_chunk_index=global_idx,
                source=page.source,
                content_hash=chunk_hash,
                file_hash=page.file_hash,
                char_count=len(cleaned_text),
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    def chunk_pages(self, pages: List[DocumentPage]) -> List[TextChunk]:
        """Process all DocumentPages into an indexed list of TextChunks."""
        all_chunks: List[TextChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page, starting_global_index=len(all_chunks))
            all_chunks.extend(page_chunks)
        return all_chunks
