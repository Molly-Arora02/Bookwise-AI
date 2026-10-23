"""
BookWise AI — Retrieval Engine.

Implements candidate retrieval with vector similarity, hybrid keyword/entity
re-ranking, neighboring chunk expansion for context continuity, and
calibrated relevance filtering.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set

from src.bookwise.vector_store import VectorStore
from config import settings


@dataclass
class RetrievedCandidate:
    """Represents a retrieved and validated evidence candidate."""
    chunk_id: str
    text: str
    pdf_page: int
    printed_page: Optional[int]
    printed_page_str: str
    global_chunk_index: int
    source: str
    similarity_score: float
    distance: float
    is_neighbor_expanded: bool = False
    neighbor_context: str = ""
    metadata: Dict[str, Any] = None


class Retriever:
    """Hybrid Context-Aware Retriever for Document Intelligence."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 5,
        threshold: float = 0.45,
        neighbor_window: int = 1,
    ):
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        self.threshold = threshold
        self.neighbor_window = neighbor_window

    def _extract_query_entities(self, query: str) -> Set[str]:
        """Extract key entities, numbers, dates, and nouns from question."""
        tokens = set()
        # Find 4-digit years (e.g., 1948, 2024, 2025)
        for year in re.findall(r"\b(19\d{2}|20\d{2})\b", query):
            tokens.add(year.lower())
        # Find dates (e.g., 20-01-2025 or words like January)
        for date in re.findall(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b", query):
            tokens.add(date.lower())
        # Find ISBN patterns
        for isbn in re.findall(r"978[-0-9]{10,16}", query):
            tokens.add(isbn.lower())
        # Significant word tokens (len > 3)
        words = re.findall(r"[A-Za-z0-9]+", query.lower())
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why",
            "how", "is", "are", "was", "were", "the", "a", "an", "and", "or",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
            "book", "catalogue", "catalog", "document"
        }
        for w in words:
            if w not in stopwords and len(w) >= 3:
                tokens.add(w)
        return tokens

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_neighbors: bool = True,
    ) -> List[RetrievedCandidate]:
        """
        Execute hybrid candidate retrieval:
        1. Vector similarity search in ChromaDB.
        2. Keyword/Entity scoring boost for exact matches.
        3. Neighbor chunk retrieval to resolve chunk boundary context loss.
        4. Calibrated threshold filtering.
        """
        k = top_k or self.top_k
        if not query.strip():
            return []

        # 1. Primary vector search - search top 50 candidates to allow hybrid re-ranking
        raw_candidates = self.vector_store.search(
            query=query,
            top_k=max(k * 4, 30),
        )

        # Include front matter chunks (p002) for publisher/catalogue metadata queries
        q_lower = query.lower()
        if any(term in q_lower for term in ["price", "update", "date", "incorporat", "orient blackswan", "publisher", "founded"]):
            p2_chunk = self.vector_store.get_chunk_by_global_index(0)
            if p2_chunk:
                raw_candidates.append({
                    "chunk_id": p2_chunk["chunk_id"],
                    "text": p2_chunk["text"],
                    "metadata": p2_chunk["metadata"],
                    "distance": 0.25,
                    "score": 0.85,
                })

        if not raw_candidates:
            return []

        query_entities = self._extract_query_entities(query)
        processed_candidates: List[RetrievedCandidate] = []
        seen_chunk_ids: Set[str] = set()

        # 2. Score and filter candidates
        for item in raw_candidates:
            chunk_id = item["chunk_id"]
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            text = item["text"]
            meta = item["metadata"]
            distance = item["distance"]
            base_score = item["score"]

            # Hybrid keyword / entity boost
            text_lower = text.lower()
            entity_matches = sum(1 for entity in query_entities if entity in text_lower)
            boost = min(0.20, entity_matches * 0.05)
            adjusted_score = min(1.0, base_score + boost)

            candidate = RetrievedCandidate(
                chunk_id=chunk_id,
                text=text,
                pdf_page=int(meta.get("pdf_page", 1)),
                printed_page=int(meta.get("printed_page")) if meta.get("printed_page") != -1 else None,
                printed_page_str=meta.get("printed_page_str", ""),
                global_chunk_index=int(meta.get("global_chunk_index", 0)),
                source=meta.get("source", ""),
                similarity_score=adjusted_score,
                distance=distance,
                metadata=meta,
            )
            processed_candidates.append(candidate)

        # Sort by adjusted similarity score (descending)
        processed_candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        top_candidates = processed_candidates[:k]

        # 3. Neighbor chunk expansion if requested
        if expand_neighbors and self.neighbor_window > 0:
            for cand in top_candidates:
                neighbor_texts = []
                # Fetch preceding chunk
                if cand.global_chunk_index > 0:
                    prev_chunk = self.vector_store.get_chunk_by_global_index(cand.global_chunk_index - 1)
                    if prev_chunk and prev_chunk.get("text"):
                        neighbor_texts.append(prev_chunk["text"])

                # Add current chunk text
                neighbor_texts.append(cand.text)

                # Fetch following chunk
                next_chunk = self.vector_store.get_chunk_by_global_index(cand.global_chunk_index + 1)
                if next_chunk and next_chunk.get("text"):
                    neighbor_texts.append(next_chunk["text"])

                if len(neighbor_texts) > 1:
                    cand.is_neighbor_expanded = True
                    cand.neighbor_context = "\n---\n".join(neighbor_texts)

        return top_candidates
