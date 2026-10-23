"""
BookWise AI — Automated Tests for Candidate Retrieval and Neighbor Expansion.
"""

import pytest
from src.bookwise.retrieval import Retriever


@pytest.fixture
def retriever():
    return Retriever()


def test_retrieval_answerable_question(retriever):
    """Verify retriever finds relevant candidates for factual query."""
    candidates = retriever.retrieve("In which year was Orient BlackSwan incorporated?", top_k=5)
    assert len(candidates) > 0
    top = candidates[0]
    assert "1948" in top.text
    assert top.pdf_page == 2


def test_neighbor_expansion(retriever):
    """Verify neighbor context expansion works."""
    candidates = retriever.retrieve("Orient BlackSwan publisher books distributor", top_k=3, expand_neighbors=True)
    assert len(candidates) > 0
    assert any(c.is_neighbor_expanded for c in candidates)


def test_empty_query_retrieval(retriever):
    """Verify empty query returns empty list."""
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
