"""
BookWise AI — Acceptance Tests for the 5 Required Questions.

Strictly verifies the product definition and acceptance promises:
1. Grounded factual answer when evidence exists.
2. Verified citations pointing to correct PDF page and printed page.
3. Exact string 'Not found' when unanswerable, off-topic, or subjective.
"""

import pytest
from src.bookwise.answer_engine import AnswerEngine, NOT_FOUND_FALLBACK


@pytest.fixture(scope="module")
def engine():
    return AnswerEngine()


def test_question_1_price_update_date(engine):
    """
    Question 1: On what date were prices in the catalogue last updated?
    Expected: 20 January 2025
    Reference: PDF Page 2, printed page 1
    """
    res = engine.ask("On what date were prices in the catalogue last updated?")
    assert res.is_supported is True
    assert "20 January 2025" in res.answer
    assert len(res.citations) == 1
    citation = res.citations[0]
    assert citation.pdf_page == 2
    assert citation.printed_page == 1
    assert "20-01-2025" in citation.evidence_snippet


def test_question_2_incorporation_year(engine):
    """
    Question 2: In which year was Orient BlackSwan incorporated?
    Expected: 1948
    Reference: PDF Page 2, printed page 1
    """
    res = engine.ask("In which year was Orient BlackSwan incorporated?")
    assert res.is_supported is True
    assert res.answer == "1948"
    assert len(res.citations) == 1
    citation = res.citations[0]
    assert citation.pdf_page == 2
    assert citation.printed_page == 1
    assert "1948" in citation.evidence_snippet


def test_question_3_physics_textbook_price(engine):
    """
    Question 3: What is the price of a Physics textbook?
    Expected: Exactly 'Not found'
    """
    res = engine.ask("What is the price of a Physics textbook?")
    assert res.answer == NOT_FOUND_FALLBACK
    assert res.is_supported is False
    assert len(res.citations) == 0


def test_question_4_minister_of_education(engine):
    """
    Question 4: Who is the current Minister of Education?
    Expected: Exactly 'Not found'
    """
    res = engine.ask("Who is the current Minister of Education?")
    assert res.answer == NOT_FOUND_FALLBACK
    assert res.is_supported is False
    assert len(res.citations) == 0


def test_question_5_best_history_book(engine):
    """
    Question 5: Which is the "best" history book for a beginner?
    Expected: Exactly 'Not found'
    """
    res = engine.ask('Which is the "best" history book for a beginner?')
    assert res.answer == NOT_FOUND_FALLBACK
    assert res.is_supported is False
    assert len(res.citations) == 0


def test_exact_fallback_string_equality(engine):
    """Verify exact fallback string equality without extra punctuation or spaces."""
    res = engine.ask("Completely unrelated query about mars exploration?")
    assert res.answer == "Not found"
    assert res.answer != "Not found."
    assert res.answer != "Sorry, not found"
