"""
BookWise AI — Automated Tests for Evidence Validation and Abstention.
"""

import pytest
from src.bookwise.answer_engine import AnswerEngine, NOT_FOUND_FALLBACK, EvidenceValidator


@pytest.fixture
def engine():
    return AnswerEngine()


def test_question_validation():
    """Verify empty questions and oversized questions fail validation."""
    valid, msg = EvidenceValidator.validate_question("")
    assert valid is False

    valid, msg = EvidenceValidator.validate_question("   ")
    assert valid is False

    valid, msg = EvidenceValidator.validate_question("a" * 850)
    assert valid is False

    valid, msg = EvidenceValidator.validate_question("What is the price of history books?")
    assert valid is True


def test_subjective_query_abstention(engine):
    """Verify subjective inquiries return exactly 'Not found'."""
    subjective_questions = [
        "Which is the \"best\" history book for a beginner?",
        "What is the best history book?",
        "What is the greatest book in the catalogue?",
        "Which book is recommended for beginner?",
    ]
    for q in subjective_questions:
        res = engine.ask(q)
        assert res.answer == NOT_FOUND_FALLBACK, f"Failed on: {q}"
        assert res.is_supported is False
        assert len(res.citations) == 0


def test_off_topic_query_abstention(engine):
    """Verify off-topic inquiries return exactly 'Not found'."""
    off_topic_questions = [
        "What is the price of a Physics textbook?",
        "Who is the current Minister of Education?",
        "What is the stock price of Apple?",
        "What is the weather in Delhi?",
    ]
    for q in off_topic_questions:
        res = engine.ask(q)
        assert res.answer == NOT_FOUND_FALLBACK, f"Failed on: {q}"
        assert res.is_supported is False
        assert len(res.citations) == 0


def test_empty_input_abstention(engine):
    """Verify empty or whitespace query returns exactly 'Not found'."""
    assert engine.ask("").answer == NOT_FOUND_FALLBACK
    assert engine.ask("    ").answer == NOT_FOUND_FALLBACK


def test_book_specific_queries(engine):
    """Verify arbitrary catalogue book queries resolve price, author, ISBN, and imprints."""
    # Price lookup
    res_price = engine.ask("What is the price of Discovering India Anew?")
    assert res_price.is_supported is True
    assert "750" in res_price.answer
    assert res_price.citations[0].pdf_page == 5

    # Author lookup
    res_author = engine.ask("Who is the author of Discovering India Anew?")
    assert res_author.is_supported is True
    assert "Alan Machado" in res_author.answer
    assert res_author.citations[0].pdf_page == 5

    # ISBN lookup
    res_isbn = engine.ask("What is the ISBN of Discovering India Anew?")
    assert res_isbn.is_supported is True
    assert "978-93-5442-837-1" in res_isbn.answer
    assert res_isbn.citations[0].pdf_page == 5

    # Exclusive distributor imprints
    res_dist = engine.ask("Which publishers does Orient BlackSwan distribute for?")
    assert res_dist.is_supported is True
    assert "Permanent Black" in res_dist.answer
    assert "Universities Press" in res_dist.answer
    assert res_dist.citations[0].pdf_page == 2
