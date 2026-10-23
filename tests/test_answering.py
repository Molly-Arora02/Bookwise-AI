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
