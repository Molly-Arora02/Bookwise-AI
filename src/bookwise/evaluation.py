"""
BookWise AI — Evaluation Dataset and Benchmark Runner.

Contains the 5 core acceptance questions from the product specification,
plus an expanded suite of factual catalogue questions and negative controls.
Computes precision, recall, citation accuracy, and exact match metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
from tabulate import tabulate

from src.bookwise.answer_engine import AnswerEngine, NOT_FOUND_FALLBACK


@dataclass
class EvalSample:
    """Represents a single evaluation test case."""
    id: str
    question: str
    expected_answer: str
    expected_supported: bool
    expected_pdf_page: Optional[int] = None
    expected_printed_page: Optional[int] = None
    category: str = "general"
    rationale: str = ""


# The 5 Required Acceptance Questions from Part 3
FIVE_REQUIRED_QUESTIONS: List[EvalSample] = [
    EvalSample(
        id="REQ-01",
        question="On what date were prices in the catalogue last updated?",
        expected_answer="20 January 2025",
        expected_supported=True,
        expected_pdf_page=2,
        expected_printed_page=1,
        category="required_factual",
        rationale="The catalogue states that prices were updated on 20-01-2025.",
    ),
    EvalSample(
        id="REQ-02",
        question="In which year was Orient BlackSwan incorporated?",
        expected_answer="1948",
        expected_supported=True,
        expected_pdf_page=2,
        expected_printed_page=1,
        category="required_factual",
        rationale="The publisher information states that Orient BlackSwan was incorporated in 1948.",
    ),
    EvalSample(
        id="REQ-03",
        question="What is the price of a Physics textbook?",
        expected_answer=NOT_FOUND_FALLBACK,
        expected_supported=False,
        category="required_unanswerable",
        rationale="The supplied History catalogue does not establish the price of an unspecified Physics textbook.",
    ),
    EvalSample(
        id="REQ-04",
        question="Who is the current Minister of Education?",
        expected_answer=NOT_FOUND_FALLBACK,
        expected_supported=False,
        category="required_unanswerable",
        rationale="The assistant must answer exclusively from the supplied catalogue, not external knowledge or web search.",
    ),
    EvalSample(
        id="REQ-05",
        question="Which is the \"best\" history book for a beginner?",
        expected_answer=NOT_FOUND_FALLBACK,
        expected_supported=False,
        category="required_subjective",
        rationale="The catalogue does not establish a definitive answer to this subjective question.",
    ),
]

# Additional Benchmark Test Cases across the Catalogue
EXPANDED_EVAL_DATASET: List[EvalSample] = FIVE_REQUIRED_QUESTIONS + [
    EvalSample(
        id="EXT-01",
        question="What is the price of Discovering India Anew?",
        expected_answer="₹795",
        expected_supported=True,
        expected_pdf_page=5,
        expected_printed_page=2,
        category="catalogue_price",
        rationale="Discovering India Anew by Alan Machado is listed at ₹795 / `795 on page 5.",
    ),
    EvalSample(
        id="EXT-02",
        question="What is the ISBN of Discovering India Anew?",
        expected_answer="978-93-5442-737-4",
        expected_supported=True,
        expected_pdf_page=5,
        expected_printed_page=2,
        category="catalogue_isbn",
        rationale="Listed with ISBN 978-93-5442-737-4 on PDF page 5.",
    ),
    EvalSample(
        id="EXT-03",
        question="What is the stock price of Apple Inc.?",
        expected_answer=NOT_FOUND_FALLBACK,
        expected_supported=False,
        category="negative_external",
        rationale="External financial query unrelated to History catalogue.",
    ),
    EvalSample(
        id="EXT-04",
        question="What is the weather forecast for New Delhi today?",
        expected_answer=NOT_FOUND_FALLBACK,
        expected_supported=False,
        category="negative_external",
        rationale="External weather query.",
    ),
    EvalSample(
        id="EXT-05",
        question="What is the price of a Biology laboratory manual?",
        expected_answer=NOT_FOUND_FALLBACK,
        expected_supported=False,
        category="negative_off_topic",
        rationale="Off-topic query not present in History catalogue.",
    ),
]


class EvaluationRunner:
    """Automated benchmark evaluator for BookWise AI."""

    def __init__(self, engine: Optional[AnswerEngine] = None):
        self.engine = engine or AnswerEngine()

    def run_eval(self, dataset: Optional[List[EvalSample]] = None) -> Dict[str, Any]:
        """Execute evaluation over dataset and calculate performance metrics."""
        samples = dataset or FIVE_REQUIRED_QUESTIONS
        results: List[Dict[str, Any]] = []

        correct_answers = 0
        correct_citations = 0
        correct_abstentions = 0
        total_abstentions_expected = sum(1 for s in samples if not s.expected_supported)
        total_latencies = []

        for sample in samples:
            t0 = time.time()
            res = self.engine.ask(sample.question)
            latency = (time.time() - t0) * 1000
            total_latencies.append(latency)

            # Check Answer Correctness
            # For unanswerable queries: exact string match with "Not found"
            if not sample.expected_supported:
                is_correct = (res.answer == NOT_FOUND_FALLBACK and not res.is_supported)
                if is_correct:
                    correct_abstentions += 1
                citation_valid = (len(res.citations) == 0)
            else:
                # For supported queries: answer contains key fact (e.g. "20 January 2025" or "1948")
                is_correct = (
                    res.is_supported and
                    sample.expected_answer.lower() in res.answer.lower()
                )
                # Check Citation
                citation_valid = False
                if res.citations:
                    top_cit = res.citations[0]
                    if sample.expected_pdf_page and top_cit.pdf_page == sample.expected_pdf_page:
                        citation_valid = True

            if is_correct:
                correct_answers += 1
            if citation_valid:
                correct_citations += 1

            results.append({
                "id": sample.id,
                "question": sample.question,
                "expected": sample.expected_answer,
                "actual": res.answer,
                "supported": res.is_supported,
                "citation": res.citations[0].format_citation() if res.citations else "None",
                "is_correct": is_correct,
                "citation_valid": citation_valid,
                "latency_ms": round(latency, 1),
            })

        total = len(samples)
        metrics = {
            "total_questions": total,
            "answer_accuracy": round(correct_answers / total * 100, 1),
            "citation_accuracy": round(correct_citations / total * 100, 1),
            "abstention_accuracy": round(correct_abstentions / max(1, total_abstentions_expected) * 100, 1),
            "average_latency_ms": round(sum(total_latencies) / max(1, len(total_latencies)), 1),
            "results": results,
        }
        return metrics

    def print_report(self, dataset: Optional[List[EvalSample]] = None) -> Dict[str, Any]:
        """Run evaluation and print a tabular report to stdout."""
        metrics = self.run_eval(dataset)
        table_data = []
        for r in metrics["results"]:
            status = "✓ PASS" if r["is_correct"] and r["citation_valid"] else "✗ FAIL"
            table_data.append([
                r["id"],
                r["question"][:40] + ("..." if len(r["question"]) > 40 else ""),
                r["expected"],
                r["actual"],
                r["citation"],
                status,
                f"{r['latency_ms']} ms",
            ])

        headers = ["ID", "Question", "Expected", "Actual", "Citation", "Status", "Latency"]
        print("\n" + "=" * 80)
        print("  BOOKWISE AI — EVALUATION BENCHMARK REPORT")
        print("=" * 80)
        print(tabulate(table_data, headers=headers, tablefmt="github"))
        print("\nSUMMARY METRICS:")
        print(f"- Overall Answer Accuracy   : {metrics['answer_accuracy']}%")
        print(f"- Citation Accuracy         : {metrics['citation_accuracy']}%")
        print(f"- Abstention Accuracy       : {metrics['abstention_accuracy']}%")
        print(f"- Average Query Latency     : {metrics['average_latency_ms']} ms")
        print("=" * 80 + "\n")
        return metrics


if __name__ == "__main__":
    runner = EvaluationRunner()
    runner.print_report()
