from __future__ import annotations

from doc_assistant.evaluation.metrics import (
    SourceCandidate,
    aggregate_scores,
    score_generation_case,
    score_retrieval_case,
)


def test_retrieval_metrics_score_gold_marker_rank() -> None:
    gold_sources = [{"file_name": "contract.pdf", "page": 1, "chunk_id": 3, "marker": "EVAL-C-12.2"}]
    retrieved = [
        SourceCandidate(file_name="contract.pdf", page=0, chunk_id=1, text="Section 5"),
        SourceCandidate(file_name="contract.pdf", page=1, chunk_id=3, text="Marker: EVAL-C-12.2."),
    ]

    scores = score_retrieval_case(gold_sources, retrieved, k=5)

    assert scores == {
        "recall": 1.0,
        "hit": 1.0,
        "precision": 0.2,
        "mrr": 0.5,
    }


def test_generation_metrics_score_answer_and_citation() -> None:
    case = {
        "answer_type": "answerable",
        "required_answer_terms": ["10%", "delayed shipment value"],
        "forbidden_answer_terms": ["20%"],
        "gold_sources": [{"file_name": "contract.pdf", "page": 1, "chunk_id": 3, "marker": "EVAL-C-12.2"}],
    }
    citations = [
        SourceCandidate(
            source_id="S1",
            file_name="contract.pdf",
            page=1,
            chunk_id=3,
            text="Marker: EVAL-C-12.2. Liquidated damages are capped at 10% of the delayed shipment value.",
        )
    ]

    scores = score_generation_case(
        case,
        "Liquidated damages are capped at 10% of the delayed shipment value [S1].",
        citations,
    )

    assert scores == {
        "answer_correctness": 1.0,
        "faithfulness": 1.0,
        "citation_accuracy": 1.0,
        "refusal_accuracy": None,
    }


def test_refusal_accuracy_scores_unanswerable_case() -> None:
    case = {
        "answer_type": "unanswerable",
        "required_refusal_terms": ["not found"],
        "gold_sources": [],
    }

    scores = score_generation_case(case, "The relevant text was not found in the indexed documents.", [])

    assert scores["answer_correctness"] == 1.0
    assert scores["faithfulness"] == 1.0
    assert scores["citation_accuracy"] is None
    assert scores["refusal_accuracy"] == 1.0


def test_aggregate_scores_ignores_not_applicable_values() -> None:
    aggregate = aggregate_scores(
        [
            {"recall": 1.0, "refusal_accuracy": None},
            {"recall": 0.0, "refusal_accuracy": 1.0},
        ]
    )

    assert aggregate == {"recall": 0.5, "refusal_accuracy": 1.0}

