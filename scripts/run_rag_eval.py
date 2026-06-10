from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from doc_assistant.evaluation.metrics import (
    aggregate_scores,
    score_generation_case,
    score_retrieval_case,
    source_candidate_from_citation,
    source_candidate_from_document,
)
from doc_assistant.retrieval.vector_store import DocumentVectorStore
from doc_assistant.services.qa_service import DocumentQAService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the starter RAG evaluation dataset.")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "data" / "eval" / "eval_dataset.json"),
        help="Path to eval_dataset.json.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "eval" / "latest_report.json"),
        help="Path to write the JSON report.",
    )
    parser.add_argument("--max-k", type=int, default=10, help="Maximum retrieval depth to score.")
    parser.add_argument(
        "--min-score",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help=(
            "Fail when a summary metric is below VALUE. "
            "Example: --min-score retrieval.at_5.recall=0.8 "
            "--min-score generation.citation_accuracy=0.95"
        ),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    vector_store = DocumentVectorStore(
        collection_name="legal_documents_eval",
        persist_directory=PROJECT_ROOT / "data" / "eval" / "vector_store",
    )
    service = DocumentQAService(vector_store=vector_store)

    ingest_results = []
    for document in dataset.get("documents", []):
        document_path = PROJECT_ROOT / document["path"]
        ingest_results.append(asdict(vector_store.ingest_file(document_path)))

    records: list[dict[str, Any]] = []
    retrieval_case_scores: dict[int, list[dict[str, float | None]]] = {5: [], 10: []}
    generation_case_scores: list[dict[str, float | None]] = []

    for case in dataset.get("cases", []):
        question = case["question"]

        retrieved_documents = vector_store.search(question, k=args.max_k)
        retrieved_candidates = [source_candidate_from_document(document) for document in retrieved_documents]

        case_retrieval_scores = {}
        for k in (5, 10):
            if k <= args.max_k:
                scores = score_retrieval_case(case.get("gold_sources") or [], retrieved_candidates, k)
                retrieval_case_scores[k].append(scores)
                case_retrieval_scores[f"retrieval_at_{k}"] = scores

        started_at = time.perf_counter()
        answer = service.ask(question)
        latency_seconds = time.perf_counter() - started_at

        citation_candidates = [source_candidate_from_citation(citation) for citation in answer.citations]
        generation_scores = score_generation_case(case, answer.content, citation_candidates)
        generation_case_scores.append(generation_scores)

        records.append(
            {
                "id": case["id"],
                "question": question,
                "answer": answer.content,
                "latency_seconds": latency_seconds,
                "retrieved_sources": [candidate.__dict__ for candidate in retrieved_candidates],
                "citations": [asdict(citation) for citation in answer.citations],
                "scores": {
                    **case_retrieval_scores,
                    "generation": generation_scores,
                },
            }
        )

    report = {
        "dataset": str(dataset_path),
        "ingest_results": ingest_results,
        "summary": {
            "retrieval": {
                f"at_{k}": aggregate_scores(scores) for k, scores in retrieval_case_scores.items()
            },
            "generation": aggregate_scores(generation_case_scores),
            "average_latency_seconds": (
                sum(record["latency_seconds"] for record in records) / len(records) if records else None
            ),
        },
        "records": records,
    }
    try:
        threshold_results = _evaluate_thresholds(report["summary"], args.min_score)
    except ValueError as exc:
        parser.error(str(exc))
    if threshold_results:
        report["thresholds"] = threshold_results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    failures = [result for result in threshold_results if not result["passed"]]
    if threshold_results:
        print(json.dumps({"thresholds": threshold_results}, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")
    if failures:
        print(_format_threshold_failures(failures), file=sys.stderr)
        raise SystemExit(1)


def _evaluate_thresholds(
    summary: dict[str, Any],
    threshold_args: list[str],
) -> list[dict[str, Any]]:
    return [
        _evaluate_threshold(summary, metric_path, minimum)
        for metric_path, minimum in (_parse_min_score(value) for value in threshold_args)
    ]


def _parse_min_score(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise ValueError("--min-score must use METRIC=VALUE, for example retrieval.at_5.recall=0.8")

    metric_path, raw_minimum = value.split("=", 1)
    metric_path = metric_path.strip()
    if not metric_path:
        raise ValueError("--min-score metric path cannot be empty.")

    try:
        minimum = float(raw_minimum)
    except ValueError as exc:
        raise ValueError(f"--min-score value for {metric_path!r} must be numeric.") from exc

    return metric_path, minimum


def _evaluate_threshold(
    summary: dict[str, Any],
    metric_path: str,
    minimum: float,
) -> dict[str, Any]:
    actual = _resolve_metric(summary, metric_path)
    passed = isinstance(actual, int | float) and float(actual) >= minimum
    return {
        "metric": metric_path,
        "minimum": minimum,
        "actual": actual,
        "passed": passed,
    }


def _resolve_metric(summary: dict[str, Any], metric_path: str) -> float | None:
    current: Any = summary
    for part in metric_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    if isinstance(current, int | float):
        return float(current)
    return None


def _format_threshold_failures(failures: list[dict[str, Any]]) -> str:
    lines = ["RAG evaluation thresholds failed:"]
    for failure in failures:
        lines.append(
            "- {metric}: actual={actual}, minimum={minimum}".format(
                metric=failure["metric"],
                actual=failure["actual"],
                minimum=failure["minimum"],
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
