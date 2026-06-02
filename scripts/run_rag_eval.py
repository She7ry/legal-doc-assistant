from __future__ import annotations

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
