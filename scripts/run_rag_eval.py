from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai.config.settings import settings
from ai.rag.evaluation.metrics import (
    aggregate_scores,
    score_generation_case,
    score_retrieval_case,
    source_candidate_from_citation,
    source_candidate_from_document,
)
from ai.rag.qa_service import DocumentQAService
from ai.rag.retrieval.vector_store import (
    INGESTION_CHUNK_SEPARATORS,
    DocumentVectorStore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_SCORE_KEYS = ("hit", "mrr", "ndcg", "precision", "recall")
GENERATION_SCORE_KEYS = (
    "answer_correctness",
    "citation_accuracy",
    "faithfulness",
    "refusal_accuracy",
    "unsupported_claim_count",
)


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
        "--clean",
        action="store_true",
        help="Delete the eval vector store before ingesting documents.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse the existing eval vector store and skip document ingestion.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of evaluation cases to run concurrently.",
    )
    parser.add_argument(
        "--baseline",
        help="Optional path to a previous report JSON to compare against.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when --baseline detects a numeric metric regression.",
    )
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
    if args.max_k <= 0:
        parser.error("--max-k must be greater than 0.")
    if args.concurrency <= 0:
        parser.error("--concurrency must be greater than 0.")
    if args.clean and args.skip_ingest:
        parser.error("--clean cannot be combined with --skip-ingest.")
    for key_name, key_value in (
        ("DOC_ASSISTANT_CHAT_API_KEY", str(settings.chat_api_key)),
        ("DOC_ASSISTANT_EMBEDDING_API_KEY", str(settings.embedding_api_key)),
    ):
        if key_value and not key_value.isascii():
            parser.error(f"{key_name} must contain only ASCII header characters.")

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    chunking_validation = _validate_dataset_chunking(dataset)

    vector_store_path = PROJECT_ROOT / "data" / "eval" / "vector_store"
    if args.clean:
        _clean_eval_vector_store(vector_store_path)
    vector_store = DocumentVectorStore(
        collection_name="legal_documents_eval",
        persist_directory=vector_store_path,
    )
    service = DocumentQAService(vector_store=vector_store)

    ingest_results = []
    if not args.skip_ingest:
        for document in dataset.get("documents", []):
            document_path = PROJECT_ROOT / document["path"]
            ingest_results.append(asdict(vector_store.ingest_file(document_path)))

    records: list[dict[str, Any]] = []
    score_ks = [k for k in (5, 10) if k <= args.max_k]
    retrieval_case_scores: dict[int, list[dict[str, float | None]]] = {k: [] for k in score_ks}
    generation_case_scores: list[dict[str, float | None]] = []

    cases = [_case_with_dataset_defaults(case, dataset) for case in dataset.get("cases", [])]
    case_results = asyncio.run(
        _evaluate_cases(
            cases,
            vector_store=vector_store,
            service=service,
            max_k=args.max_k,
            score_ks=score_ks,
            concurrency=args.concurrency,
        )
    )
    for result in case_results:
        records.append(result["record"])
        generation_case_scores.append(result["generation_scores"])
        for k, scores in result["retrieval_scores"].items():
            retrieval_case_scores[k].append(scores)

    latencies = [
        float(record["latency_seconds"])
        for record in records
        if isinstance(record.get("latency_seconds"), int | float)
    ]
    error_records = [record for record in records if record.get("status") == "error"]
    report = {
        "dataset": str(dataset_path),
        "run_config": {
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "dataset_version": dataset.get("version"),
            "chat_provider": settings.chat_provider,
            "chat_model": settings.chat_model_name,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model_name,
            "retrieval_mode": settings.retrieval_mode,
            "top_k": settings.top_k,
        },
        "ingest_results": ingest_results,
        "chunking_validation": chunking_validation,
        "summary": {
            "retrieval": {
                f"at_{k}": aggregate_scores(scores, keys=RETRIEVAL_SCORE_KEYS)
                for k, scores in retrieval_case_scores.items()
            },
            "generation": aggregate_scores(generation_case_scores, keys=GENERATION_SCORE_KEYS),
            "average_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
            "evaluation_error_count": len(error_records),
        },
        "records": records,
    }
    baseline_comparison = None
    if args.baseline:
        baseline_report = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline_comparison = _compare_with_baseline(report, baseline_report)
        report["baseline_comparison"] = baseline_comparison

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
    if baseline_comparison:
        print(json.dumps({"baseline_comparison": baseline_comparison}, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")
    if failures:
        print(_format_threshold_failures(failures), file=sys.stderr)
        raise SystemExit(1)
    if error_records:
        print(
            f"RAG evaluation failed for {len(error_records)} case(s); see report records.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.fail_on_regression and baseline_comparison:
        if not baseline_comparison["comparable"]:
            print(
                "RAG evaluation baseline is not comparable:\n- "
                + "\n- ".join(baseline_comparison["compatibility_issues"]),
                file=sys.stderr,
            )
            raise SystemExit(1)
        if baseline_comparison["regressions"]:
            print(_format_regressions(baseline_comparison["regressions"]), file=sys.stderr)
            raise SystemExit(1)


async def _evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    vector_store: DocumentVectorStore,
    service: DocumentQAService,
    max_k: int,
    score_ks: list[int],
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(
                _evaluate_case,
                case,
                vector_store=vector_store,
                service=service,
                max_k=max_k,
                score_ks=score_ks,
            )

    return await asyncio.gather(*(run_case(case) for case in cases))


def _evaluate_case(
    case: dict[str, Any],
    *,
    vector_store: DocumentVectorStore,
    service: DocumentQAService,
    max_k: int,
    score_ks: list[int],
) -> dict[str, Any]:
    question = case["question"]
    case_retrieval_scores = {}
    retrieved_candidates = []
    try:
        retrieved_documents = vector_store.search(question, k=max_k)
        retrieved_candidates = [
            source_candidate_from_document(document) for document in retrieved_documents
        ]
        retrieval_scores = {}
        for k in score_ks:
            scores = score_retrieval_case(case.get("gold_sources") or [], retrieved_candidates, k)
            retrieval_scores[k] = scores
            case_retrieval_scores[f"retrieval_at_{k}"] = scores
    except Exception as exc:
        retrieval_scores = _empty_retrieval_scores(score_ks)
        return _error_case_result(
            case,
            stage="retrieval",
            error=exc,
            retrieved_candidates=retrieved_candidates,
            retrieval_scores=retrieval_scores,
            case_retrieval_scores=case_retrieval_scores,
        )

    started_at = time.perf_counter()
    try:
        answer = service.ask(question)
        latency_seconds = time.perf_counter() - started_at
        citation_candidates = [
            source_candidate_from_citation(citation) for citation in answer.citations
        ]
        generation_scores = score_generation_case(
            case,
            answer.content,
            citation_candidates,
            answer.metadata,
        )
        record = {
            "id": case["id"],
            "question": question,
            "category": case.get("category"),
            "tags": case.get("tags", []),
            "status": "ok",
            "answer": answer.content,
            "latency_seconds": latency_seconds,
            "retrieved_sources": [candidate.__dict__ for candidate in retrieved_candidates],
            "citations": [asdict(citation) for citation in answer.citations],
            "answer_metadata": answer.metadata,
            "scores": {
                **case_retrieval_scores,
                "generation": generation_scores,
            },
        }
        return {
            "record": record,
            "retrieval_scores": retrieval_scores,
            "generation_scores": generation_scores,
        }
    except Exception as exc:
        latency_seconds = time.perf_counter() - started_at
        return _error_case_result(
            case,
            stage="generation",
            error=exc,
            retrieved_candidates=retrieved_candidates,
            retrieval_scores=retrieval_scores,
            case_retrieval_scores=case_retrieval_scores,
            latency_seconds=latency_seconds,
        )


def _error_case_result(
    case: dict[str, Any],
    *,
    stage: str,
    error: Exception,
    retrieved_candidates: list[Any],
    retrieval_scores: dict[int, dict[str, float | None]],
    case_retrieval_scores: dict[str, dict[str, float | None]],
    latency_seconds: float | None = None,
) -> dict[str, Any]:
    generation_scores = {key: None for key in GENERATION_SCORE_KEYS}
    record = {
        "id": case["id"],
        "question": case["question"],
        "category": case.get("category"),
        "tags": case.get("tags", []),
        "status": "error",
        "answer": "",
        "latency_seconds": latency_seconds,
        "retrieved_sources": [candidate.__dict__ for candidate in retrieved_candidates],
        "citations": [],
        "error": {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        },
        "scores": {
            **case_retrieval_scores,
            "generation": generation_scores,
        },
    }
    return {
        "record": record,
        "retrieval_scores": retrieval_scores,
        "generation_scores": generation_scores,
    }


def _empty_retrieval_scores(score_ks: list[int]) -> dict[int, dict[str, float | None]]:
    return {k: {key: None for key in RETRIEVAL_SCORE_KEYS} for k in score_ks}


def _case_with_dataset_defaults(
    case: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(case)
    if (
        merged.get("answer_type") == "unanswerable"
        and "required_refusal_terms" not in merged
        and dataset.get("default_refusal_terms")
    ):
        merged["required_refusal_terms"] = list(dataset["default_refusal_terms"])
    return merged


def _clean_eval_vector_store(path: Path) -> None:
    target = path.resolve()
    eval_root = (PROJECT_ROOT / "data" / "eval").resolve()
    if target != eval_root and eval_root not in target.parents:
        raise ValueError(f"Refusing to clean non-eval vector store path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _validate_dataset_chunking(dataset: dict[str, Any]) -> dict[str, Any]:
    expected = dataset.get("chunking")
    current = _current_chunking_metadata()
    if not isinstance(expected, dict):
        return {
            "status": "missing",
            "current": current,
            "message": "Dataset does not declare chunking metadata.",
        }
    return {
        "status": "ok" if expected.get("config_hash") == current["config_hash"] else "mismatch",
        "expected": expected,
        "current": current,
    }


def _current_chunking_metadata() -> dict[str, Any]:
    payload = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "separators": list(INGESTION_CHUNK_SEPARATORS),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return {**payload, "config_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def _compare_with_baseline(
    current_report: dict[str, Any],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    current_summary = current_report["summary"]
    baseline_summary = baseline_report.get("summary", baseline_report)
    current_config = current_report.get("run_config", {})
    baseline_config = baseline_report.get("run_config", {})
    compatibility_issues = []
    for key in (
        "dataset_sha256",
        "chat_provider",
        "chat_model",
        "embedding_provider",
        "embedding_model",
        "retrieval_mode",
        "top_k",
    ):
        if key not in baseline_config:
            compatibility_issues.append(f"Baseline does not record {key}.")
        elif baseline_config.get(key) != current_config.get(key):
            compatibility_issues.append(
                f"Run configuration differs for {key}: "
                f"baseline={baseline_config.get(key)!r}, current={current_config.get(key)!r}."
            )
    changes = []
    regressions = []
    for metric_path, current_value in _numeric_leaves(current_summary):
        baseline_value = _resolve_metric(baseline_summary, metric_path)
        if baseline_value is None:
            continue
        delta = current_value - baseline_value
        change = {
            "metric": metric_path,
            "baseline": baseline_value,
            "current": current_value,
            "delta": delta,
        }
        changes.append(change)
        lower_is_better = metric_path.endswith(
            (
                "latency_seconds",
                "unsupported_claim_count",
                "evaluation_error_count",
            )
        )
        regressed = delta > 0 if lower_is_better else delta < 0
        if regressed:
            regressions.append(change)
    return {
        "comparable": not compatibility_issues,
        "compatibility_issues": compatibility_issues,
        "changes": changes,
        "regressions": regressions,
    }


def _numeric_leaves(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    if isinstance(value, dict):
        leaves = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(_numeric_leaves(child, child_prefix))
        return leaves
    if isinstance(value, int | float):
        return [(prefix, float(value))]
    return []


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
            f"- {failure['metric']}: actual={failure['actual']}, "
            f"minimum={failure['minimum']}"
        )
    return "\n".join(lines)


def _format_regressions(regressions: list[dict[str, Any]]) -> str:
    lines = ["RAG evaluation baseline regressions detected:"]
    for regression in regressions:
        lines.append(
            f"- {regression['metric']}: baseline={regression['baseline']}, "
            f"current={regression['current']}, delta={regression['delta']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
