from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langsmith import evaluate

from ai.agent.evaluation import build_agent_judge
from ai.agent.react_task import run_react_agent_task
from ai.config.settings import settings
from ai.llm import build_chat_model
from ai.observability import agent_trace_outputs, get_langsmith_client
from ai.rag.qa_service import DocumentQAService
from ai.rag.retrieval.vector_store import DocumentVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = PROJECT_ROOT / "data" / "eval" / "agent_eval_dataset.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync the Agent dataset and run a LangSmith LLM-as-Judge experiment."
    )
    parser.add_argument("--seed", default=str(DEFAULT_SEED))
    parser.add_argument("--dataset-name", help="Override the dataset name from the seed file.")
    parser.add_argument("--experiment-prefix", default="legal-agent")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument(
        "--max-average-latency-seconds",
        type=float,
        help="Fail if average root-run latency exceeds this value.",
    )
    parser.add_argument(
        "--min-score",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Fail if an average Judge score is below VALUE.",
    )
    args = parser.parse_args()
    if args.max_concurrency <= 0:
        parser.error("--max-concurrency must be greater than 0.")
    if args.max_average_latency_seconds is not None and args.max_average_latency_seconds <= 0:
        parser.error("--max-average-latency-seconds must be greater than 0.")
    if not settings.langsmith_api_key:
        parser.error("LANGSMITH_API_KEY is required.")
    for key_name, key_value in (
        ("DOC_ASSISTANT_CHAT_API_KEY or DEEPSEEK_API_KEY", settings.chat_api_key or settings.deepseek_api_key),
        ("DOC_ASSISTANT_EMBEDDING_API_KEY or DASHSCOPE_API_KEY", settings.embedding_api_key),
    ):
        if not key_value or not key_value.isascii():
            parser.error(f"{key_name} must be configured with a real ASCII API key.")

    thresholds = _parse_thresholds(args.min_score, parser)
    seed = json.loads(Path(args.seed).read_text(encoding="utf-8"))
    _validate_seed(seed)
    dataset_name = args.dataset_name or seed["name"]
    client = get_langsmith_client()
    _sync_dataset(client, seed, dataset_name)
    print(f"LangSmith dataset synced: {dataset_name} ({len(seed['cases'])} examples)")
    if args.sync_only:
        return

    vector_store = DocumentVectorStore(
        collection_name="legal_documents_agent_eval",
        persist_directory=PROJECT_ROOT / "data" / "eval" / "vector_store",
    )
    if not args.skip_ingest:
        for document in seed["documents"]:
            vector_store.ingest_file(PROJECT_ROOT / document["path"])
    service = DocumentQAService(vector_store=vector_store)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        result = run_react_agent_task(
            service,
            objective=inputs["objective"],
            focus_areas=inputs.get("focus_areas"),
            user_role=inputs.get("user_role", "ordinary"),
            max_steps=inputs.get("max_steps", 6),
            user_id="langsmith-eval",
        )
        return agent_trace_outputs(result)

    # ponytail: reuse the configured model until calibration shows self-judging bias matters.
    judge = build_agent_judge(build_chat_model())
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[judge],
        metadata={"dataset_version": seed["version"], "evaluation_method": "llm-as-judge"},
        experiment_prefix=args.experiment_prefix,
        max_concurrency=args.max_concurrency,
        client=client,
        blocking=True,
    )
    rows = list(results)
    quality = _aggregate_scores(rows)
    efficiency = _aggregate_efficiency(rows)
    summary = {"quality": quality, "efficiency": efficiency}
    print(f"LangSmith experiment: {results.experiment_name}")
    if results.url:
        print(f"Results: {results.url}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = [
        f"{metric}={quality.get(metric, 0):.3f} < {minimum:.3f}"
        for metric, minimum in thresholds.items()
        if quality.get(metric, 0) < minimum
    ]
    if (
        args.max_average_latency_seconds is not None
        and efficiency["average_latency_seconds"] > args.max_average_latency_seconds
    ):
        failures.append(
            "average_latency_seconds="
            f"{efficiency['average_latency_seconds']:.3f} > "
            f"{args.max_average_latency_seconds:.3f}"
        )
    if failures:
        raise SystemExit("Quality gate failed: " + "; ".join(failures))


def _validate_seed(seed: dict[str, Any]) -> None:
    required = {"version", "name", "documents", "cases"}
    missing = sorted(required - seed.keys())
    if missing:
        raise ValueError(f"Agent eval seed is missing: {', '.join(missing)}")
    if not seed["cases"]:
        raise ValueError("Agent eval seed must contain at least one case.")


def _sync_dataset(client, seed: dict[str, Any], dataset_name: str) -> None:
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name,
            description=seed.get("description"),
            metadata={"version": seed["version"]},
        )

    examples = []
    for case in seed["cases"]:
        metadata = dict(case.get("metadata", {}))
        split = metadata.pop("split", None)
        examples.append(
            {
                "id": uuid5(NAMESPACE_URL, f"{dataset_name}:{case['id']}"),
                "inputs": case["inputs"],
                "outputs": case["reference_outputs"],
                "metadata": {"case_id": case["id"], **metadata},
                "split": split,
            }
        )
    example_ids = [example["id"] for example in examples]
    existing_ids = {
        example.id
        for example in client.list_examples(dataset_id=dataset.id, example_ids=example_ids)
    }
    updates = [example for example in examples if example["id"] in existing_ids]
    creates = [example for example in examples if example["id"] not in existing_ids]
    if updates:
        client.update_examples(dataset_id=dataset.id, updates=updates)
    if creates:
        client.create_examples(dataset_id=dataset.id, examples=creates)


def _parse_thresholds(values: list[str], parser: argparse.ArgumentParser) -> dict[str, float]:
    thresholds = {}
    for value in values:
        try:
            metric, raw_minimum = value.split("=", 1)
            minimum = float(raw_minimum)
        except ValueError:
            parser.error(f"Invalid --min-score '{value}'; expected METRIC=VALUE.")
        if not metric or not 0 <= minimum <= 1:
            parser.error(f"Invalid --min-score '{value}'; VALUE must be between 0 and 1.")
        thresholds[metric] = minimum
    return thresholds


def _aggregate_scores(rows) -> dict[str, float]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        evaluation_results = row["evaluation_results"]
        results = (
            evaluation_results["results"]
            if isinstance(evaluation_results, dict)
            else evaluation_results.results
        )
        for result in results:
            key = result.get("key") if isinstance(result, dict) else result.key
            score = result.get("score") if isinstance(result, dict) else result.score
            if isinstance(score, int | float):
                scores[key].append(float(score))
    return {key: statistics.fmean(values) for key, values in sorted(scores.items())}


def _aggregate_efficiency(rows) -> dict[str, float | int | None]:
    latencies = [
        (row["run"].end_time - row["run"].start_time).total_seconds()
        for row in rows
        if row["run"].end_time is not None
    ]
    token_counts = [
        tokens
        for row in rows
        if (tokens := getattr(row["run"], "total_tokens", None)) is not None
    ]
    tool_call_counts = [
        outputs["tool_call_count"]
        for row in rows
        if isinstance(outputs := getattr(row["run"], "outputs", None), dict)
        and isinstance(outputs.get("tool_call_count"), int | float)
    ]
    autonomous_completions = [
        (getattr(row["run"], "outputs", None) or {}).get("status") == "completed" for row in rows
    ]
    sorted_latencies = sorted(latencies)
    p95_index = round(0.95 * (len(sorted_latencies) - 1)) if sorted_latencies else 0
    return {
        "case_count": len(rows),
        "autonomous_completion_rate": (
            statistics.fmean(autonomous_completions) if autonomous_completions else 0
        ),
        "error_rate": (
            sum(getattr(row["run"], "error", None) is not None for row in rows) / len(rows)
            if rows
            else 0
        ),
        "average_latency_seconds": statistics.fmean(latencies) if latencies else 0,
        "p95_latency_seconds": sorted_latencies[p95_index] if sorted_latencies else 0,
        "average_total_tokens": statistics.fmean(token_counts) if token_counts else None,
        "average_tool_call_count": (
            statistics.fmean(tool_call_counts) if tool_call_counts else None
        ),
    }


if __name__ == "__main__":
    main()
