from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import NAMESPACE_URL, UUID, uuid5

from scripts.run_agent_eval import _aggregate_efficiency, _aggregate_scores, _sync_dataset


def test_agent_eval_aggregates_quality_and_efficiency() -> None:
    started = datetime.now(UTC)
    rows = [
        {
            "run": SimpleNamespace(
                start_time=started,
                end_time=started + timedelta(seconds=2),
                total_tokens=100,
                error=None,
                outputs={"tool_call_count": 1, "status": "completed"},
            ),
            "evaluation_results": {
                "results": [SimpleNamespace(key="task_completion", score=1.0)]
            },
        },
        {
            "run": SimpleNamespace(
                start_time=started,
                end_time=started + timedelta(seconds=4),
                error="failed",
                outputs={"tool_call_count": 3, "status": "needs_human_review"},
            ),
            "evaluation_results": SimpleNamespace(
                results=[SimpleNamespace(key="task_completion", score=0.0)]
            ),
        },
    ]

    assert _aggregate_scores(rows) == {"task_completion": 0.5}
    assert _aggregate_efficiency(rows) == {
        "case_count": 2,
        "autonomous_completion_rate": 0.5,
        "error_rate": 0.5,
        "average_latency_seconds": 3.0,
        "p95_latency_seconds": 4.0,
        "average_total_tokens": 100.0,
        "average_tool_call_count": 2.0,
    }


class _FakeLangSmithClient:
    def __init__(self, existing_ids=()) -> None:
        self.examples = None
        self.existing_ids = set(existing_ids)
        self.updates = None

    def has_dataset(self, *, dataset_name: str) -> bool:
        return False

    def create_dataset(self, dataset_name: str, **kwargs):
        return SimpleNamespace(id="dataset-id")

    def list_examples(self, *, dataset_id: str, example_ids):
        return [SimpleNamespace(id=example_id) for example_id in self.existing_ids]

    def update_examples(self, *, dataset_id: str, updates):
        self.updates = updates

    def create_examples(self, *, dataset_id: str, examples):
        self.examples = examples


def test_agent_eval_dataset_sync_uses_deterministic_example_ids() -> None:
    seed = {
        "version": "1",
        "description": "test",
        "cases": [
            {
                "id": "case-1",
                "inputs": {"objective": "Review"},
                "reference_outputs": {"expected_facts": ["fact"]},
                "metadata": {"split": "regression", "category": "contract"},
            }
        ],
    }
    first_client = _FakeLangSmithClient()
    second_client = _FakeLangSmithClient()

    _sync_dataset(first_client, seed, "dataset-name")
    _sync_dataset(second_client, seed, "dataset-name")

    first = first_client.examples[0]
    second = second_client.examples[0]
    assert isinstance(first["id"], UUID)
    assert first["id"] == second["id"]
    assert first["split"] == "regression"
    assert first["metadata"] == {"case_id": "case-1", "category": "contract"}

    existing_id = uuid5(NAMESPACE_URL, "dataset-name:case-1")
    existing_client = _FakeLangSmithClient(existing_ids=[existing_id])
    _sync_dataset(existing_client, seed, "dataset-name")
    assert existing_client.examples is None
    assert existing_client.updates[0]["id"] == existing_id
