from __future__ import annotations

from ai.agent.evaluation import AgentJudgeScore, build_agent_judge


class _FakeStructuredJudge:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AgentJudgeScore(
            task_completed=True,
            requirement_coverage=1.0,
            factual_accuracy=0.9,
            evidence_faithfulness=0.8,
            citation_correctness=0.7,
            comment="All requirements were addressed; one citation was indirect.",
        )


class _FakeChatModel:
    def __init__(self) -> None:
        self.structured_judge = _FakeStructuredJudge()

    def with_structured_output(self, schema, **kwargs):
        assert schema is AgentJudgeScore
        assert kwargs == {}
        return self.structured_judge


def test_agent_judge_emits_stable_quality_metrics_in_one_call() -> None:
    model = _FakeChatModel()
    judge = build_agent_judge(model)  # type: ignore[arg-type]

    evaluation = judge(
        {"objective": "审查付款条款"},
        {"report": "发票应在30日内支付 [S1]。", "citations": []},
        {
            "task_requirements": ["说明付款期限"],
            "expected_facts": ["发票应在30日内支付。"],
            "must_refuse": False,
        },
    )

    results = evaluation["results"]
    assert [result["key"] for result in results] == [
        "task_completion",
        "requirement_coverage",
        "factual_accuracy",
        "evidence_faithfulness",
        "citation_correctness",
    ]
    assert results[0]["score"] == 1.0
    assert len(model.structured_judge.messages) == 2
    assert "用户满意度" in model.structured_judge.messages[0].content
