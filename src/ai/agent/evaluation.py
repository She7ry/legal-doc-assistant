"""LangSmith LLM-as-Judge evaluator for complete Agent tasks."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ai.llm import structured_chat_output


class AgentJudgeScore(BaseModel):
    """One Judge call produces all quality signals used by the experiment."""

    task_completed: bool = Field(
        description="Whether every material task requirement was completed or correctly refused."
    )
    requirement_coverage: float = Field(ge=0, le=1)
    factual_accuracy: float = Field(ge=0, le=1)
    evidence_faithfulness: float = Field(ge=0, le=1)
    citation_correctness: float = Field(ge=0, le=1)
    comment: str = Field(description="Concise evidence-based explanation of the scores.")


JUDGE_SYSTEM_PROMPT = """你是严格的中文法律文档 Agent 评估器。
只评估任务完成度与答案准确性，不评价文风、语气或用户满意度。将 Agent 输出与参考要求和事实比较。

评分规则：
- 仅当所有实质性要求均已处理时，task_completed 才为 true；must_refuse 为 true 时，正确拒答也算完成。
- requirement_coverage 表示已处理的实质性要求占比。
- factual_accuracy 评价法律事实、数字、条件和结论是否正确。
- evidence_faithfulness 评价主张是否得到所提供引用摘录的支持。
- citation_correctness 评价引用是否指向预期来源并支持对应主张；正确拒答且无需引用时可得 1.0。
- 对无依据补充和矛盾内容扣分，不因内容看似合理而给分。
- 分数范围为 0.0 至 1.0，并在 comment 中说明具体遗漏或错误。
- 仅返回包含 task_completed、requirement_coverage、factual_accuracy、evidence_faithfulness、
  citation_correctness 和 comment 的 JSON 对象。
"""


def build_agent_judge(chat_model: BaseChatModel):
    """Create a LangSmith evaluator that emits five stable feedback keys."""
    structured_judge = structured_chat_output(chat_model, AgentJudgeScore)

    def evaluate_agent(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        payload = {
            "task_input": inputs,
            "agent_output": outputs,
            "reference": reference_outputs,
        }
        raw_score = structured_judge.invoke(
            [
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
            ]
        )
        score = AgentJudgeScore.model_validate(raw_score)
        values = {
            "task_completion": float(score.task_completed),
            "requirement_coverage": score.requirement_coverage,
            "factual_accuracy": score.factual_accuracy,
            "evidence_faithfulness": score.evidence_faithfulness,
            "citation_correctness": score.citation_correctness,
        }
        return {
            "results": [
                {"key": key, "score": value, "comment": score.comment}
                for key, value in values.items()
            ]
        }

    return evaluate_agent


__all__ = ["AgentJudgeScore", "build_agent_judge"]
