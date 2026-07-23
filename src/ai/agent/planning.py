"""Bounded task planning and final synthesis helpers for the ReAct Agent."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any

from ai.agent._helpers import _clean_text
from ai.agent.schemas import AgentPlanOutput, AgentStepResult
from ai.config.settings import settings
from ai.llm import structured_chat_output
from ai.rag.qa_service import DocumentQAService
from ai.rag.schemas import Citation
from ai.utils.tokens import count_text_tokens, truncate_text_tokens

logger = logging.getLogger(__name__)

_COMPLEX_TASK = re.compile(
    r"[;；\n]|(?:以及|并且|同时|比较|对比|分别|两者|多个|跨文档)"
)


def is_l2_candidate(objective: str, focus_areas: list[str]) -> bool:
    normalized_focus = {_clean_text(value).casefold() for value in focus_areas}
    normalized_focus.discard("")
    return (
        len(normalized_focus) >= 2
        or len(objective) >= 160
        or bool(_COMPLEX_TASK.search(objective))
    )


def plan_task(
    qa_service: DocumentQAService,
    *,
    objective: str,
    focus_areas: list[str],
    user_role: str,
) -> tuple[list[tuple[str, str]], str]:
    normalized_focus = list(
        dict.fromkeys(area for area in (_clean_text(value) for value in focus_areas) if area)
    )
    if len(normalized_focus) >= 2:
        return [
            (
                area[:120],
                f"只分析“{area}”这一关注点，并形成可独立阅读的法律审阅发现。",
            )
            for area in normalized_focus[:5]
        ], "focus_areas"

    if len(objective) < 160 and not _COMPLEX_TASK.search(objective):
        return [], "single"

    try:
        output = AgentPlanOutput.model_validate(
            structured_chat_output(qa_service.chat_model, AgentPlanOutput).invoke(
                qa_service._build_messages(_planner_prompt(objective, normalized_focus, user_role))
            )
        )
    except Exception:
        logger.warning("Agent planning failed; falling back to single-step ReAct.", exc_info=True)
        return [], "single"

    steps: list[tuple[str, str]] = []
    seen: set[str] = set()
    for step in output.steps:
        title = _clean_text(step.title)
        instruction = _clean_text(step.instruction)
        key = instruction.casefold()
        if not title or not instruction or key in seen:
            continue
        seen.add(key)
        steps.append((title, instruction))
    if len(steps) < 2:
        logger.warning("Agent plan was invalid; falling back to single-step ReAct.")
        return [], "single"
    return steps[:5], "planner"


def renumber_citations(
    source_citations: list[Citation],
    citations_by_key: dict[tuple[Any, ...], Citation],
    citations: list[Citation],
) -> tuple[dict[str, str], list[Citation]]:
    mapping: dict[str, str] = {}
    step_citations: list[Citation] = []
    for citation in source_citations:
        key = _citation_key(citation)
        global_citation = citations_by_key.get(key)
        if global_citation is None:
            global_citation = replace(citation, source_id=f"D{len(citations) + 1}")
            citations_by_key[key] = global_citation
            citations.append(global_citation)
        mapping[citation.source_id.upper()] = global_citation.source_id
        if global_citation not in step_citations:
            step_citations.append(global_citation)
    return mapping, step_citations


def _citation_key(citation: Citation) -> tuple[Any, ...]:
    return (
        citation.source_type,
        citation.document_key or citation.file_id or citation.file_name,
        citation.document_version,
        citation.page,
        citation.chunk_id,
        citation.char_start,
        citation.char_end,
        citation.exact_quote or citation.preview,
    )


def synthesize_steps(
    qa_service: DocumentQAService,
    *,
    objective: str,
    user_role: str,
    steps: list[AgentStepResult],
    citations: list[Citation],
) -> tuple[str, list[str]]:
    prompt = _synthesis_prompt(objective, user_role, steps, citations)
    for attempt in range(1, 3):
        try:
            return qa_service._invoke_chat_messages(qa_service._build_messages(prompt)), []
        except Exception:
            logger.warning(
                "Final Agent synthesis failed%s.",
                "; retrying once" if attempt == 1 else " after one retry",
                extra={"attempt": attempt},
                exc_info=True,
            )
    return _fallback_report(steps), ["最终汇总重试一次后仍失败。"]


def step_question(objective: str, title: str, instruction: str, user_role: str) -> str:
    return (
        f"任务目标（仅作背景）：\n{objective.strip()}"
        f"\n\n当前独立步骤：{title}\n{instruction}"
        "\n\n只处理当前步骤，不要执行其他计划步骤，也不要汇总全部任务。"
        "使用可用工具，并引用检索到的证据。"
        f"\n\n目标读者：{user_role}"
    )


def _planner_prompt(objective: str, focus_areas: list[str], user_role: str) -> str:
    return (
        "将这个复合法律任务拆分为 2—5 个扁平、相互独立的步骤。"
        "每个步骤都必须能单独使用文档检索和法律审阅工具完成。"
        "不要创建步骤依赖、递归子步骤或最终汇总步骤。"
        "每个步骤只需返回标题和明确的执行指令。"
        "只返回合法 JSON 对象，格式为 {\"steps\":[{\"title\":\"...\",\"instruction\":\"...\"}]}。"
        f"\n\n任务目标：\n{objective.strip()}"
        f"\n\n已有关注点：\n{', '.join(focus_areas) or '无'}"
        f"\n\n目标读者：\n{user_role}"
    )


def _synthesis_prompt(
    objective: str,
    user_role: str,
    steps: list[AgentStepResult],
    citations: list[Citation],
) -> str:
    base = (
        "根据以下独立步骤摘要，生成一份简洁、完整的最终法律工作成果。"
        "只能使用步骤摘要和所给引用中的事实；必须原样保留引用 ID，并在每个实质性段落后引用证据。"
        "明确披露失败或需要人工审阅的步骤，不要提及工具调用轨迹。默认使用简体中文。"
        f"\n\n任务目标：\n{objective.strip()}"
        f"\n\n目标读者：\n{user_role}"
    )
    available = max(100, settings.chat_input_max_tokens - count_text_tokens(base) - 20)
    step_budget = available * 2 // 3
    citation_budget = available - step_budget
    per_step = max(1, step_budget // max(1, len(steps)))
    per_citation = max(1, citation_budget // max(1, len(citations)))

    step_blocks = []
    for step in steps:
        guard_status = "通过" if not step.guard_warnings else "; ".join(step.guard_warnings[:5])
        source_ids = ", ".join(citation.source_id for citation in step.citations) or "无"
        step_blocks.append(
            truncate_text_tokens(
                f"[{step.step_id}] {step.title}\n"
                f"状态：{step.status}\n校验：{guard_status}\n引用：{source_ids}\n"
                f"摘要：\n{step.summary}",
                per_step,
            )
        )
    citation_lines = [
        truncate_text_tokens(
            f"- {citation.source_id} | {citation.file_name}{citation.location_label()} | "
            f"{(citation.exact_quote or citation.preview).strip()}",
            per_citation,
        )
        for citation in citations
    ]
    step_context = "\n\n".join(step_blocks)
    citation_context = "\n".join(citation_lines) or "无"
    return base + f"\n\n步骤摘要：\n{step_context}\n\n可用引用：\n{citation_context}"


def _fallback_report(steps: list[AgentStepResult]) -> str:
    lines = ["最终汇总暂不可用，以下为各独立步骤的结果。"]
    for step in steps:
        lines.extend(("", f"## {step.title} ({step.status})", step.summary))
    return "\n".join(lines)
