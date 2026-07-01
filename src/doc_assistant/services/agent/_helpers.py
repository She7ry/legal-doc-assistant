"""Agent 通用工具函数：文本清理、去重、引用格式化、进度回调、CitationRegistry。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from inspect import Parameter, signature
from typing import Any

from doc_assistant.config.settings import settings
from doc_assistant.schemas.citation import Citation
from doc_assistant.services.agent.schemas import (
    AgentFinding,
    AgentPlanStep,
    AgentStepResult,
)

SOURCE_REF_PATTERN = re.compile(r"\[([SCDPW]\d+)\]", re.IGNORECASE)
BARE_SOURCE_REF_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([SCDPW]\d+)(?![A-Za-z0-9])", re.IGNORECASE
)
ProgressCallback = Callable[..., None]


# ── 文本清理 ──────────────────────────────────────────────────────────────────


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        result = []
        for item in value:
            text = _clean_text(item)
            if text:
                result.append(text)
        return result
    return []


def _dedupe_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _first_text(values: list[str]) -> str:
    return values[0] if values else ""


def _mentions_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.casefold() in text for keyword in keywords)


# ── 引用编号与格式 ────────────────────────────────────────────────────────────


def _source_id_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    source_ids: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        for match in SOURCE_REF_PATTERN.finditer(item):
            source_id = match.group(1).upper()
            if source_id not in source_ids:
                source_ids.append(source_id)
        for match in BARE_SOURCE_REF_PATTERN.finditer(item):
            source_id = match.group(1).upper()
            if source_id not in source_ids:
                source_ids.append(source_id)
    return source_ids


def _format_refs(source_ids: list[str]) -> str:
    refs: list[str] = []
    for source_id in source_ids:
        if not isinstance(source_id, str):
            continue
        normalized = source_id.strip().strip("[]").upper()
        if re.fullmatch(r"S\d+", normalized) and normalized not in refs:
            refs.append(normalized)
    return " " + " ".join(f"[{source_id}]" for source_id in refs) if refs else ""


def _renumber_findings(findings: list[AgentFinding]) -> list[AgentFinding]:
    return [
        replace(finding, finding_id=f"f{index}")
        for index, finding in enumerate(findings, start=1)
    ]


def _remap_source_refs(text: str, mapping: dict[str, str]) -> str:
    def replace_match(match: re.Match[str]) -> str:
        source_id = match.group(1).upper()
        return f"[{mapping.get(source_id, source_id)}]"

    remapped = SOURCE_REF_PATTERN.sub(replace_match, text or "")

    def replace_bare_match(match: re.Match[str]) -> str:
        source_id = match.group(1).upper()
        return mapping.get(source_id, source_id)

    return BARE_SOURCE_REF_PATTERN.sub(replace_bare_match, remapped)


def _remap_metadata(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _remap_source_refs(value, mapping)
    if isinstance(value, list):
        return [_remap_metadata(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _remap_metadata(item, mapping) for key, item in value.items()}
    return value


def _metadata_missing_information(metadata: dict[str, Any]) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    missing = _as_text_list(metadata.get("missing_information"))
    evidence = metadata.get("evidence")
    if isinstance(evidence, dict):
        missing.extend(_as_text_list(evidence.get("missing_evidence")))
    return _dedupe_texts(missing)


# ── Citation 去重与步骤缺失信息 ───────────────────────────────────────────────


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    deduped: list[Citation] = []
    seen: set[tuple[str, ...]] = set()
    for citation in citations:
        key = (
            citation.source_id,
            citation.file_id,
            citation.document_key,
            citation.document_version,
            citation.page,
            citation.chunk_id,
            citation.file_name,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _step_missing_information(step: AgentStepResult) -> list[str]:
    return _as_text_list(step.output.get("missing_information"))


def _is_generated_no_evidence_missing(item: str) -> bool:
    return _clean_text(item).casefold().startswith(
        "no cited document evidence was found for step:"
    )


# ── 进度回调与序列化 ──────────────────────────────────────────────────────────


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    event_type: str,
    stage: str,
    progress: int,
    message: str,
    step_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    callback(
        event_type=event_type,
        stage=stage,
        progress=progress,
        message=message,
        step_id=step_id,
        payload=payload or {},
    )


def _plan_step_payload(step: AgentPlanStep) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "title": step.title,
        "purpose": step.purpose,
        "tool": step.tool,
        "arguments": step.arguments,
        "requires_confirmation": step.requires_confirmation,
    }


def _step_result_payload(step: AgentStepResult) -> dict[str, Any]:
    react_trace = step.output.get("react_trace")
    return {
        "step_id": step.step_id,
        "title": step.title,
        "tool": step.tool,
        "status": step.status,
        "citation_count": len(step.citations),
        "guard_warning_count": len(step.guard_warnings),
        "react_action_count": len(react_trace) if isinstance(react_trace, list) else 0,
    }


# ── 步骤历史管理 ──────────────────────────────────────────────────────────────


def _append_agent_step_history(
    history: list[dict[str, object]],
    step: AgentStepResult,
) -> list[dict[str, object]]:
    summary = _clean_text(step.summary)
    if len(summary) > 1200:
        summary = f"{summary[:1197]}..."
    content = (
        f"Completed agent step '{step.title}' using {step.tool}. "
        f"Status: {step.status}. Summary: {summary}"
    )
    updated = [*history, {"role": "assistant", "content": content}]
    window = max(1, settings.chat_history_window)
    return updated[-window:]


def _call_accepts_keyword(func: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == Parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in parameters
    )


# ── CitationRegistry ─────────────────────────────────────────────────────────


@dataclass
class _CitationRegistry:
    """跨步骤全局引用编号器。

    各 QA 步骤独立返回 S1、S2… 会在最终报告中冲突；
    此处统一重编号为全局 S1、S2…，并返回 mapping 供重写文本中的 [Sx]。

    P1-1: 改为 frozen=False dataclass 以支持 LangGraph checkpoint msgpack 序列化。
    """

    citations: list[Citation] = field(default_factory=list)

    def add_step_citations(
        self,
        step_id: str,
        citations: list[Citation],
    ) -> tuple[dict[str, str], list[Citation]]:
        """把本步骤的 citations 追加到全局列表，返回 old→new 的 [Sx] 映射。"""
        mapping: dict[str, str] = {}
        registered: list[Citation] = []
        for citation in citations:
            new_source_id = f"S{len(self.citations) + 1}"
            mapping[citation.source_id.upper()] = new_source_id
            mapped = replace(citation, source_id=new_source_id)
            self.citations.append(mapped)
            registered.append(mapped)
        return mapping, registered
