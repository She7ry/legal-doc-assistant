"""JSON/LLM 输出解析与类型转换工具集。

从 LLM 返回的非结构化文本中提取 JSON、转换布尔/风险等级/冲突类型等；
由 ``qa_service``、``clause_review``、``conflict_check`` 共用。
"""

from __future__ import annotations

import json
import re
from typing import Any

from doc_assistant.schemas.citation import Citation
from doc_assistant.services.review_taxonomy import allowed_conflict_type_keys


def extract_json_object(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced_match.group(1)] if fenced_match else []
    candidates.append(text)
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    if isinstance(value, (int, float, bool)):
        return str(value)
    return default


def optional_str(value: Any) -> str | None:
    text = as_str(value)
    return text or None


def as_list_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        result = []
        for item in value:
            text = as_str(item)
            if text:
                result.append(text)
        return result
    return []


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "y", "found"}:
            return True
        if normalized in {"false", "no", "n", "not found"}:
            return False
    return None


def coerce_risk_level(value: Any) -> str:
    if not isinstance(value, str):
        return "Needs human review"
    normalized = value.strip().casefold().replace("_", " ")
    if "human" in normalized or "review" in normalized:
        return "Needs human review"
    for level in ("Low", "Medium", "High"):
        if normalized == level.casefold() or level.casefold() in normalized:
            return level
    return "Needs human review"


def coerce_conflict_status(value: Any) -> str:
    if not isinstance(value, str):
        return "Insufficient information"
    normalized = value.strip().casefold()
    if "potential" in normalized or ("conflict" in normalized and "no" not in normalized):
        return "Potential conflict"
    if "no conflict" in normalized or normalized == "none":
        return "No conflict found"
    return "Insufficient information"


def coerce_conflict_type(value: Any) -> str:
    if not isinstance(value, str):
        return "ambiguous_relationship"
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
    aliases = {
        "timeline_conflict": "deadline_mismatch",
        "time_conflict": "deadline_mismatch",
        "deadline_conflict": "deadline_mismatch",
        "amount_conflict": "amount_mismatch",
        "money_conflict": "amount_mismatch",
        "definition_conflict": "definition_mismatch",
        "scope_conflict": "scope_mismatch",
        "process_conflict": "process_mismatch",
        "procedural_conflict": "process_mismatch",
        "direct_conflict": "direct_contradiction",
        "contradiction": "direct_contradiction",
        "none": "none",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in allowed_conflict_type_keys():
        return normalized
    return "ambiguous_relationship"


def first_source_id(citations: list[Citation], prefix: str | None = None) -> str | None:
    for citation in citations:
        if not prefix or citation.source_id.startswith(prefix):
            return citation.source_id
    return None


def source_id_list(
    value: Any,
    citations: list[Citation],
    prefix: str | None = None,
) -> list[str]:
    valid_source_ids = {
        citation.source_id
        for citation in citations
        if citation.source_id and (not prefix or citation.source_id.startswith(prefix))
    }
    if not valid_source_ids:
        return []

    raw_values: list[Any]
    if value is None:
        raw_values = []
    elif isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]

    source_ids: list[str] = []
    for raw_value in raw_values:
        text = as_str(raw_value)
        if not text:
            continue
        for match in re.findall(r"\[?([SCDPW]\d+)\]?", text, flags=re.IGNORECASE):
            source_id = match.upper()
            if source_id in valid_source_ids and source_id not in source_ids:
                source_ids.append(source_id)
    return source_ids


def format_source_refs(source_ids: list[Any]) -> str:
    refs: list[str] = []
    for source_id in source_ids:
        if not isinstance(source_id, str):
            continue
        normalized = source_id.strip().strip("[]").upper()
        if re.fullmatch(r"[SCDPW]\d+", normalized) and normalized not in refs:
            refs.append(normalized)
    return " " + " ".join(f"[{source_id}]" for source_id in refs) if refs else ""


def citation_suffix(source_ids: list[Any], citations: list[Citation]) -> str:
    normalized_ids: list[str] = []
    valid_source_ids = {citation.source_id for citation in citations}
    for value in source_ids:
        for sid in source_id_list(value, citations):
            if sid not in normalized_ids:
                normalized_ids.append(sid)
    if not normalized_ids:
        fsi = first_source_id(citations)
        if fsi:
            normalized_ids.append(fsi)
    normalized_ids = [sid for sid in normalized_ids if sid in valid_source_ids]
    return format_source_refs(normalized_ids)


def risk_reason_list(value: Any, citations: list[Citation]) -> list[dict[str, str | None]]:
    default_citation = first_source_id(citations, prefix="S")
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    reasons: list[dict[str, str | None]] = []
    for item in raw_items:
        if isinstance(item, dict):
            reason = as_str(item.get("reason") or item.get("text") or item.get("issue"))
            cit = source_id_list(
                item.get("citation") or item.get("citations"),
                citations,
                prefix="S",
            )
            citation_id = cit[0] if cit else default_citation
        else:
            reason = as_str(item)
            citation_id = default_citation
        if reason:
            reasons.append({"reason": reason, "citation": citation_id})
    return reasons
