"""对话摘要：规则 + LLM 两种路径，从对话历史中提取结构化摘要。

所有函数都是模块级无副作用函数，不持有 MemoryService 实例引用。
"""

from __future__ import annotations

import re
from typing import Any

from doc_assistant.memory.schemas import MemoryRecord, MessageRecord

_RULE_SUMMARY_MAX_CHARS = 2000


def _summarize_conversation(
    messages: list[MessageRecord],
    *,
    previous_summary: str | None = None,
) -> str:
    previous = _normalize_summary_text(previous_summary)
    usable_messages = [
        message
        for message in messages
        if message.role in {"user", "assistant"} and message.content.strip()
    ]
    if not usable_messages:
        return _ensure_summary_prefix(previous) if previous else ""

    source_text = "\n".join(
        [previous, *(message.content for message in usable_messages if message.content.strip())]
    ).strip()
    sections = ["Conversation summary:"]

    if previous:
        sections.append(
            _summary_section(
                "Established context",
                [_truncate_text(_strip_summary_prefix(previous), 650)],
            )
        )

    parties = _extract_legal_parties(source_text)
    if parties:
        sections.append(_summary_section("Key parties and entities", parties))

    dates = _extract_dates_and_deadlines(source_text)
    if dates:
        sections.append(_summary_section("Key dates and deadlines", dates))

    legal_context = _extract_text_snippets(source_text, _LEGAL_CONTEXT_KEYWORDS, limit=5)
    if legal_context:
        sections.append(_summary_section("Legal and document context", legal_context))

    concerns = _extract_message_snippets(
        usable_messages,
        roles={"user"},
        keywords=_CORE_ISSUE_KEYWORDS,
        limit=6,
    )
    if concerns:
        sections.append(_summary_section("User concerns and review scope", concerns))

    conclusions = _extract_message_snippets(
        usable_messages,
        roles={"assistant"},
        keywords=_CONCLUSION_KEYWORDS,
        limit=6,
    )
    if conclusions:
        sections.append(_summary_section("Findings and conclusions", conclusions))

    open_items = _extract_message_snippets(
        usable_messages,
        roles={"user", "assistant"},
        keywords=_OPEN_ITEM_KEYWORDS,
        limit=5,
    )
    if open_items:
        sections.append(_summary_section("Open questions and next steps", open_items))

    recent_entries = _conversation_summary_entries(usable_messages, max_entries=6)
    if recent_entries:
        sections.append(_summary_section("Recent exchange", recent_entries))

    summary = "\n".join(section for section in sections if section.strip())
    return _truncate_summary(summary)


_LEGAL_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&'.,-]*\s+){0,5}[A-Z][A-Za-z0-9&'.,-]*\s+"
    r"(?:Inc\.?|LLC|Ltd\.?|Limited|Corp\.?|Corporation|Company|Co\.?|LLP|LP|PLC|GmbH|S\.A\.)\b"
)
_DATE_PATTERNS = (
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(
        r"\b\d+\s+(?:business\s+)?(?:day|days|month|months|year|years|week|weeks)\b",
        re.IGNORECASE,
    ),
)
_LEGAL_CONTEXT_KEYWORDS = (
    "agreement",
    "contract",
    "msa",
    "nda",
    "dpa",
    "addendum",
    "governing law",
    "jurisdiction",
    "venue",
    "court",
    "document type",
)
_CORE_ISSUE_KEYWORDS = (
    "review",
    "risk",
    "issue",
    "concern",
    "focus",
    "clause",
    "indemn",
    "liability",
    "termination",
    "renewal",
    "notice",
    "governing law",
    "jurisdiction",
    "confidential",
    "data processing",
    "payment",
    "compliance",
)
_CONCLUSION_KEYWORDS = (
    "risk",
    "issue",
    "requires",
    "require",
    "must",
    "should",
    "recommend",
    "finding",
    "conclusion",
    "uncapped",
    "missing",
    "confirm",
    "notice",
    "governing law",
    "jurisdiction",
    "liability",
    "indemn",
)
_OPEN_ITEM_KEYWORDS = (
    "open",
    "next step",
    "confirm",
    "clarify",
    "missing",
    "unresolved",
    "follow up",
    "human review",
    "need",
)


def _summary_section(title: str, items: list[str]) -> str:
    cleaned = [_truncate_text(item, 320) for item in _unique_nonempty(items)]
    if not cleaned:
        return ""
    return "\n".join([f"{title}:", *(f"- {item}" for item in cleaned)])


def _extract_legal_parties(text: str, *, limit: int = 8) -> list[str]:
    parties = [
        _clean_summary_item(match.group(0))
        for match in _LEGAL_ENTITY_PATTERN.finditer(text)
    ]
    return _unique_nonempty(parties)[:limit]


def _extract_dates_and_deadlines(text: str, *, limit: int = 8) -> list[str]:
    dates: list[str] = []
    for pattern in _DATE_PATTERNS:
        dates.extend(_clean_summary_item(match.group(0)) for match in pattern.finditer(text))
    date_context = _extract_text_snippets(
        text,
        ("effective date", "expiration", "deadline", "notice", "renewal", "term"),
        limit=4,
    )
    return _unique_nonempty([*dates, *date_context])[:limit]


def _extract_text_snippets(
    text: str,
    keywords: tuple[str, ...],
    *,
    limit: int,
    max_length: int = 260,
) -> list[str]:
    snippets = []
    for sentence in _split_summary_sentences(text):
        lowered = sentence.casefold()
        if any(keyword in lowered for keyword in keywords):
            snippets.append(_summary_snippet(sentence, max_length=max_length))
    return _unique_nonempty(snippets)[:limit]


def _extract_message_snippets(
    messages: list[MessageRecord],
    *,
    roles: set[str],
    keywords: tuple[str, ...],
    limit: int,
    max_length: int = 260,
) -> list[str]:
    snippets = []
    for message in messages:
        if message.role not in roles:
            continue
        content = " ".join(message.content.split())
        if not content:
            continue
        lowered = content.casefold()
        if any(keyword in lowered for keyword in keywords):
            snippets.append(_summary_snippet(content, max_length=max_length))
    return _unique_nonempty(snippets)[:limit]


def _conversation_summary_entries(
    messages: list[MessageRecord],
    *,
    max_entries: int,
) -> list[str]:
    entries: list[str] = []
    pending_user: str | None = None
    for message in messages:
        snippet = _summary_snippet(message.content, max_length=260)
        if not snippet:
            continue
        if message.role == "user":
            if pending_user:
                entries.append(f"User asked: {pending_user}.")
            pending_user = snippet
            continue
        if message.role == "assistant":
            if pending_user:
                entries.append(f"User asked: {pending_user}. Assistant answered: {snippet}.")
                pending_user = None
            else:
                entries.append(f"Assistant noted: {snippet}.")
    if pending_user:
        entries.append(f"User asked: {pending_user}.")
    return entries[-max_entries:]


def _split_summary_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    parts = re.split(r"(?<=[.!?。！？])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _normalize_summary_text(summary: str | None) -> str:
    return " ".join(str(summary or "").split())


def _strip_summary_prefix(summary: str) -> str:
    prefix = "conversation summary:"
    stripped = summary.strip()
    if stripped.casefold().startswith(prefix):
        return stripped[len(prefix) :].strip()
    return stripped


def _ensure_summary_prefix(summary: str) -> str:
    if not summary:
        return ""
    if summary.casefold().startswith("conversation summary:"):
        return _truncate_summary(summary)
    return _truncate_summary(f"Conversation summary: {summary}")


def _truncate_summary(summary: str) -> str:
    return _truncate_text(summary, _RULE_SUMMARY_MAX_CHARS)


def _truncate_text(text: str, max_length: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max(0, max_length - 3)]}..."


def _clean_summary_item(item: str) -> str:
    return item.strip(" \t\r\n,;:.")


def _unique_nonempty(items: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_summary_item(" ".join(str(item or "").split()))
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _summarize_conversation_llm_structured(
    messages: list[MessageRecord],
    chat_model: object,
    *,
    previous_summary: str | None = None,
) -> str:
    previous = _normalize_summary_text(previous_summary)
    if not messages and previous:
        return _ensure_summary_prefix(_truncate_text(previous, 1200))
    if not messages:
        return ""

    transcript = "\n".join(
        f"{message.role}: {' '.join(message.content.split())[:300]}"
        for message in messages[-20:]
        if message.content.strip()
    )
    if not transcript and previous:
        return _ensure_summary_prefix(_truncate_text(previous, 1200))
    if not transcript:
        return ""

    previous_block = previous or "None."
    prompt = f"""Summarize this legal document review conversation for session memory.
Preserve concrete facts and decisions. Include, when available:
1. document or contract type
2. parties and roles
3. key dates, notice periods, deadlines, governing law, and jurisdiction
4. user review scope and core legal issues
5. findings, conclusions, unresolved questions, and next steps

Existing summary:
{previous_block}

New transcript:
{transcript}

Return a concise structured summary under 900 characters. Do not invent facts."""
    messages_payload = [
        {
            "role": "system",
            "content": (
                "You summarize legal document assistant conversations for durable session memory. "
                "Preserve concrete contract facts, parties, dates, obligations, user concerns, and open issues."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    summary = _invoke_text_model(chat_model, messages_payload)
    summary = " ".join(summary.split())
    if not summary:
        return ""
    if len(summary) > 1200:
        summary = f"{summary[:1197]}..."
    if summary.casefold().startswith("conversation summary:"):
        return summary
    return f"Conversation summary: {summary}"


def _invoke_text_model(chat_model: object, messages: list[dict[str, str]]) -> str:
    invoke_messages = getattr(chat_model, "invoke_messages", None)
    if callable(invoke_messages):
        response = invoke_messages(messages)
        if isinstance(response, dict):
            return str(response.get("content") or "")
        return str(getattr(response, "content", response))

    invoke = getattr(chat_model, "invoke", None)
    if callable(invoke):
        try:
            response = invoke(messages=messages)
        except TypeError:
            response = invoke(messages)
        return str(getattr(response, "content", response))

    raise ValueError("The configured chat model does not support conversation summarization.")


def _summary_snippet(content: str, *, max_length: int = 180) -> str:
    normalized = " ".join(content.split())
    if not normalized:
        return ""
    first_sentence = _first_sentence(normalized)
    if len(first_sentence) <= max_length:
        return first_sentence
    return f"{first_sentence[: max_length - 3]}..."


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    parts = [part.strip() for part in parts if part.strip()]
    return parts[0].strip() if parts else text.strip()


def _conversation_summary_key(conversation_id: str) -> str:
    return f"conversation_summary_{conversation_id[:40]}"


def _summary_message_count(memory: MemoryRecord | None) -> int:
    if memory is None or not isinstance(memory.value_json, dict):
        return 0
    raw_count = memory.value_json.get("message_count")
    try:
        return int(raw_count)
    except (TypeError, ValueError):
        return 0
