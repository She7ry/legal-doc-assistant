from __future__ import annotations

import re

from doc_assistant.memory.schemas import MemoryWriteIntent

_EXPLICIT_WRITE_MARKERS = (
    "remember",
    "remember that",
    "remember my",
    "from now on",
    "going forward",
    "always answer",
    "以后",
    "以后都",
    "以后请",
    "记住",
    "请记住",
    "帮我记住",
    "我的偏好",
    "偏好是",
)

_TEMPORARY_MARKERS = (
    "just this time",
    "for this question",
    "for this task",
    "本次",
    "这次",
    "当前问题",
    "临时",
)

_ANSWER_STYLE_TERMS = (
    "answer",
    "reply",
    "response",
    "style",
    "concise",
    "detailed",
    "language",
    "中文",
    "英文",
    "回答",
    "回复",
    "风格",
    "简洁",
    "详细",
    "实现",
)


def extract_memory_write_intents(user_text: str) -> list[MemoryWriteIntent]:
    """Return explicit long-term memory write intents from a user message.

    The policy is intentionally conservative: ordinary chat is not persisted as
    long-term memory unless the user clearly asks the assistant to remember it.
    """

    text = " ".join(user_text.split())
    if not text or _looks_temporary(text):
        return []

    normalized = text.casefold()
    if not any(marker.casefold() in normalized for marker in _EXPLICIT_WRITE_MARKERS):
        return []

    content = _strip_write_marker(text)
    if len(content) < 4:
        return []

    key = _infer_key(content)
    memory_type = "preference" if _looks_like_preference(content) else "fact"
    return [
        MemoryWriteIntent(
            type=memory_type,  # type: ignore[arg-type]
            key=key,
            content=content,
            value_json={"text": content},
            source="explicit",
            confidence=0.95,
        )
    ]


def _looks_temporary(text: str) -> bool:
    normalized = text.casefold()
    return any(marker.casefold() in normalized for marker in _TEMPORARY_MARKERS)


def _looks_like_preference(text: str) -> bool:
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in _ANSWER_STYLE_TERMS) or any(
        marker in normalized for marker in ("prefer", "喜欢", "偏好", "希望")
    )


def _infer_key(content: str) -> str:
    normalized = content.casefold()
    if any(term.casefold() in normalized for term in _ANSWER_STYLE_TERMS):
        return "answer_style"
    if any(term in normalized for term in ("name", "call me", "称呼", "叫我")):
        return "form_of_address"
    if any(term in normalized for term in ("role", "job", "company", "业务", "背景")):
        return "business_context"
    words = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", content.casefold())
    return "_".join(words[:5])[:80] or "user_memory"


def _strip_write_marker(text: str) -> str:
    replacements = [
        r"^\s*(please\s+)?remember\s+(that\s+)?",
        r"^\s*from\s+now\s+on[:,]?\s*",
        r"^\s*going\s+forward[:,]?\s*",
        r"^\s*请?帮?我?记住[：:\s]*",
        r"^\s*以后(都|请)?[：:\s]*",
        r"^\s*我的偏好是[：:\s]*",
        r"^\s*偏好是[：:\s]*",
    ]
    content = text.strip()
    for pattern in replacements:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    return content.strip(" ：:。.")

