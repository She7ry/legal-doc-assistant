from __future__ import annotations

import re
from hashlib import sha1

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

_IMPLICIT_PROFILE_MARKERS = (
    "i am ",
    "i'm ",
    "i work as",
    "my role is",
    "我是",
    "我从事",
    "我做",
    "我的职位是",
    "我的岗位是",
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
    explicit = any(marker.casefold() in normalized for marker in _EXPLICIT_WRITE_MARKERS)
    implicit_profile = _looks_like_implicit_profile(text)
    if not explicit and not implicit_profile:
        return []

    content = _strip_write_marker(text) if explicit else text
    intents = []
    for part in _split_memory_content(content):
        clean_part = _strip_write_marker(part)
        if len(clean_part) < 4:
            continue
        key = _infer_key(clean_part)
        memory_type = "preference" if _looks_like_preference(clean_part) else "fact"
        intents.append(
            MemoryWriteIntent(
                type=memory_type,  # type: ignore[arg-type]
                key=key,
                content=clean_part,
                value_json={"text": clean_part},
                source="explicit" if explicit else "inferred",
                confidence=0.95 if explicit else 0.75,
            )
        )
    return intents


def _looks_temporary(text: str) -> bool:
    normalized = text.casefold()
    return any(marker.casefold() in normalized for marker in _TEMPORARY_MARKERS)


def _looks_like_preference(text: str) -> bool:
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in _ANSWER_STYLE_TERMS) or any(
        marker in normalized for marker in ("prefer", "喜欢", "偏好", "希望")
    )


def _looks_like_implicit_profile(text: str) -> bool:
    normalized = text.casefold().strip()
    if not any(marker.casefold() in normalized for marker in _IMPLICIT_PROFILE_MARKERS):
        return False
    return any(
        term in normalized
        for term in (
            "role",
            "job",
            "company",
            "legal",
            "law",
            "ip",
            "知识产权",
            "法务",
            "律师",
            "公司",
            "岗位",
            "职位",
            "行业",
        )
    )


def _infer_key(content: str) -> str:
    normalized = content.casefold()
    if any(term.casefold() in normalized for term in _ANSWER_STYLE_TERMS):
        return "answer_style"
    if any(term in normalized for term in ("name", "call me", "称呼", "叫我")):
        return "form_of_address"
    if any(
        term in normalized
        for term in (
            "role",
            "job",
            "company",
            "business",
            "industry",
            "legal",
            "law",
            "业务",
            "背景",
            "职位",
            "岗位",
            "公司",
            "行业",
            "法务",
            "律师",
            "知识产权",
        )
    ):
        return "business_context"
    if any(term in normalized for term in ("prefer", "喜欢", "偏好", "希望")):
        return "user_preference"
    words = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", content.casefold())
    canonical = "_".join(words[:8])[:48]
    digest = sha1(" ".join(words).encode("utf-8")).hexdigest()[:10] if words else ""
    if canonical and digest:
        return f"{canonical}_{digest}"[:80]
    return "user_memory"


def _strip_write_marker(text: str) -> str:
    replacements = [
        r"^\s*(please\s+)?remember\s+(that\s+)?",
        r"^\s*(please\s+)?remember\s+my\s+",
        r"^\s*from\s+now\s+on[:,]?\s*",
        r"^\s*going\s+forward[:,]?\s*",
        r"^\s*always\s+answer\s*[:,]?\s*",
        r"^\s*请?帮?我?记住[：:\s]*",
        r"^\s*以后(都|请)?[：:\s]*",
        r"^\s*我的偏好是[：:\s]*",
        r"^\s*偏好是[：:\s]*",
    ]
    content = text.strip()
    for pattern in replacements:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    return content.strip(" ：:。.")


def _split_memory_content(content: str) -> list[str]:
    normalized = re.sub(
        r"\s+(?:and also|also|plus|and my)\s+",
        "；",
        content,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"(并且|而且|另外|同时)", "；", normalized)
    parts = re.split(r"[;；。]\s*|[，,]\s*(?=(?:我|my|i\s|i'm|以后|always|from now on))", normalized)
    return [part.strip(" ：:。.") for part in parts if part.strip(" ：:。.")]
