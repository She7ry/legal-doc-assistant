"""Memory 注入 prompt 时的格式化与截断工具。"""

from __future__ import annotations

import math
import re
from datetime import datetime

from doc_assistant.memory._conflict import _superseded_from_content
from doc_assistant.memory.maintenance import _effective_confidence
from doc_assistant.memory.schemas import MemoryCandidate


def _format_memory_prompt_line(candidate: MemoryCandidate) -> str:
    memory = candidate.memory
    qualifier = f"{memory.key}"
    content = " ".join(memory.content.split())
    if len(content) > 500:
        content = f"{content[:497]}..."
    line = f"- {qualifier} ({memory.source}, confidence {memory.confidence:.2f}): {content}"
    previous = memory.superseded_from_content or _superseded_from_content(memory.value_json)
    if memory.superseded_conflicting and previous:
        previous = " ".join(previous.split())
        if len(previous) > 160:
            previous = f"{previous[:157]}..."
        line += f" Note: this preference was recently updated from '{previous}' to this value."
    return line


def _estimate_prompt_tokens(text: str) -> int:
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[㐀-鿿]", text))
    other_chars = max(0, len(text) - cjk_chars)
    return cjk_chars + math.ceil(other_chars / 4)


def _truncate_to_prompt_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if _estimate_prompt_tokens(text) <= max_tokens:
        return text
    suffix = "..."
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = f"{text[:mid].rstrip()}{suffix}"
        if _estimate_prompt_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _prompt_candidate_rank(candidate: MemoryCandidate) -> tuple[float, float, datetime, datetime]:
    memory = candidate.memory
    relevance = candidate.score if candidate.score is not None else 0.0
    recency = memory.last_accessed_at or memory.updated_at
    return (_effective_confidence(memory), relevance, recency, memory.created_at)
