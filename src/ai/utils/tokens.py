"""Small, provider-neutral token budget helpers."""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil
from typing import Any

from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool

_CHARS_PER_TOKEN = 2.0  # Conservative for Chinese legal text.


def count_message_tokens(
    messages: Iterable[Any],
    *,
    tools: list[BaseTool | dict[str, Any]] | None = None,
) -> int:
    return count_tokens_approximately(
        messages,
        chars_per_token=_CHARS_PER_TOKEN,
        tools=tools,
    )


def count_text_tokens(value: str) -> int:
    return ceil(len(value) / _CHARS_PER_TOKEN)


def truncate_text_tokens(value: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = int(max_tokens * _CHARS_PER_TOKEN)
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def truncate_middle_text_tokens(value: str, max_tokens: int) -> str:
    if count_text_tokens(value) <= max_tokens:
        return value
    max_chars = max(0, int(max_tokens * _CHARS_PER_TOKEN))
    if max_chars <= 5:
        return value[:max_chars]
    head = (max_chars - 5) * 2 // 3
    tail = max_chars - 5 - head
    return value[:head].rstrip() + "\n...\n" + value[-tail:].lstrip()
