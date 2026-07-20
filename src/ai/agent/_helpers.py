"""Small helpers shared by the ReAct Agent adapter."""

from __future__ import annotations

import re
from typing import Any

from ai.utils.text import compact_text

_clean_text = compact_text

SOURCE_REF_PATTERN = re.compile(r"\[([SCDPW]\d+)\]", re.IGNORECASE)
BARE_SOURCE_REF_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?<!\[)([SCDPW]\d+)(?![A-Za-z0-9])(?!\])",
    re.IGNORECASE,
)


def _mentions_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.casefold() in text for keyword in keywords)


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
