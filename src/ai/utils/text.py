"""Pure text normalization helpers shared across application packages."""

from __future__ import annotations

from typing import Any


def compact_text(value: Any) -> str:
    """Return scalar input as text with repeated whitespace collapsed."""
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def optional_text(value: Any) -> str | None:
    """Return trimmed scalar text or ``None`` for empty and complex values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def as_text_list(value: Any) -> list[str]:
    """Normalize a string or list of scalar values to non-empty text items."""
    if value is None:
        return []
    if isinstance(value, str):
        text = compact_text(value)
        return [text] if text else []
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := compact_text(item))]


def dedupe_texts(values: list[str]) -> list[str]:
    """Deduplicate normalized text case-insensitively while preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = compact_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
