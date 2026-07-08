"""JSON parsing helpers for persisted data and loosely formatted LLM output."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(value: object) -> dict[str, Any] | None:
    """Parse a JSON object from a string, returning ``None`` for invalid input."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_json_object(content: str) -> dict[str, Any] | None:
    """Extract the first usable JSON object from plain or fenced LLM output."""
    return _extract_json_container(content, opening="{", closing="}", expected_type=dict)


def extract_json_array(content: str) -> list[Any] | None:
    """Extract the first usable JSON array from plain or fenced LLM output."""
    return _extract_json_container(content, opening="[", closing="]", expected_type=list)


def _extract_json_container(
    content: str,
    *,
    opening: str,
    closing: str,
    expected_type: type[dict] | type[list],
) -> Any | None:
    text = (content or "").strip()
    if not text:
        return None

    fenced_match = re.search(
        rf"```(?:json)?\s*({re.escape(opening)}.*?{re.escape(closing)})\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = [fenced_match.group(1)] if fenced_match else []
    candidates.append(text)
    first_delimiter = text.find(opening)
    last_delimiter = text.rfind(closing)
    if 0 <= first_delimiter < last_delimiter:
        candidates.append(text[first_delimiter : last_delimiter + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, expected_type):
            return parsed
    return None
