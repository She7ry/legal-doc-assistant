"""JSON parsing helpers for persisted data."""

from __future__ import annotations

import json
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
