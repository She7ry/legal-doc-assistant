from __future__ import annotations

from typing import TypeVar

from fastapi import HTTPException, status

_T = TypeVar("_T")


def require_found(value: _T | None, detail: str = "Not found.") -> _T:
    """Raise 404 if *value* is ``None``, otherwise return it unchanged."""
    if value is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    return value


def get_fields_set(body) -> set[str]:
    """Retrieve the set of explicitly-provided fields from a Pydantic model."""
    return getattr(body, "model_fields_set", getattr(body, "__fields_set__", set()))
