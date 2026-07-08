"""Matter 数据强制转换工具函数。"""

from __future__ import annotations

from typing import Any

from doc_assistant.utils.text import (
    as_text_list,
    compact_text,
    dedupe_texts,
)

_clean_text = compact_text
_as_text_list = as_text_list
_dedupe_texts = dedupe_texts


def _as_dict(value: Any) -> dict[str, Any]:
    """安全转为 dict，非 dict 类型返回空 dict。"""
    return value if isinstance(value, dict) else {}


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    """安全转为 dict 列表，过滤掉非 dict 元素。"""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_bool(value: Any, *, default: bool) -> bool:
    """宽松的布尔值解析，支持 bool / int / str 类型。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default
