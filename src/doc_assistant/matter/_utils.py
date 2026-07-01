"""通用数据强制转换与时间工具函数（无业务逻辑）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _clean_text(value: Any) -> str:
    """将任意值安全转为去冗余空白的纯文本字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _as_dict(value: Any) -> dict[str, Any]:
    """安全转为 dict，非 dict 类型返回空 dict。"""
    return value if isinstance(value, dict) else {}


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    """安全转为 dict 列表，过滤掉非 dict 元素。"""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_text_list(value: Any) -> list[str]:
    """安全转为非空文本列表。"""
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)]


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


def _dedupe_texts(values: list[str]) -> list[str]:
    """去重文本列表（大小写不敏感），保持首次出现顺序。"""
    result = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _utc_now() -> datetime:
    """返回当前 UTC 时间。"""
    return datetime.now(timezone.utc)


def _datetime_to_db(value: datetime | None) -> str | None:
    """将 datetime 转为 ISO 格式字符串用于 SQLite 存储。"""
    return value.isoformat() if value else None


def _datetime_from_db(value: str | None) -> datetime | None:
    """从 SQLite 文本字段解析 datetime，缺失时区则补为 UTC。"""
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
