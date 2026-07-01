"""Memory 冲突检测、等价判断与反馈评分归一化。"""

from __future__ import annotations

from doc_assistant.memory.schemas import MemoryRecord

_LANGUAGE_SIGNAL_GROUPS: tuple[tuple[str, ...], tuple[str, ...]] = (
    ("chinese", "中文", "汉语", "普通话", "mandarin"),
    ("english", "英文", "英语"),
)
_DETAIL_SIGNAL_GROUPS: tuple[tuple[str, ...], tuple[str, ...]] = (
    ("concise", "brief", "short", "简洁", "简短", "精简"),
    ("detailed", "detail", "elaborate", "详细", "展开", "详尽"),
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _has_opposed_signal(
    previous: str,
    new: str,
    groups: tuple[tuple[str, ...], tuple[str, ...]],
) -> bool:
    previous_first = _contains_any(previous, groups[0])
    previous_second = _contains_any(previous, groups[1])
    new_first = _contains_any(new, groups[0])
    new_second = _contains_any(new, groups[1])
    return (previous_first and new_second and not new_first) or (
        previous_second and new_first and not new_second
    )


def _is_conflicting_memory_update(previous_content: str, new_content: str) -> bool:
    previous = previous_content.casefold()
    new = new_content.casefold()
    return _has_opposed_signal(previous, new, _LANGUAGE_SIGNAL_GROUPS) or _has_opposed_signal(
        previous,
        new,
        _DETAIL_SIGNAL_GROUPS,
    )


def _with_supersede_conflict_metadata(
    value_json: dict | None,
    previous_content: str,
) -> dict:
    metadata = dict(value_json or {})
    metadata["_superseded_conflicting"] = True
    metadata["_superseded_from"] = previous_content[:500]
    return metadata


def _superseded_from_content(value_json: dict | None) -> str | None:
    if not isinstance(value_json, dict):
        return None
    value = value_json.get("_superseded_from")
    return str(value) if value else None


def _is_equivalent_memory(
    memory: MemoryRecord,
    *,
    content: str,
    value_json: dict | None,
    visibility: str,
    permissions: tuple[str, ...],
    task_id: str | None,
    expires_at: object | None,
) -> bool:
    return (
        memory.content == content.strip()
        and _visible_value_json(memory.value_json) == _visible_value_json(value_json)
        and memory.visibility == visibility
        and memory.permissions == permissions
        and memory.task_id == task_id
        and memory.expires_at == expires_at
    )


def _visible_value_json(value_json: dict | None) -> dict | None:
    if not isinstance(value_json, dict):
        return value_json
    visible = {
        key: value
        for key, value in value_json.items()
        if not str(key).startswith("_superseded_")
    }
    return visible or None


def _normalize_feedback_rating(rating: int | str) -> int:
    if isinstance(rating, str):
        normalized = rating.strip().casefold()
        if normalized in {"positive", "+1", "1", "up", "thumbs_up"}:
            return 1
        if normalized in {"negative", "-1", "down", "thumbs_down"}:
            return -1
    if rating == 1:
        return 1
    if rating == -1:
        return -1
    raise ValueError("Feedback rating must be positive/negative or 1/-1.")


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))
