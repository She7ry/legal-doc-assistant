"""记忆维护：过期清理、容量限制、置信度衰减。

所有函数是无副作用的模块级工具函数。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import MemoryRecord


def _default_expires_at(scope: str) -> datetime | None:
    now = datetime.now(timezone.utc)
    if scope == "session" and settings.memory_session_ttl_hours > 0:
        return now + timedelta(hours=settings.memory_session_ttl_hours)
    if scope == "task" and settings.memory_task_ttl_hours > 0:
        return now + timedelta(hours=settings.memory_task_ttl_hours)
    return None


def _effective_confidence(memory: MemoryRecord) -> float:
    half_life_days = settings.memory_decay_half_life_days
    if half_life_days <= 0:
        return memory.confidence
    last_signal = memory.last_accessed_at or memory.updated_at
    age_days = max(0.0, (datetime.now(timezone.utc) - last_signal).total_seconds() / 86400)
    decay_factor = math.pow(0.5, age_days / half_life_days)
    return memory.confidence * decay_factor


def _memory_retention_rank(memory: MemoryRecord) -> float:
    return _effective_confidence(memory) + min(memory.access_count, 20) * 0.01
