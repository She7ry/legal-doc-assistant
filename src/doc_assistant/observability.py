"""轻量可观测性：为检索/入库等操作记录耗时与错误日志。"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("doc_assistant")


@contextmanager
def traced_operation(operation: str, **context: Any) -> Iterator[None]:
    """上下文管理器：记录 operation 名称、耗时(ms) 与可选 context 字段到日志。"""
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "Operation failed",
            extra={
                "operation": operation,
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
                **context,
            },
            exc_info=True,
        )
        raise
    else:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Operation completed",
            extra={"operation": operation, "elapsed_ms": elapsed_ms, **context},
        )
