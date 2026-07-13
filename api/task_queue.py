from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from doc_assistant.config.settings import settings

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_lock = Lock()
_submitted_keys: set[str] = set()


def submit_background_task(key: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    global _executor
    with _lock:
        if key in _submitted_keys:
            return False
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max(1, int(getattr(settings, "background_max_workers", 4))),
                thread_name_prefix="legal-doc-background",
            )
        _submitted_keys.add(key)
        try:
            future = _executor.submit(func, *args, **kwargs)
        except BaseException:
            _submitted_keys.discard(key)
            raise
    future.add_done_callback(lambda finished: _on_task_done(key, finished.exception()))
    return True


def shutdown_background_tasks(*, wait: bool = True) -> None:
    global _executor
    with _lock:
        executor = _executor
        _executor = None
    if executor is not None:
        executor.shutdown(wait=wait)


def _on_task_done(key: str, error: BaseException | None) -> None:
    with _lock:
        _submitted_keys.discard(key)
    if error is not None:
        logger.error(
            "Background task crashed",
            extra={"task_key": key},
            exc_info=(type(error), error, error.__traceback__),
        )
