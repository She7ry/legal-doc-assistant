from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from doc_assistant.config.settings import settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=max(1, int(getattr(settings, "background_max_workers", 4))))
_lock = Lock()
_submitted_keys: set[str] = set()


def submit_background_task(key: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    with _lock:
        if key in _submitted_keys:
            return False
        _submitted_keys.add(key)

    future = _executor.submit(func, *args, **kwargs)
    future.add_done_callback(lambda finished: _on_task_done(key, finished.exception()))
    return True


def _on_task_done(key: str, error: BaseException | None) -> None:
    with _lock:
        _submitted_keys.discard(key)
    if error is not None:
        logger.error(
            "Background task crashed",
            extra={"task_key": key},
            exc_info=(type(error), error, error.__traceback__),
        )
