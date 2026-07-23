from __future__ import annotations

import time
from collections import deque
from threading import Lock


class SlidingWindowRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max(1, int(max_requests))
        self._window_seconds = max(1, int(window_seconds))
        self._requests: dict[str, deque[float]] = {}
        self._last_cleanup = time.monotonic()
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - self._window_seconds
        with self._lock:
            if now - self._last_cleanup >= self._window_seconds:
                for stored_key, stored_requests in list(self._requests.items()):
                    while stored_requests and stored_requests[0] <= window_start:
                        stored_requests.popleft()
                    if not stored_requests:
                        self._requests.pop(stored_key, None)
                self._last_cleanup = now

            requests = self._requests.setdefault(key, deque())
            while requests and requests[0] <= window_start:
                requests.popleft()
            if len(requests) >= self._max_requests:
                return False
            requests.append(now)
            return True
