from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock


class SlidingWindowRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max(1, int(max_requests))
        self._window_seconds = max(1, int(window_seconds))
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_start = now - self._window_seconds
        with self._lock:
            requests = self._requests[key]
            while requests and requests[0] <= window_start:
                requests.popleft()
            if len(requests) >= self._max_requests:
                return False
            requests.append(now)
            return True
