"""Client-side rate limiting.

OpenF1's free historical-data tier enforces two simultaneous limits
(e.g. 3 requests/second AND 30 requests/minute). A naive ``time.sleep(1)``
between calls only respects one window and still bursts past the other,
so this implements a sliding-window rate limiter that tracks recent call
timestamps per window and blocks just long enough to stay under every
configured limit.

It is thread-safe (guarded by a single lock) so that it can be shared
safely if multiple callers ever use the same :class:`F1Client` instance
concurrently, even though the current client issues requests sequentially.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Tuple


class SlidingWindowRateLimiter:
    """Blocks callers as needed to respect one or more requests-per-window limits."""

    def __init__(
        self,
        requests_per_second: float,
        requests_per_minute: float,
        *,
        time_func: Callable[[], float] = time.monotonic,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        self._windows: List[Tuple[float, float]] = [
            (1.0, requests_per_second),
            (60.0, requests_per_minute),
        ]
        self._timestamps: Dict[float, Deque[float]] = {window: deque() for window, _ in self._windows}
        self._lock = threading.Lock()
        self._time = time_func
        self._sleep = sleep_func

    def acquire(self) -> None:
        """Block, if necessary, until a call can be made without breaching any limit."""
        while True:
            with self._lock:
                now = self._time()
                wait_time = 0.0

                for window_seconds, limit in self._windows:
                    timestamps = self._timestamps[window_seconds]
                    while timestamps and now - timestamps[0] >= window_seconds:
                        timestamps.popleft()
                    if len(timestamps) >= limit:
                        wait_time = max(wait_time, window_seconds - (now - timestamps[0]))

                if wait_time <= 0:
                    for window_seconds, _ in self._windows:
                        self._timestamps[window_seconds].append(now)
                    return

            self._sleep(wait_time)
