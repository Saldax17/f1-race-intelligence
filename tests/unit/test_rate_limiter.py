from f1_race_intelligence.ingestion.rate_limiter import SlidingWindowRateLimiter


class FakeClock:
    """Deterministic time/sleep double so tests don't actually wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_limiter(rps: float, rpm: float, clock: FakeClock) -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(
        requests_per_second=rps,
        requests_per_minute=rpm,
        time_func=clock.time,
        sleep_func=clock.sleep,
    )


def test_allows_calls_under_the_per_second_limit() -> None:
    clock = FakeClock()
    limiter = make_limiter(rps=3, rpm=30, clock=clock)

    for _ in range(3):
        limiter.acquire()

    assert clock.sleeps == []


def test_blocks_when_per_second_limit_is_exceeded() -> None:
    clock = FakeClock()
    limiter = make_limiter(rps=3, rpm=30, clock=clock)

    for _ in range(3):
        limiter.acquire()
    limiter.acquire()  # 4th call within the same second window

    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] > 0


def test_blocks_when_per_minute_limit_is_exceeded_even_if_per_second_ok() -> None:
    clock = FakeClock()
    # High per-second limit so only the per-minute window is the binding constraint.
    limiter = make_limiter(rps=1000, rpm=2, clock=clock)

    limiter.acquire()
    clock.now += 1  # stay within the 60s window
    limiter.acquire()
    clock.now += 1
    limiter.acquire()  # 3rd call should be blocked by the per-minute limit

    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] > 0


def test_old_timestamps_fall_out_of_the_window() -> None:
    clock = FakeClock()
    limiter = make_limiter(rps=1, rpm=30, clock=clock)

    limiter.acquire()
    clock.now += 1.5  # past the 1-second window
    limiter.acquire()

    assert clock.sleeps == []
