"""Injectable time sources so hardware loops can be tested without waiting."""

import time
from typing import Protocol


class Clock(Protocol):
    """Monotonic time plus sleep."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary, never-decreasing origin."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for `seconds` (no-op for non-positive values)."""
        ...


class SystemClock:
    """Wall-clock implementation backed by `time.monotonic` and `time.sleep`."""

    def monotonic(self) -> float:
        """Current monotonic time."""
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Sleep if `seconds` is positive."""
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """Manual clock: `sleep` advances time instantly; `advance` simulates slow work."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        """Current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record the request and jump forward."""
        self.sleeps.append(seconds)
        self.now += max(seconds, 0.0)

    def advance(self, seconds: float) -> None:
        """Move time forward without a sleep (e.g. a slow bus transaction)."""
        self.now += seconds
