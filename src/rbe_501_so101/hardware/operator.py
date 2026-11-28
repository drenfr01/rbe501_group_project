"""Operator input for hardware scripts: non-blocking keys, a software abort, and in-place tables."""

import select
import sys
import termios
import tty
from collections import deque
from collections.abc import Iterable
from types import TracebackType
from typing import Protocol, Self

import numpy as np

from rbe_501_so101.envs.types import RobotState
from rbe_501_so101.hardware.safety import SafetyAbort
from rbe_501_so101.rl.policies import EvalPolicy

ABORT_KEY = "q"


class KeySource(Protocol):
    """Anything that can report a pressed key without blocking for long."""

    def poll(self, timeout_s: float) -> str | None:
        """Return one pressed key, or None if none arrives within `timeout_s`."""
        ...


class TerminalKeys:
    """Single-key reads from stdin in cbreak mode; use as a context manager."""

    def __enter__(self) -> Self:
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def poll(self, timeout_s: float) -> str | None:
        """Return one pressed key, or None after `timeout_s`."""
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        return sys.stdin.read(1) if ready else None

    def prompt(self, text: str) -> str:
        """Read a full line with normal echo, then return to cbreak mode."""
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        try:
            return input(text)
        finally:
            tty.setcbreak(self._fd)


class ScriptedKeys:
    """Replays a fixed key sequence (None means no key); for tests and `--sim` runs."""

    def __init__(self, keys: Iterable[str | None] = ()) -> None:
        self._keys = deque(keys)

    def poll(self, timeout_s: float) -> str | None:
        """Next scripted key, or None once exhausted."""
        return self._keys.popleft() if self._keys else None


class OperatorAbortPolicy:
    """Wraps a policy so a keypress between steps raises `SafetyAbort`."""

    def __init__(self, inner: EvalPolicy, keys: KeySource, abort_key: str = ABORT_KEY) -> None:
        self._inner = inner
        self._keys = keys
        self._abort_key = abort_key

    def reset(self, target: np.ndarray, state: RobotState) -> None:
        """Reset the wrapped policy."""
        self._inner.reset(target, state)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Wrapped action, unless the operator pressed the abort key."""
        if self._keys.poll(0.0) == self._abort_key:
            raise SafetyAbort("operator abort key pressed")
        return self._inner(obs)


def redraw_text(lines: list[str], first: bool) -> str:
    """Escape sequence that overwrites the previously drawn `lines` in place."""
    prefix = "" if first else f"\x1b[{len(lines)}F"
    return prefix + "".join(f"{line}\x1b[K\n" for line in lines)


def redraw(lines: list[str], first: bool) -> None:
    """Write `lines` over the previous table (clearing the screen floods some scrollbacks)."""
    sys.stdout.write(redraw_text(lines, first))
    sys.stdout.flush()
