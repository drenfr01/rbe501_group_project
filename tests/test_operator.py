"""Operator keyboard helpers: scripted keys, software abort, and in-place redraw."""

import numpy as np
import pytest

from rbe_501_so101.hardware.operator import OperatorAbortPolicy, ScriptedKeys, redraw_text
from rbe_501_so101.hardware.safety import SafetyAbort


class _ZeroPolicy:
    """Policy that records resets and returns zeros."""

    def __init__(self) -> None:
        self.resets = 0

    def reset(self, target, state) -> None:
        """Count resets."""
        self.resets += 1

    def __call__(self, obs):
        """Zero action."""
        return np.zeros(3)


def test_scripted_keys_yield_in_order_then_none():
    keys = ScriptedKeys(["a", None, "b"])
    assert [keys.poll(0.0) for _ in range(4)] == ["a", None, "b", None]


def test_abort_policy_passes_through_until_abort_key():
    inner = _ZeroPolicy()
    policy = OperatorAbortPolicy(inner, ScriptedKeys([None, "x", "q"]), abort_key="q")
    policy.reset(None, None)
    assert inner.resets == 1
    np.testing.assert_array_equal(policy(np.zeros(1)), np.zeros(3))
    policy(np.zeros(1))
    with pytest.raises(SafetyAbort, match="operator"):
        policy(np.zeros(1))


def test_redraw_moves_cursor_up_after_first_draw():
    assert redraw_text(["a", "b"], first=True) == "a\x1b[K\nb\x1b[K\n"
    assert redraw_text(["a", "b"], first=False).startswith("\x1b[2F")
