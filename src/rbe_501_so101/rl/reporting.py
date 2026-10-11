"""Load training logs and evaluation reports, and plot them for an experiment record."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

MONITOR_HEADER_LINES = 1
SMOOTHING_EPISODES = 50


@dataclass(frozen=True)
class TrainingCurve:
    """Per-episode training reward for one seed, indexed by cumulative env steps."""

    seed: str
    timesteps: np.ndarray
    rewards: np.ndarray

    @classmethod
    def from_monitor(cls, path: Path, seed: str) -> Self:
        """Parse an SB3 `Monitor` CSV (columns r, l, t after a JSON header line)."""
        frame = pd.read_csv(path, skiprows=MONITOR_HEADER_LINES)
        return cls(seed=seed, timesteps=frame["l"].cumsum().to_numpy(), rewards=frame["r"].to_numpy())

    def smoothed(self, window: int = SMOOTHING_EPISODES) -> np.ndarray:
        """Trailing rolling mean of episode reward."""
        return pd.Series(self.rewards).rolling(window, min_periods=1).mean().to_numpy()

    def tail_mean(self, fraction: float) -> float:
        """Mean reward over the last `fraction` of episodes."""
        start = int(len(self.rewards) * (1.0 - fraction))
        return float(np.mean(self.rewards[start:]))


def load_curves(model_dir: Path, seeds: tuple[int, ...], monitor_name: str) -> list[TrainingCurve]:
    """Load one training curve per seed from `<model_dir>/seed_<n>/<monitor_name>`."""
    return [TrainingCurve.from_monitor(model_dir / f"seed_{s}" / monitor_name, str(s)) for s in seeds]


def load_report(path: Path) -> dict:
    """Read an evaluation JSON report, skipping any log lines before the JSON body."""
    text = path.read_text()
    return json.loads(text[text.index("{") :])
