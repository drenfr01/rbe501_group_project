"""Helpers for the curated `reports/` folder."""

import shutil
from pathlib import Path

from rbe_501_so101.config import REPORTS_DIR

FIGURES_SUBDIR = "figures"


def publish_figure(path: Path, stable_name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Copy an experiment plot into `reports/figures/` under a stable file name."""
    dest = reports_dir / FIGURES_SUBDIR / stable_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    return dest
