"""Repository for per-seed checkpoints: model, normalizer, run config, and monitor."""

import json
from pathlib import Path

from rbe_501_so101.config import PRESETS, ExperimentPreset, TrainConfig

LEGACY_PRESET = "legacy"


class CheckpointRepository:
    """Owns the `<model_dir>/seed_<n>/` layout.

    Each seed folder holds the SB3 model, the `VecNormalize` statistics, the
    training monitor CSV, and `run_config.json` naming the preset the env was
    built from. Folders without a run config predate presets and are treated
    as `legacy`.
    """

    def __init__(self, model_dir: Path, cfg: TrainConfig) -> None:
        self.model_dir = Path(model_dir)
        self._cfg = cfg

    def seed_dir(self, seed: int) -> Path:
        """Folder for one training seed."""
        return self.model_dir / f"seed_{seed}"

    def model_path(self, seed: int) -> Path:
        """Saved SB3 model."""
        return self.seed_dir(seed) / self._cfg.model_filename

    def normalizer_path(self, seed: int) -> Path:
        """Saved `VecNormalize` statistics."""
        return self.seed_dir(seed) / self._cfg.vecnormalize_filename

    def monitor_stem(self, seed: int) -> Path:
        """Monitor path without the `.monitor.csv` suffix SB3 appends."""
        return self.seed_dir(seed) / self._cfg.monitor_stem

    def ensure_untrained(self, seed: int) -> None:
        """Raise rather than overwrite an existing checkpoint."""
        if self.model_path(seed).exists():
            raise FileExistsError(f"{self.model_path(seed)} exists; choose a new --model-dir")

    def save_preset(self, seed: int, name: str, preset: ExperimentPreset) -> None:
        """Record which preset this seed was trained with."""
        self.seed_dir(seed).mkdir(parents=True, exist_ok=True)
        payload = {"preset": name, **preset.to_dict()}
        self._run_config_path(seed).write_text(json.dumps(payload, indent=2))

    def load_preset(self, seed: int) -> ExperimentPreset:
        """Preset recorded for this seed, or `legacy` if none was recorded."""
        path = self._run_config_path(seed)
        if not path.exists():
            return PRESETS[LEGACY_PRESET]
        return ExperimentPreset.from_dict(json.loads(path.read_text()))

    def load_preset_name(self, seed: int) -> str:
        """Name of the preset recorded for this seed."""
        path = self._run_config_path(seed)
        if not path.exists():
            return LEGACY_PRESET
        return str(json.loads(path.read_text())["preset"])

    def _run_config_path(self, seed: int) -> Path:
        """Location of the seed's run config."""
        return self.seed_dir(seed) / self._cfg.run_config_filename
