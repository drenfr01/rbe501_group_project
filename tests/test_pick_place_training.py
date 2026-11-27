"""New full-task runs must not overwrite old pilots or accept configuration drift."""

from dataclasses import replace

import pytest

from rbe_501_so101.config import TrainConfig
from rbe_501_so101.envs.pick_place_config import PickPlaceConfig
from rbe_501_so101.rl.pick_place_training import PickPlaceRepository


def test_full_task_preset_roundtrip_and_no_overwrite(tmp_path) -> None:
    """Persist the exact environment/controller/reward definition beside fresh weights."""
    repo = PickPlaceRepository(tmp_path / "new_full_task")
    cfg = PickPlaceConfig()
    repo.create(cfg, replace(TrainConfig(), timesteps=10000), seed=0)
    assert repo.load() == cfg
    with pytest.raises(FileExistsError):
        repo.create(cfg, TrainConfig(), seed=0)
    original = repo.path.read_text()
    repo.path.write_text(original.replace('"residual_scale_m": 0.001', '"residual_scale_m": 0.01', 1))
    with pytest.raises(ValueError):
        repo.load()
