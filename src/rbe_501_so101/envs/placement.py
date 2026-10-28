"""Where the object starts: a band of radii and azimuths around the shoulder-pan axis."""

import numpy as np

from rbe_501_so101.config import ObjectPlacementConfig


class TablePlacer:
    """Sample reachable cube poses around the pan axis."""

    def __init__(self, cfg: ObjectPlacementConfig, pan_axis_xy: np.ndarray, rest_z: float) -> None:
        self._cfg = cfg
        self._axis_xy = np.asarray(pan_axis_xy, dtype=float)
        self._rest_z = rest_z

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """Return the object center p_s (m) and yaw (rad)."""
        radius = rng.uniform(self._cfg.radial_min_m, self._cfg.radial_max_m)
        phi_max = np.deg2rad(self._cfg.azimuth_max_deg)
        phi = float(rng.uniform(-phi_max, phi_max))
        xy = self._axis_xy + radius * np.array([np.cos(phi), np.sin(phi)])
        return np.array([xy[0], xy[1], self._rest_z]), phi
