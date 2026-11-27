"""Versioned full-task configuration, observation layout and placement reward semantics."""

from dataclasses import dataclass, field

import numpy as np

from rbe_501_so101.control.carry_place import TransferConfig
from rbe_501_so101.control.supported_place import SupportSearchConfig
from rbe_501_so101.control.timed_oracle_acquisition import TimingConfig
from rbe_501_so101.control.tracked_place import PlacementTrackingConfig
from rbe_501_so101.rl.cartesian_zero_training import CartesianZeroPreset


@dataclass(frozen=True)
class PlacementRewardConfig:
    """Potential shaping plus full-placement reward; no success bonus for lifting alone."""

    gamma: float = .98
    distance_scale: float = 10.
    step_cost: float = .001
    residual_cost: float = .001
    success_bonus: float = 10.
    failure_penalty: float = 10.


@dataclass(frozen=True)
class PickPlaceConfig:
    """Separate residual task; prior reach/pick-lift configurations are not modified."""

    preset: CartesianZeroPreset = field(default_factory=CartesianZeroPreset)
    transfer: TransferConfig = TransferConfig()
    support: SupportSearchConfig = SupportSearchConfig()
    tracking: PlacementTrackingConfig = PlacementTrackingConfig()
    timing: TimingConfig = TimingConfig()
    reward: PlacementRewardConfig = PlacementRewardConfig()
    max_episode_steps: int = 720
    residual_scale_m: float = .001
    yaw_offsets_deg: tuple[float, ...] = (-30., 0., 30.)
    angular_speed_tolerance_rad_s: float = .1

    def __post_init__(self) -> None:
        """Require a bounded action authority and a valid finite episode."""
        if self.max_episode_steps < 1 or not 0 < self.residual_scale_m <= .001:
            raise ValueError("Positive horizon and residual scale <=1mm required")
        if not self.yaw_offsets_deg or not all(np.isfinite(self.yaw_offsets_deg)):
            raise ValueError("At least one finite yaw offset required")


@dataclass(frozen=True)
class PlacementStatus:
    """Independent physical completion evidence, not a proxy based on target proximity."""

    lifted: bool
    released: bool
    retreated: bool
    settled: bool
    supported: bool
    grasped: bool
    xy_error_m: float
    height_error_m: float
    speed_m_s: float
    angular_speed_rad_s: float = 0.


def placement_success(status: PlacementStatus, cfg: PickPlaceConfig) -> bool:
    """Require the completed sequence and a released, supported cube at rest at the destination."""
    ready = status.lifted and status.released and status.retreated and status.settled
    return bool(ready and status.supported and not status.grasped
                and status.xy_error_m <= cfg.transfer.final_tolerance_m
                and abs(status.height_error_m) <= cfg.transfer.support_height_m
                and status.speed_m_s <= cfg.transfer.speed_tolerance_m_s
                and status.angular_speed_rad_s <= cfg.angular_speed_tolerance_rad_s)


OBSERVATION_FIELDS = (
    ("q", 6), ("qvel", 6), ("command_error", 6), ("pinch", 3), ("cube", 3),
    ("cube_quaternion", 4), ("cube_velocity", 6), ("destination", 3), ("destination_error", 3),
    ("previous_action", 3), ("phase", 6), ("progress", 3), ("contacts", 2), ("mass", 1),
    ("reference_error", 3), ("tracking_bias", 3), ("jaw_axis", 3), ("carry_height", 1),
    ("retreat_error", 3), ("reference_valid", 1), ("milestones", 3),
)


@dataclass(frozen=True)
class PickPlaceObservationLayout:
    """Named slices of a fixed flat vector suitable for SAC/PPO MLP policies."""

    @property
    def size(self) -> int:
        """Total scalar observation count."""
        return sum(count for _, count in OBSERVATION_FIELDS)

    def get(self, name: str) -> slice:
        """Return a stable slice without duplicating offsets in policies/tests."""
        start = 0
        for field_name, count in OBSERVATION_FIELDS:
            if field_name == name:
                return slice(start, start + count)
            start += count
        raise KeyError(name)


@dataclass(frozen=True)
class PlacementRewardTerms:
    """Auditable dense progress, costs and sparse task-completion outcome."""

    progress: float
    time: float
    residual: float
    completion: float
    failure: float

    @property
    def total(self) -> float:
        """Scalar reward passed to Gymnasium/SB3."""
        return self.progress + self.time + self.residual + self.completion + self.failure


def reward_terms(before: float, after: float, action: np.ndarray, success: bool, failed: bool,
                 cfg: PlacementRewardConfig) -> PlacementRewardTerms:
    """Use absorbing terminal potential only for termination, not time-limit truncation."""
    terminal = success or failed
    shaping = cfg.gamma * (0. if terminal else after) - before
    return PlacementRewardTerms(shaping, -cfg.step_cost, -cfg.residual_cost * float(action @ action),
                                cfg.success_bonus if success else 0., -cfg.failure_penalty if failed else 0.)
