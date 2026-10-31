"""The `JointBus` boundary to the physical arm, its LeRobot adapter, and a test fake."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from rbe_501_so101.config import HardwareConfig
from rbe_501_so101.hardware.joint_mapping import pos_key
from rbe_501_so101.hardware.types import MotorSettings, MotorStatus

if TYPE_CHECKING:
    from lerobot.robots.so_follower import SO101Follower

GRIPPER_PERCENT_LIMITS = (0.0, 100.0)
LOAD_FULL_SCALE = 1000.0
VOLTAGE_LSB_V = 0.1
SETTINGS_REGISTERS: dict[str, str] = {
    "p": "P_Coefficient",
    "i": "I_Coefficient",
    "d": "D_Coefficient",
    "acceleration": "Acceleration",
    "max_acceleration": "Maximum_Acceleration",
    "max_torque_limit": "Max_Torque_Limit",
    "protection_current": "Protection_Current",
    "overload_torque": "Overload_Torque",
}


def clip_relative(
    goal: Mapping[str, float], present: Mapping[str, float], max_relative_deg: float | None
) -> dict[str, float]:
    """Each goal limited to `max_relative_deg` from its present position (LeRobot's `max_relative_target`)."""
    if max_relative_deg is None:
        return {k: float(v) for k, v in goal.items()}
    return {k: float(min(max(v, present[k] - max_relative_deg), present[k] + max_relative_deg)) for k, v in goal.items()}


@runtime_checkable
class JointBus(Protocol):
    """Minimal physical-arm interface the rest of the code depends on.

    Positions use LeRobot units (`"<joint>.pos"` keys, degrees, gripper 0-100).
    """

    def read_positions(self) -> dict[str, float]:
        """Present joint positions."""
        ...

    def read_status(self) -> tuple[MotorStatus, ...]:
        """Temperature, voltage, and load of every motor."""
        ...

    def read_settings(self) -> tuple[MotorSettings, ...]:
        """Servo registers that affect sag."""
        ...

    def send_positions(self, goal: dict[str, float]) -> dict[str, float]:
        """Send a goal; return what was actually sent (possibly clipped)."""
        ...

    def set_torque(self, enabled: bool) -> None:
        """Enable or disable torque on every motor."""
        ...

    def limits_deg(self) -> dict[str, tuple[float, float]]:
        """Calibrated position range of every joint, in LeRobot units."""
        ...

    def close(self) -> None:
        """Disconnect from the arm."""
        ...


class LeRobotJointBus:
    """`JointBus` over LeRobot's `SO101Follower` (which keeps `max_relative_target` in force)."""

    def __init__(self, robot: "SO101Follower", degrees_per_count: float) -> None:
        self._robot = robot
        self._deg_per_count = degrees_per_count

    @classmethod
    def connect(cls, cfg: HardwareConfig = HardwareConfig(), calibrate: bool = False) -> Self:
        """Connect to the follower with the configured port, id, and relative-target cap."""
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        robot_cfg = SO101FollowerConfig(port=cfg.port, id=cfg.robot_id, max_relative_target=cfg.max_relative_target_deg)
        robot = SO101Follower(robot_cfg)
        robot.connect(calibrate=calibrate)
        return cls(robot, cfg.lerobot_degrees_per_count)

    @property
    def motors(self) -> list[str]:
        """Motor names in bus order."""
        return list(self._robot.bus.motors)

    def read_positions(self) -> dict[str, float]:
        """Present positions via `get_observation()`."""
        obs = self._robot.get_observation()
        return {pos_key(m): float(obs[pos_key(m)]) for m in self.motors}

    def read_status(self) -> tuple[MotorStatus, ...]:
        """Temperature (C), voltage (V), and |load| as a fraction of full scale."""
        temp = self._read_raw("Present_Temperature")
        volt = self._read_raw("Present_Voltage")
        load = self._read_raw("Present_Load")
        return tuple(
            MotorStatus(m, float(temp[m]), VOLTAGE_LSB_V * volt[m], abs(load[m]) / LOAD_FULL_SCALE) for m in self.motors
        )

    def read_settings(self) -> tuple[MotorSettings, ...]:
        """Every register LeRobot configures, read back from each motor."""
        raw = {field: self._read_raw(register) for field, register in SETTINGS_REGISTERS.items()}
        return tuple(MotorSettings(m, **{f: int(raw[f][m]) for f in raw}) for m in self.motors)

    def send_positions(self, goal: dict[str, float]) -> dict[str, float]:
        """`send_action`: LeRobot clips each joint relative to its present (sagged) position."""
        return {k: float(v) for k, v in self._robot.send_action(goal).items()}

    def set_torque(self, enabled: bool) -> None:
        """Enable or disable torque on every motor."""
        if enabled:
            self._robot.bus.enable_torque()
        else:
            self._robot.bus.disable_torque()

    def limits_deg(self) -> dict[str, tuple[float, float]]:
        """Arm ranges are symmetric about the calibrated midpoint; the gripper is 0-100."""
        limits = {}
        for motor in self.motors:
            cal = self._robot.calibration[motor]
            half = (cal.range_max - cal.range_min) / 2 * self._deg_per_count
            limits[motor] = (-half, half)
        limits[self.motors[-1]] = GRIPPER_PERCENT_LIMITS
        return limits

    def close(self) -> None:
        """Disconnect (LeRobot disables torque on disconnect by default)."""
        self._robot.disconnect()

    def _read_raw(self, register: str) -> dict[str, float]:
        """Unnormalized register values for every motor."""
        return self._robot.bus.sync_read(register, normalize=False)


class FakeJointBus:
    """In-memory arm for tests: each goal settles instantly to `goal - sag_deg`.

    With `max_relative_deg`, goals are first clipped to within that distance of
    the present position, as LeRobot's `max_relative_target` does.
    """

    def __init__(
        self,
        joints: tuple[str, ...],
        start_deg: Mapping[str, float] | None = None,
        sag_deg: Mapping[str, float] | None = None,
        limits: Mapping[str, tuple[float, float]] | None = None,
        status: tuple[MotorStatus, ...] | None = None,
        max_relative_deg: float | None = None,
    ) -> None:
        self.joints = joints
        self.positions = {pos_key(j): float((start_deg or {}).get(j, 0.0)) for j in joints}
        self.sag_deg = dict(sag_deg or {})
        self.max_relative_deg = max_relative_deg
        self._limits = dict(limits or {j: (-180.0, 180.0) for j in joints})
        self.status = status or tuple(MotorStatus(j, 30.0, 12.0, 0.1) for j in joints)
        self.sent: list[dict[str, float]] = []
        self.torque_log: list[bool] = []
        self.closed = False

    def read_positions(self) -> dict[str, float]:
        """Current positions (a copy)."""
        return dict(self.positions)

    def read_status(self) -> tuple[MotorStatus, ...]:
        """Configured status readings."""
        return self.status

    def read_settings(self) -> tuple[MotorSettings, ...]:
        """LeRobot's default gains on every motor."""
        defaults = dict(p=16, i=0, d=32, acceleration=254, max_acceleration=254)
        extra = dict(max_torque_limit=1000, protection_current=500, overload_torque=80)
        return tuple(MotorSettings(j, **defaults, **extra) for j in self.joints)

    def send_positions(self, goal: dict[str, float]) -> dict[str, float]:
        """Clip (if configured), record the goal, and settle at it minus the configured sag."""
        sent = clip_relative(goal, self.positions, self.max_relative_deg)
        self.sent.append(dict(sent))
        for key, value in sent.items():
            self.positions[key] = value - self.sag_deg.get(key.removesuffix(".pos"), 0.0)
        return sent

    def set_torque(self, enabled: bool) -> None:
        """Record torque changes."""
        self.torque_log.append(enabled)

    def limits_deg(self) -> dict[str, tuple[float, float]]:
        """Configured joint ranges."""
        return dict(self._limits)

    def close(self) -> None:
        """Mark the bus closed."""
        self.closed = True
