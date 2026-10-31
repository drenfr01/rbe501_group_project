"""Typed motor readings shared by the bus adapters, sag check, and safety monitor."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorStatus:
    """Readings that change how strong a motor is: temperature, voltage, load."""

    name: str
    temperature_c: float
    voltage_v: float
    load_fraction: float


@dataclass(frozen=True)
class MotorSettings:
    """Servo registers that affect sag, read back from the motor."""

    name: str
    p: int
    i: int
    d: int
    acceleration: int
    max_acceleration: int
    max_torque_limit: int
    protection_current: int
    overload_torque: int
