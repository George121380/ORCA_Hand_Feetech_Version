from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .model import JOINT_NAMES, MODEL_ID, ModelSpec, vector


@dataclass(frozen=True)
class JointTarget:
    positions: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)
    joint_names: tuple[str, ...] = JOINT_NAMES
    model_id: str = MODEL_ID

    def __post_init__(self):
        if self.joint_names != JOINT_NAMES or self.model_id != MODEL_ID:
            raise ValueError("Command model or named joint ordering mismatch")
        if not np.isfinite(self.timestamp):
            raise ValueError("Invalid command timestamp")
        q = vector(self.positions)
        q.flags.writeable = False
        object.__setattr__(self, "positions", q)


@dataclass
class HandState:
    positions: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)
    estimated: bool = False
    telemetry: dict = field(default_factory=dict)


class CommandGuard:
    """Bound joint targets and slew rate. Stale inputs stop target advancement."""
    def __init__(self, initial, lower=None, upper=None, max_velocity=0.8, max_acceleration=2.0):
        spec = ModelSpec.load()
        self.lower = spec.lower if lower is None else vector(lower)
        self.upper = spec.upper if upper is None else vector(upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("Empty joint range")
        self.q = vector(initial)
        if np.any(self.q < self.lower - 0.02) or np.any(self.q > self.upper + 0.02):
            raise ValueError("Initial state outside calibrated range; inspect before arming")
        self.q = np.clip(self.q, self.lower, self.upper)
        if max_velocity <= 0 or max_acceleration <= 0:
            raise ValueError("Motion limits must be positive")
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.velocity = np.zeros(17)
        self.last_source_time = -np.inf

    def apply(self, target: JointTarget | None, dt: float, now=None) -> tuple[np.ndarray, bool]:
        now = time.monotonic() if now is None else now
        if not 0 < dt <= 0.25:
            raise ValueError("Control timestep must be in (0, 0.25] seconds")
        fresh = target is not None and 0 <= now - target.timestamp <= 0.25
        if not fresh:
            self.velocity[:] = 0
            return self.q.copy(), False
        if target.timestamp < self.last_source_time:
            raise ValueError("Out-of-order command")
        self.last_source_time = target.timestamp
        destination = np.clip(target.positions, self.lower, self.upper)
        delta = destination - self.q
        desired_v = np.clip(delta / dt, -self.max_velocity, self.max_velocity)
        # Reduce speed as the remaining distance approaches the stopping distance.
        braking_v = np.sqrt(2 * self.max_acceleration * np.abs(delta))
        desired_v = np.sign(desired_v) * np.minimum(np.abs(desired_v), braking_v)
        self.velocity += np.clip(desired_v - self.velocity,
                                 -self.max_acceleration * dt, self.max_acceleration * dt)
        step = self.velocity * dt
        reaches = (np.sign(step) == np.sign(delta)) & (np.abs(step) >= np.abs(delta))
        step[reaches] = delta[reaches]
        self.velocity[reaches] = step[reaches] / dt
        self.q = np.clip(self.q + step, self.lower, self.upper)
        return self.q.copy(), True


def minimum_jerk(start, end, elapsed: float, duration: float) -> np.ndarray:
    if duration <= 0:
        raise ValueError("Duration must be positive")
    u = np.clip(elapsed / duration, 0, 1)
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    a, b = vector(start), vector(end)
    return a + blend * (b - a)


def move_duration(start, end, requested=2.5, velocity=0.8, acceleration=2.0) -> float:
    distance = float(np.max(np.abs(vector(end) - vector(start))))
    return max(requested, 1.875 * distance / velocity, np.sqrt(5.774 * distance / acceleration))
