"""The official BaseHand interface backed by the project's guarded backends."""
from __future__ import annotations

import time

import numpy as np
from orca_core import BaseHand, BaseHandConfig

from .backends import create_backend
from .control import CommandGuard, JointTarget, minimum_jerk, move_duration
from .model import JOINT_NAMES, ModelSpec, poses, to_orca


class OrcaHandController(BaseHand):
    """Accepts official OrcaJointPositions in degrees of the v1 MJCF coordinates.

    All inputs pass through the same guard as the CLI. Use move_to_pose for a
    complete smooth motion, or stream set_joint_positions from a control loop.
    """
    def __init__(self, backend="mock", *, hardware_config=None, render_mode=None):
        spec = ModelSpec.load()
        self.backend = create_backend(backend, hardware_config=hardware_config, render_mode=render_mode)
        try:
            config = BaseHandConfig(
                config_path="project-model-metadata", type="right", joint_ids=list(JOINT_NAMES),
                joint_roms_dict={n: np.rad2deg([self.backend.lower[i], self.backend.upper[i]]).tolist()
                                 for i, n in enumerate(JOINT_NAMES)},
                neutral_position=dict(zip(JOINT_NAMES, np.rad2deg(spec.neutral).tolist())))
            super().__init__(config=config)
            self.guard = CommandGuard(self.backend.read().positions, self.backend.lower, self.backend.upper)
            self.last_write = time.monotonic() - 0.05
        except BaseException:
            self.backend.close()
            raise

    def _get_joint_positions(self):
        return to_orca(self.backend.read().positions)

    def _set_joint_positions(self, joint_pos):
        q = self.guard.q.copy()
        for joint, degrees in joint_pos:
            if joint not in JOINT_NAMES or degrees is None or not np.isfinite(degrees):
                raise ValueError("Expected known joints with finite angles")
            q[JOINT_NAMES.index(joint)] = np.deg2rad(degrees)
        now = time.monotonic()
        target = JointTarget(q, timestamp=now)
        command, _ = self.guard.apply(target, min(0.25, max(1e-5, now - self.last_write)), now=now)
        self.backend.write(command, target.timestamp)
        self.last_write = now
        return True

    def move_to_pose(self, name, duration=2.5):
        end = poses()[name]
        start = self.backend.read().positions
        seconds = move_duration(start, end, duration)
        begin = time.monotonic()
        for step in range(1, int(np.ceil(seconds * 20)) + 21):
            time.sleep(max(0, begin + step / 20 - time.monotonic()))
            self.set_joint_positions(to_orca(minimum_jerk(start, end, step / 20, seconds)))
            self.backend.render()

    def close(self):
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
