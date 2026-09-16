"""Object-centred pen reorientation on the pinned, palm-up ORCA v1 hand."""
from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .model import JOINT_NAMES, ModelSpec
from .paths import upstream, workspace

TASK_ID = "orca-v1-pen-reorientation-v1"
PEN_SPAWN = np.array([0.135, 0.049, 0.182])


def build_pen_scene() -> Path:
    """Reuse the official cube task's palm-up hand; change only the task object."""
    package = upstream("orca_sim") / "src/orca_sim"
    source = package / "models/v1/right_cube_orientation.mjcf"
    root = ET.parse(source).getroot()
    root.set("model", TASK_ID)
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((source.parent / mesh.get("file")).resolve()))
    ET.SubElement(root, "include", file=str(package / "scenes/v1/scene.xml"))
    world = root.find("worldbody")
    pen = ET.SubElement(world, "body", name="task_pen", pos=" ".join(map(str, PEN_SPAWN)))
    ET.SubElement(pen, "freejoint", name="pen_freejoint")
    ET.SubElement(pen, "geom", name="pen_collision", type="capsule", fromto="0 -0.065 0 0 0.065 0",
                  size="0.007", mass="0.018", rgba="0.96 0.64 0.16 1", friction="1.2 0.01 0.002")
    ET.SubElement(pen, "geom", name="pen_tip", type="capsule", fromto="0 0.055 0 0 0.065 0", size="0.0074",
                  mass="0", rgba="0.85 0.12 0.12 1", contype="0", conaffinity="0")
    goal = ET.SubElement(world, "body", name="pen_goal", mocap="true", pos="0.135 0.049 0.22")
    ET.SubElement(goal, "geom", type="capsule", fromto="0 -0.065 0 0 0.065 0", size="0.003",
                  rgba="0.25 0.9 0.75 0.28", contype="0", conaffinity="0")
    ET.SubElement(goal, "geom", type="sphere", pos="0 0.068 0", size="0.004",
                  rgba="0.25 0.9 0.75 0.5", contype="0", conaffinity="0")
    output = workspace() / "local/tasks/v1/pen.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = ET.tostring(root, encoding="unicode")
    if not output.exists() or output.read_text() != payload:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as tmp:
            tmp.write(payload)
        os.replace(tmp.name, output)
    return output


class PenReorientationEnv(gym.Env):
    """Rotate a free pen by a requested angle, with contact and retention checks.

    The policy observes simulator object state (a teacher policy), not camera RGB.
    No forces, welds, or direct pose changes are applied to the pen after reset.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, turn_degrees=90, horizon=200):
        from orca_sim.envs import BaseOrcaHandEnv
        if not 10 <= turn_degrees <= 180:
            raise ValueError("Turn angle must be 10..180 degrees")
        self.base = BaseOrcaHandEnv(str(build_pen_scene()), version="v1", frame_skip=25)
        self.model, self.data = self.base.model, self.base.data
        self.hand_spec = ModelSpec.load()
        self.dt = self.model.opt.timestep * self.base.frame_skip
        if not np.isclose(self.dt, 0.05):
            raise ValueError("Expected the pinned model's 2 ms physics timestep")
        self.qadr = np.array([self.model.joint("right_" + n).qposadr[0] for n in JOINT_NAMES])
        self.vadr = np.array([self.model.joint("right_" + n).dofadr[0] for n in JOINT_NAMES])
        self.actuators = np.array([self.model.actuator("right_" + n + "_actuator").id for n in JOINT_NAMES])
        self.pen_id = self.model.body("task_pen").id
        self.pen_qadr = self.model.joint("pen_freejoint").qposadr[0]
        self.pen_vadr = self.model.joint("pen_freejoint").dofadr[0]
        self.pen_geom = self.model.geom("pen_collision").id
        self.goal_mocap = self.model.body("pen_goal").mocapid[0]
        self.render_mode, self.turn_degrees, self.horizon = render_mode, float(turn_degrees), horizon
        self.renderer = None
        self.action_space = gym.spaces.Box(-1, 1, (16,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (69,), dtype=np.float32)
        self.goal_axis = np.array([1., 0., 0.])
        self.previous = self.hand_spec.neutral.copy()
        self.steps = self.stable = 0
        self.succeeded = False
        self.initial_position = PEN_SPAWN.copy()
        self.previous_angle = np.pi / 2

    def _axis(self):
        return self.data.xmat[self.pen_id].reshape(3, 3)[:, 1].copy()

    def _contacts(self):
        flags = np.zeros(6)
        for contact in self.data.contact:
            if self.pen_geom not in (contact.geom1, contact.geom2) or contact.dist > 0:
                continue
            other = contact.geom2 if contact.geom1 == self.pen_geom else contact.geom1
            name = self.model.body(self.model.geom_bodyid[other]).name or ""
            for i, finger in enumerate(("thumb", "index", "middle", "ring", "pinky", "palm")):
                if name.startswith("right_" + finger):
                    flags[i] = 1
        return flags

    def _normalize(self, q):
        return 2 * (q - self.hand_spec.lower[:16]) / (self.hand_spec.upper[:16] - self.hand_spec.lower[:16]) - 1

    def _observation(self):
        return np.concatenate([
            self._normalize(self.data.qpos[self.qadr[:16]]),
            np.clip(self.data.qvel[self.vadr[:16]] / 5, -5, 5),
            self._normalize(self.previous[:16]),
            (self.data.xpos[self.pen_id] - self.initial_position) / .1,
            self._axis(), np.clip(self.data.qvel[self.pen_vadr:self.pen_vadr + 6], -10, 10),
            self.goal_axis, self._contacts(),
        ]).astype(np.float32)

    def _info(self):
        pos = self.data.xpos[self.pen_id]
        velocity = self.data.qvel[self.pen_vadr:self.pen_vadr + 6]
        angle = float(np.arccos(np.clip(np.dot(self._axis(), self.goal_axis), -1, 1)))
        displacement = float(np.linalg.norm(pos - self.initial_position))
        dropped = bool(pos[2] < .08 or displacement > .15)
        contact = bool(self._contacts().any())
        stable = (angle < np.deg2rad(10) and displacement < .04 and contact
                  and np.linalg.norm(velocity[:3]) < .10 and np.linalg.norm(velocity[3:]) < 1.0)
        return {"angle_error_deg": float(np.rad2deg(angle)), "angle_error_rad": angle,
                "displacement_m": displacement, "pen_position": pos.copy(), "pen_axis": self._axis(),
                "hand_contact": contact, "dropped": dropped, "at_goal_stable": bool(stable),
                "is_success": self.succeeded, "elapsed_s": self.steps * self.dt}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Small pose jitter can occasionally roll the pen off the palm during
        # settling. Reject only those initializations, not failures during play.
        for attempt in range(20):
            obs, info = self._reset_once(options)
            if not info["dropped"] and info["hand_contact"]:
                info["reset_attempts"] = attempt + 1
                return obs, info
        raise RuntimeError("Could not sample a supported pen reset in 20 attempts")

    def _reset_once(self, options):
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        self.previous = self.hand_spec.neutral.copy()
        self.data.qpos[self.qadr] = self.previous
        self.data.ctrl[self.actuators] = self.previous
        yaw = np.pi / 2 + self.np_random.uniform(-.08, .08)
        xy = self.np_random.uniform(-.002, .002, 2)
        self.data.qpos[self.pen_qadr:self.pen_qadr + 3] = PEN_SPAWN + [*xy, 0]
        self.data.qpos[self.pen_qadr + 3:self.pen_qadr + 7] = [np.cos((yaw - np.pi / 2) / 2), 0, 0, np.sin((yaw - np.pi / 2) / 2)]
        mujoco.mj_forward(self.model, self.data)
        # Let the free object settle under gravity before defining the target.
        mujoco.mj_step(self.model, self.data, nstep=150)
        mujoco.mj_forward(self.model, self.data)
        self.initial_position = self.data.xpos[self.pen_id].copy()
        initial_axis = self._axis()
        direction = options.get("direction", int(self.np_random.choice([-1, 1])))
        if direction not in (-1, 1):
            raise ValueError("Direction must be -1 or 1")
        goal_yaw = np.arctan2(initial_axis[1], initial_axis[0]) + direction * np.deg2rad(self.turn_degrees)
        self.goal_axis = np.array([np.cos(goal_yaw), np.sin(goal_yaw), 0.])
        self.data.mocap_pos[self.goal_mocap] = self.initial_position + [0, 0, .035]
        self.data.mocap_quat[self.goal_mocap] = [np.cos((goal_yaw - np.pi / 2) / 2), 0, 0, np.sin((goal_yaw - np.pi / 2) / 2)]
        self.data.time = 0
        self.steps = self.stable = 0
        self.succeeded = False
        mujoco.mj_forward(self.model, self.data)
        info = self._info()
        self.previous_angle = info["angle_error_rad"]
        return self._observation(), info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("Expected 16 finite finger actions")
        action = np.clip(action, -1, 1)
        self.previous[:16] = np.clip(self.previous[:16] + .8 * self.dt * action,
                                    self.hand_spec.lower[:16], self.hand_spec.upper[:16])
        self.data.ctrl[self.actuators] = self.previous
        mujoco.mj_step(self.model, self.data, nstep=self.base.frame_skip)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        info = self._info()
        self.stable = self.stable + 1 if info["at_goal_stable"] and not info["dropped"] else 0
        newly_succeeded = self.stable >= 10 and not self.succeeded
        self.succeeded |= newly_succeeded
        angle = info["angle_error_rad"]
        reward = (8 * (self.previous_angle - angle) + .3 * (np.cos(angle) + 1)
                  + .1 * info["hand_contact"] - 4 * info["displacement_m"]
                  - .01 * np.mean(action**2) + 10 * newly_succeeded - 5 * info["dropped"])
        self.previous_angle = angle
        info["is_success"] = self.succeeded
        return self._observation(), float(reward), info["dropped"] or self.succeeded, self.steps >= self.horizon, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, camera)
        camera.lookat[:] = [.15, .04, .17]
        camera.distance, camera.azimuth, camera.elevation = .40, 145, -55
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()
        self.base.close()
