"""Deployable-observation planar object reorientation on the pinned ORCA v1."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

from .model import JOINT_NAMES, ModelSpec
from .objects import specification, geometry_signature
from .paths import upstream

TASK_ID = "orca-v1-printable-yaw-v7"
DT = .05
HISTORY = 4
FRAME_SIZE = 74
RESET_LINEAR_SPEED = .015
RESET_ANGULAR_SPEED = .3
RESET_STABLE_SAMPLES = 4
RESET_MAX_SAMPLES = 100


@dataclass
class TaskConfig:
    object: str = "cube"
    angle: float = 15
    turns: int = 1
    horizon: int = 240
    randomization: float = 1.
    mass_kg: float | None = None
    friction: float = 1.
    max_velocity: float = .8
    max_acceleration: float = 2.
    delay_max_steps: int = 2
    command_bias_deg: float = 2.
    encoder_bias_deg: float = 1.5
    lower: list | None = None
    upper: list | None = None

    def __post_init__(self):
        if self.object not in ("cube", "pen") or not 10 <= self.angle <= 180:
            raise ValueError("Use cube/pen and angle 10..180")
        if not 1 <= self.turns <= 8 or self.horizon < 20 or not 0 <= self.randomization <= 2:
            raise ValueError("Invalid turns, horizon or randomization")
        if not .1 <= self.friction <= 3 or self.mass_kg is not None and not .005 <= self.mass_kg <= .2:
            raise ValueError("Invalid measured object mass/friction")
        if not 0 < self.max_velocity <= .8 or not 0 < self.max_acceleration <= 2:
            raise ValueError("Action limits exceed the deployment envelope")
        if (not isinstance(self.delay_max_steps, int) or not 0 <= self.delay_max_steps <= 4
                or not 0 <= self.command_bias_deg <= 15 or not 0 <= self.encoder_bias_deg <= 15):
            raise ValueError("Invalid measured actuator randomization envelope")
        limits(self)

    def fingerprint(self, *, task_id=TASK_ID):
        payload = {"task": task_id, "config": asdict(self), "geometry": geometry_signature(self.object),
                   "joints": JOINT_NAMES, "model_sha": ModelSpec.load().source_sha256}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def limits(config):
    spec = ModelSpec.load()
    lo = spec.lower.copy() if config.lower is None else np.asarray(config.lower, dtype=float)
    hi = spec.upper.copy() if config.upper is None else np.asarray(config.upper, dtype=float)
    if (lo.shape != (17,) or hi.shape != (17,) or not np.isfinite([lo, hi]).all()
            or np.any(lo < spec.lower) or np.any(hi > spec.upper) or np.any(lo >= hi)
            or np.any(spec.neutral < lo) or np.any(spec.neutral > hi)):
        raise ValueError("Task ROM must be a finite subset of model ROM containing neutral")
    return lo.copy(), hi.copy()


def yaw_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def rotation_error(rotation, goal):
    return float(np.arccos(np.clip((np.trace(goal.T @ rotation) - 1) / 2, -1, 1)))


class ActionCodec:
    """Same incremental position, velocity and acceleration limits on both sides."""
    def __init__(self, config, initial):
        self.config, self.q = config, np.asarray(initial, dtype=float).copy()
        self.lower, self.upper = limits(config)
        self.wrist = ModelSpec.load().neutral[-1]
        self.velocity = np.zeros(16)

    def step(self, action, valid=True):
        action = np.asarray(action, dtype=float)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("Expected 16 finite actions")
        if not valid:
            self.velocity[:] = 0
            return self.q.copy()
        desired = self.config.max_velocity * np.clip(action, -1, 1)
        self.velocity += np.clip(desired - self.velocity,
                                 -self.config.max_acceleration * DT, self.config.max_acceleration * DT)
        self.q[:16] = np.clip(self.q[:16] + self.velocity * DT, self.lower[:16], self.upper[:16])
        # Wrist stays at model neutral; the real fixture must match.
        self.q[-1] = self.wrist
        return self.q.copy()


class ObservationCodec:
    def __init__(self, config):
        self.lower, self.upper = limits(config)
        self.max_velocity = config.max_velocity
        self.frames = deque(maxlen=HISTORY)

    def push(self, q, command, position, rotation, goal, origin, valid, age, *, controller_velocity):
        def normalized(x):
            return 2 * (np.asarray(x)[:16] - self.lower[:16]) / (self.upper[:16] - self.lower[:16]) - 1
        velocity = np.asarray(controller_velocity, dtype=float)
        if velocity.shape != (16,) or not np.isfinite(velocity).all():
            raise ValueError("Expected 16 finite controller velocities")
        frame = np.concatenate([normalized(q), normalized(command),
                                # Absolute S-frame position locates the object relative to
                                # the fixed hand; displacement alone hides initial contact
                                # location. Both are observable through calibrated vision.
                                np.asarray(position) / .2,
                                (np.asarray(position) - origin) / .1,
                                np.asarray(rotation).reshape(9), np.asarray(goal).reshape(9),
                                [float(valid), min(age, 1.)],
                                # Available in the shared controller, not measured or
                                # simulator-truth joint velocity. At command limits its
                                # acceleration-filter state cannot be inferred from q.
                                velocity / self.max_velocity]).astype(np.float32)
        if frame.shape != (FRAME_SIZE,) or not np.isfinite(frame).all():
            raise ValueError("Non-finite or malformed observation")
        if not self.frames:
            self.frames.extend([frame.copy() for _ in range(HISTORY)])
        else:
            self.frames.append(frame)
        return np.concatenate(self.frames)


def build_model(config, *, object_spec=None, task_id=TASK_ID):
    """Build the canonical task; raw design reviews can supply a candidate solid.

    ManipulationEnv always uses the canonical geometry, so checkpoint contracts
    cannot silently train a review candidate.
    """
    source = upstream("orca_sim") / "src/orca_sim/models/v1/right_cube_orientation.mjcf"
    root = ET.parse(source).getroot()
    root.set("model", task_id)
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((source.parent / mesh.get("file")).resolve()))
    # Keep original 17 joints, lock the wrist with an equality constraint.
    equality = ET.SubElement(root, "equality")
    neutral = ModelSpec.load().neutral[-1]
    ET.SubElement(equality, "joint", joint1="right_wrist", polycoef=f"{neutral} 0 0 0 0")
    ET.SubElement(root, "option", timestep=".002", integrator="implicitfast")
    spec = object_spec if object_spec is not None else specification(config.object)
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 0 1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .01", rgba=".2 .25 .2 1")
    body = ET.SubElement(world, "body", name="task_object", pos=".135 .049 .205")
    ET.SubElement(body, "freejoint", name="object_free")
    total_volume = sum(p["volume_m3"] for p in spec["collision_parts"])
    for i, part in enumerate(spec["collision_parts"]):
        name = f"printable_mesh_{i}"
        ET.SubElement(root.find("asset"), "mesh", name=name, scale="1 1 1",
                      vertex=" ".join(map(str, np.asarray(part["vertices_m"]).ravel())),
                      face=" ".join(map(str, np.asarray(part["triangles"]).ravel())))
        ET.SubElement(body, "geom", name=f"object_collision_{i}", type="mesh", mesh=name,
                      mass=str((config.mass_kg or spec["nominal_mass_kg"])*part["volume_m3"]/total_volume),
                      rgba=".85 .60 .25 1", friction=f"{config.friction} .01 .002", condim="4")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class ManipulationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    model_builder = staticmethod(build_model)
    action_codec_class = ActionCodec

    def __init__(self, config=None, render_mode=None, negative_goal_probability=.5,
                 minimum_goal_angle=None):
        if not 0 <= negative_goal_probability <= 1:
            raise ValueError("Goal sampling probability must be in [0,1]")
        # Training-only goal distribution. The canonical evaluation and real
        # goal interface retain both directions and the same task contract.
        self.negative_goal_probability = negative_goal_probability
        self.config = config or TaskConfig()
        if minimum_goal_angle is not None and not 10 <= minimum_goal_angle <= self.config.angle:
            raise ValueError("Training minimum goal angle must be between 10 and task angle")
        self.minimum_goal_angle = minimum_goal_angle
        self.model = self.model_builder(self.config)
        self.data = mujoco.MjData(self.model)
        self.hand_spec = ModelSpec.load()
        self.lower, self.upper = limits(self.config)
        self.qadr = np.array([self.model.joint("right_" + n).qposadr[0] for n in JOINT_NAMES])
        self.actuators = np.array([self.model.actuator("right_" + n + "_actuator").id for n in JOINT_NAMES])
        self.body = self.model.body("task_object").id
        self.geoms = [self.model.geom(f"object_collision_{i}").id
                      for i in range(len(specification(self.config.object)["collision_parts"]))]
        self.oq = self.model.joint("object_free").qposadr[0]
        self.ov = self.model.joint("object_free").dofadr[0]
        self.nominal = {k: getattr(self.model, k).copy() for k in
                        ("body_mass", "body_inertia", "geom_friction", "actuator_gainprm",
                         "actuator_biasprm", "dof_damping")}
        self.action_space = gym.spaces.Box(-1, 1, (16,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (FRAME_SIZE * HISTORY,), np.float32)
        self.renderer = None
        self.render_mode = render_mode

    def pose(self):
        return self.data.xpos[self.body].copy(), self.data.xmat[self.body].reshape(3, 3).copy()

    def contact(self):
        supported, on_tower = False, False
        for c in self.data.contact:
            if any(g in (c.geom1, c.geom2) for g in self.geoms) and c.dist <= 0:
                other = c.geom2 if c.geom1 in self.geoms else c.geom1
                name = self.model.body(self.model.geom_bodyid[other]).name or ""
                supported |= name.startswith("right_") and name != "right_tower"
                on_tower |= name == "right_tower"
        return supported and not on_tower

    def _randomize(self):
        for k, value in self.nominal.items():
            getattr(self.model, k)[:] = value
        rng, strength = self.np_random, self.config.randomization
        mass_scale = np.exp(rng.uniform(-.35, .35) * strength)
        self.model.body_mass[self.body] *= mass_scale
        self.model.body_inertia[self.body] *= mass_scale
        self.model.geom_friction[self.geoms, 0] *= np.exp(rng.uniform(-.45, .45) * strength)
        gain = np.exp(rng.uniform(-.4, .4, len(self.actuators)) * strength)
        self.model.actuator_gainprm[self.actuators, 0] *= gain
        self.model.actuator_biasprm[self.actuators, 1] *= gain
        self.model.dof_damping[:] *= np.exp(rng.uniform(-.3, .3) * strength)
        mujoco.mj_setConst(self.model, self.data)
        self.delay = int(rng.integers(0, self.config.delay_max_steps + 1)) if strength else 0
        bias = np.deg2rad(self.config.command_bias_deg)
        self.bias = rng.uniform(-bias, bias, 17) * strength
        self.bias[-1] = 0
        bias = np.deg2rad(self.config.encoder_bias_deg)
        self.encoder_bias = rng.uniform(-bias, bias, 17) * strength
        self.dropout_left = 0

    def _settle(self, spawn_x):
        """Wait for physical rest before recording the target's reference pose.

        A fixed 0.4 s wait left the pen rotating at roughly 10 rad/s. Do not
        zero velocities or freeze the object: its contact equilibrium must be
        reached by stepping the same dynamics used during the episode.
        """
        self.data.ctrl[self.actuators] = np.clip(self.hand_spec.neutral + self.bias,
                                               self.lower, self.upper)
        stable = 0
        for sample in range(RESET_MAX_SAMPLES):
            mujoco.mj_step(self.model, self.data, nstep=25)
            mujoco.mj_forward(self.model, self.data)
            pos, _ = self.pose()
            vel = self.data.qvel[self.ov:self.ov + 6]
            retained = (self.contact() and pos[2] > .12
                        and np.linalg.norm(pos[:2] - [spawn_x, .049]) < .07)
            resting = (retained and np.linalg.norm(vel[:3]) < RESET_LINEAR_SPEED
                       and np.linalg.norm(vel[3:]) < RESET_ANGULAR_SPEED and self._reset_ready())
            stable = stable + 1 if resting else 0
            if stable >= RESET_STABLE_SAMPLES:
                return sample + 1
        return None

    def _reset_ready(self):
        # Legacy v7 only waits for object rest. New contracts can additionally
        # require the physical hand to settle, without changing old rollouts.
        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self._randomize()
        for attempt in range(30):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.qadr] = self.hand_spec.neutral
            self.data.ctrl[self.actuators] = self.hand_spec.neutral
            height = .21 if self.config.object == "cube" else .192
            spawn_x = .135 if self.config.object == "cube" else .190
            self.data.qpos[self.oq:self.oq + 3] = [spawn_x + self.np_random.uniform(-.003, .003),
                                                  .049 + self.np_random.uniform(-.003, .003), height]
            a = self.np_random.uniform(-.08, .08)
            self.data.qpos[self.oq + 3:self.oq + 7] = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
            mujoco.mj_forward(self.model, self.data)
            settled_samples = self._settle(spawn_x)
            pos, rot = self.pose()
            if settled_samples is not None:
                break
        else:
            raise RuntimeError("30 resets without supported physical rest: inspect placement/fixture")
        self.origin = pos.copy()
        direction = (int(self.np_random.choice([-1, 1])) if self.negative_goal_probability == .5 else
                     -1 if self.np_random.random() < self.negative_goal_probability else 1)
        self.direction = options.get("direction", direction)
        if self.direction not in (-1, 1):
            raise ValueError("Direction must be -1 or 1")
        # Training-only mixture: half at the full task angle, half uniform
        # from the requested minimum. The default consumes no extra RNG and
        # keeps canonical evaluation and deployment goals unchanged.
        self.goal_angle = self.config.angle
        if self.minimum_goal_angle is not None and self.np_random.random() < .5:
            self.goal_angle = float(self.np_random.uniform(self.minimum_goal_angle, self.config.angle))
        self.increment = yaw_rotation(np.deg2rad(self.goal_angle) * self.direction)
        self.goal = self.increment @ rot
        self.codec = self.action_codec_class(self.config, self.hand_spec.neutral)
        self.observer = ObservationCodec(self.config)
        self.queue = deque([self.hand_spec.neutral.copy() for _ in range(self.delay + 1)], maxlen=self.delay + 1)
        self.steps = self.stable = self.completed = 0
        self.age = 0.
        self.valid = True
        self.measured_pos, self.measured_rot = pos.copy(), rot.copy()
        self.last_error = rotation_error(rot, self.goal)
        self.previous_action = np.zeros(16)
        self.data.time = 0
        return self._observation(), {**self.info(), "reset_attempts": attempt + 1,
                                     "settle_seconds": settled_samples * DT}

    def _observation(self):
        return self.observer.push(self.data.qpos[self.qadr] + self.encoder_bias, self.codec.q,
                                  self.measured_pos, self.measured_rot, self.goal, self.origin,
                                  self.valid, self.age, controller_velocity=self.codec.velocity)

    def info(self):
        pos, rotation = self.pose()
        error = rotation_error(rotation, self.goal)
        displacement = float(np.linalg.norm(pos - self.origin))
        vel = self.data.qvel[self.ov:self.ov + 6]
        linear_speed, angular_speed = float(np.linalg.norm(vel[:3])), float(np.linalg.norm(vel[3:]))
        shaft_error = float(np.rad2deg(np.arccos(np.clip(rotation[:, 1] @ self.goal[:, 1], -1, 1))))
        dropped = bool(pos[2] < .10 or displacement > .12)
        stable = (error < np.deg2rad(8) and displacement < .04 and self.contact()
                  and linear_speed < .08 and angular_speed < 1.)
        return dict(angle_error_deg=float(np.rad2deg(error)), goal_angle_deg=self.goal_angle,
                    displacement_m=displacement,
                    dropped=dropped, stable=bool(stable), completed_turns=self.completed,
                    is_success=self.completed >= self.config.turns, vision_age=self.age,
                    object_position=pos.tolist(), linear_speed_mps=linear_speed,
                    angular_speed_rps=angular_speed,
                    shaft_error_deg=shaft_error if self.config.object == "pen" else None)

    def step(self, action):
        # Observation validity drives the same hold rule used by deployment.
        command = self.codec.step(action, valid=self.valid)
        self.queue.append(command)
        self.data.ctrl[self.actuators] = np.clip(self.queue[0] + self.bias, self.lower, self.upper)
        mujoco.mj_step(self.model, self.data, nstep=25)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        strength = self.config.randomization
        # Bursts of 1..4 lost observations, plus sample-and-hold camera delay.
        if self.dropout_left == 0 and self.np_random.random() < .025 * strength:
            self.dropout_left = int(self.np_random.integers(1, 5))
        self.valid = self.dropout_left == 0 and self.np_random.random() >= .12 * strength
        self.dropout_left = max(0, self.dropout_left - 1)
        if self.valid:
            pos, rot = self.pose()
            self.measured_pos = pos + self.np_random.normal(0, .0015 * strength, 3)
            self.measured_rot = yaw_rotation(self.np_random.normal(0, .015 * strength)) @ rot
            self.age = 0.
        else:
            self.age += DT
        info = self.info()
        error = np.deg2rad(info["angle_error_deg"])
        self.stable = self.stable + 1 if info["stable"] and not info["dropped"] else 0
        subgoal = self.stable >= 10
        # A positive per-step survival reward would make waiting until timeout
        # preferable to completing a short goal and terminating the episode.
        # Use progress plus a small running cost, and make dropping costlier
        # than simply waiting for this task's horizon.
        reward = (6 * (self.last_error - error) - .05 * error / np.pi - .01
                  - .3 * info["displacement_m"] - .002 * np.mean(np.square(action))
                  - .002 * np.mean(np.square(np.asarray(action) - self.previous_action))
                  + 10 * subgoal - (10 + .1*self.config.horizon) * info["dropped"])
        if subgoal:
            self.completed += 1
            self.stable = 0
            if self.completed < self.config.turns:
                self.goal = self.increment @ self.goal
        self.last_error = rotation_error(self.pose()[1], self.goal)
        self.previous_action = np.asarray(action).copy()
        info.update(completed_turns=self.completed, is_success=self.completed >= self.config.turns)
        return (self._observation(), float(reward), info["dropped"] or info["is_success"] or self.age > .25,
                self.steps >= self.config.horizon, info)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.lookat[:] = [.15, .04, .17]
        cam.distance, cam.azimuth, cam.elevation = .42, 145, -55
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()
