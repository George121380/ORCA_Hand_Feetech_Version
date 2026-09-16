from __future__ import annotations

import numpy as np

from .control import HandState
from .model import JOINT_NAMES, ModelSpec, vector


class MockBackend:
    def __init__(self):
        self.spec = ModelSpec.load()
        self.lower, self.upper = self.spec.lower, self.spec.upper
        self.q = self.spec.neutral.copy()

    def read(self):
        return HandState(self.q.copy())

    def write(self, q, source_timestamp=None):
        self.q = self.spec.clip(q)

    def render(self):
        return None

    def close(self):
        pass


class SimBackend:
    def __init__(self, render_mode=None, rate=20):
        import mujoco
        from orca_sim import OrcaHandRight
        self.mujoco = mujoco
        self.env = OrcaHandRight(version="v1", render_mode=render_mode)
        self.spec = ModelSpec.load()
        self.render_mode = render_mode
        self.renderer = None
        self.camera = None
        self.lower, self.upper = self.spec.lower, self.spec.upper
        model = self.env.model
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_{n}")
                     for n in JOINT_NAMES]
        if -1 in joint_ids or model.nu != 17:
            raise RuntimeError("Official model no longer has the expected 17 named joints")
        self.qadr = model.jnt_qposadr[joint_ids]
        self.vadr = model.jnt_dofadr[joint_ids]
        actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_{n}_actuator")
                     for n in JOINT_NAMES]
        if -1 in actuators:
            raise RuntimeError("Actuator name mismatch")
        self.actuators = np.array(actuators)
        if not np.allclose(model.actuator_ctrlrange[self.actuators],
                           np.column_stack((self.lower, self.upper)), atol=1e-7):
            raise RuntimeError("Pinned model differs from checked control limits")
        self.env.frame_skip = max(1, round(1 / rate / model.opt.timestep))
        self.dt = self.env.frame_skip * model.opt.timestep
        self.reset()

    def reset(self, q=None):
        self.env.reset()
        initial = self.spec.neutral if q is None else self.spec.clip(q)
        self.env.data.qpos[self.qadr] = initial
        self.env.data.ctrl[self.actuators] = initial
        self.mujoco.mj_forward(self.env.model, self.env.data)
        return self.read()

    def read(self):
        return HandState(self.env.data.qpos[self.qadr].copy())

    def write(self, q, source_timestamp=None):
        q = vector(q)
        action = self.env.data.ctrl.copy()
        action[self.actuators] = np.clip(q, self.lower, self.upper)
        self.env.step(action)
        if not np.isfinite(self.env.data.qpos).all():
            raise RuntimeError("Non-finite simulation state")

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = self.mujoco.Renderer(self.env.model, height=480, width=640)
                self.camera = self.mujoco.MjvCamera()
                self.mujoco.mjv_defaultFreeCamera(self.env.model, self.camera)
                self.camera.lookat[:] = [0.07, 0.0, 0.20]
                self.camera.distance = 0.56
                self.camera.elevation = -25
                self.camera.azimuth = 90
            self.renderer.update_scene(self.env.data, camera=self.camera)
            return self.renderer.render()
        return self.env.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.env.close()


def create_backend(name, *, hardware_config=None, render_mode=None, rate=20):
    if name == "mock":
        return MockBackend()
    if name == "sim":
        return SimBackend(render_mode=render_mode, rate=rate)
    if name == "hardware":
        if hardware_config is None:
            raise ValueError("Hardware requires --config with measured calibration")
        from .hardware import HardwareBackend
        return HardwareBackend(hardware_config)
    raise ValueError(f"Unknown backend: {name}")
