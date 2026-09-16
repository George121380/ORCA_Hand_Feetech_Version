import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("orca_sim")

from orca_feetech.backends import SimBackend
from orca_feetech.model import JOINT_NAMES, ModelSpec, poses

pytestmark = pytest.mark.sim


def test_actuator_order_is_named_not_positional():
    robot = SimBackend()
    try:
        assert robot.actuators[-1] == 0  # MuJoCo stores wrist first; project stores it last.
        q = poses()["index"]
        robot.write(q)
        np.testing.assert_allclose(robot.env.data.ctrl[robot.actuators], q, atol=1e-7)
    finally:
        robot.close()


def test_urdf_and_mjcf_frames_agree_for_every_gesture():
    pin = pytest.importorskip("pinocchio")
    from orca_feetech.paths import upstream
    spec = ModelSpec.load()
    robot = SimBackend()
    model = pin.buildModelFromUrdf(str(upstream("orcahand_description") / "v1/models/urdf/orcahand_right.urdf"))
    data = model.createData()
    common = []
    for body in range(1, robot.env.model.nbody):
        name = robot.mujoco.mj_id2name(robot.env.model, robot.mujoco.mjtObj.mjOBJ_BODY, body)
        if model.existFrame(name):
            common.append((body, model.getFrameId(name)))
    assert len(common) == 18
    try:
        rng = np.random.default_rng(123)
        samples = [*poses().values(), *(spec.lower + rng.uniform(size=17) * (spec.upper - spec.lower)
                                       for _ in range(5))]
        for position in samples:
            robot.reset(position)
            q = pin.neutral(model)
            for i, name in enumerate(JOINT_NAMES):
                q[model.joints[model.getJointId("right_" + name)].idx_q] = position[i] - spec.neutral[i]
            pin.framesForwardKinematics(model, data, q)
            for body, frame in common:
                np.testing.assert_allclose(robot.env.data.xpos[body], data.oMf[frame].translation, atol=1e-7)
                np.testing.assert_allclose(robot.env.data.xmat[body].reshape(3, 3),
                                           data.oMf[frame].rotation, atol=1e-7)
    finally:
        robot.close()


def test_gym_contract_and_repeatable_reset():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    from orca_feetech.learning import GestureTrackingEnv
    env = GestureTrackingEnv()
    try:
        check_env(env, warn=True)
        first, _ = env.reset(seed=17)
        second, _ = env.reset(seed=17)
        np.testing.assert_array_equal(first, second)
        obs, reward, terminated, truncated, info = env.step(env.baseline_action())
        assert obs.shape == (64,) and np.isfinite(obs).all()
        assert np.isfinite(reward) and not terminated and not truncated
        assert "mae_deg" in info
    finally:
        env.close()


def test_position_baseline_reaches_goals_and_keeps_wrist_fixed():
    from orca_feetech.learning import GestureTrackingEnv
    env = GestureTrackingEnv()
    try:
        for seed in range(5):
            env.reset(seed=seed)
            for _ in range(env.horizon):
                _, _, _, _, info = env.step(env.baseline_action())
            assert info["is_success"]
            assert info["mae_deg"] < 5
            assert abs(env.backend.env.data.ctrl[env.backend.actuators[-1]] - env.spec.neutral[-1]) < 1e-7
    finally:
        env.close()
