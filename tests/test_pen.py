import numpy as np
import pytest

pytest.importorskip("mujoco")
from orca_feetech.pen import PenReorientationEnv

pytestmark = pytest.mark.sim


def test_pen_reset_has_contact_and_no_free_goal_completion():
    env = PenReorientationEnv()
    try:
        for seed in range(20):
            obs, info = env.reset(seed=seed)
            assert env.observation_space.contains(obs)
            assert info["hand_contact"] and not info["dropped"]
            assert info["angle_error_deg"] == pytest.approx(90)
            for _ in range(20):
                obs, reward, done, truncated, info = env.step(np.zeros(16))
            assert not info["is_success"]
            assert np.isfinite(obs).all() and np.isfinite(reward)
    finally:
        env.close()


def test_reset_resamples_unsupported_initializations():
    env = PenReorientationEnv()
    try:
        for index in range(600):
            _, info = env.reset(seed=1 if index == 0 else None)
            assert info["hand_contact"] and not info["dropped"]
            assert 1 <= info["reset_attempts"] <= 20
    finally:
        env.close()


def test_airborne_pen_cannot_succeed_by_aligning_to_goal():
    import mujoco
    env = PenReorientationEnv()
    try:
        env.reset(seed=0)
        env.data.qpos[env.pen_qadr:env.pen_qadr + 3] = env.initial_position + [0, 0, .08]
        yaw = np.arctan2(env.goal_axis[1], env.goal_axis[0]) - np.pi / 2
        env.data.qpos[env.pen_qadr + 3:env.pen_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        info = env._info()
        assert info["angle_error_deg"] < 1e-5
        assert not info["hand_contact"] and not info["at_goal_stable"]
        env.data.qpos[env.pen_qadr + 2] = .02
        mujoco.mj_forward(env.model, env.data)
        assert env._info()["dropped"]
    finally:
        env.close()
