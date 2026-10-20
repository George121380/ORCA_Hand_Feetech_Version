from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from orca_feetech.manipulation import ManipulationEnv, TaskConfig, yaw_rotation


def test_default_probe_contract_unchanged_and_multiple_subgoals_have_explicit_horizon(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"scripts"))
    from probe_curriculum import probe_configs
    config = TaskConfig(object="pen",angle=30)
    assert asdict(probe_configs(config,[30])[0]) == asdict(config)
    a,b = probe_configs(config,[30],[1,2])
    assert (a.turns,a.horizon,b.turns,b.horizon) == (1,240,2,480)
    assert a.fingerprint() != b.fingerprint()
    assert a.max_velocity == b.max_velocity and a.randomization == b.randomization
    with pytest.raises(ValueError):
        probe_configs(config,[30],[1,1])
    with pytest.raises(ValueError):
        probe_configs(config,[30],[9])


def test_second_subgoal_needs_another_full_pose_hold_and_changes_only_goal_after_first():
    envs = [ManipulationEnv(TaskConfig(object="pen",angle=30,turns=n,horizon=240*n,randomization=0))
            for n in (1,2)]
    try:
        for env in envs:
            env.reset(seed=31)
            # A stationary aligned fixture isolates the existing hold/counter
            # transition; this is a unit fixture, never an evaluation reset.
            env.goal = env.pose()[1].copy()
            env.last_error = 0.
        first_goal = envs[1].goal.copy()
        for _ in range(15):
            rows = [env.step(np.zeros(16)) for env in envs]
            np.testing.assert_array_equal(envs[0].data.qpos,envs[1].data.qpos)
            if rows[0][2]:
                break
        assert rows[0][4]["is_success"] and rows[0][4]["completed_turns"] == 1
        assert not rows[1][2] and not rows[1][4]["is_success"]
        assert rows[1][4]["completed_turns"] == 1 and envs[1].stable == 0
        np.testing.assert_allclose(envs[1].goal,yaw_rotation(np.deg2rad(30))@first_goal)
        assert envs[1].info()["angle_error_deg"] > 8
        # Merely advancing the goal cannot count another completed subgoal.
        assert envs[1].completed == 1
    finally:
        for env in envs:
            env.close()
