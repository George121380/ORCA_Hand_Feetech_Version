from pathlib import Path

import numpy as np

from orca_feetech.pen_grasp import GraspConfig,PenGraspEnv


def test_goal_handoff_keeps_physics_controls_rng_and_preceding_history(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from probe_grasp_rotation_readiness import handoff_goal,physical_control_signature
    env=PenGraspEnv(GraspConfig(randomization=.3))
    try:
        env.reset(seed=2111000,options={'placement_index':0})
        env.step(np.zeros(16))
        before=physical_control_signature(env)
        history=[f.copy() for f in env.observer.frames]
        rotation=env.pose()[1].copy()
        handoff_goal(env,'yaw_plus15')
        assert before==physical_control_signature(env)
        assert np.array_equal(rotation,env.pose()[1])
        assert not np.array_equal(rotation,env.goal)
        assert all(np.array_equal(a,b) for a,b in zip(history[1:],list(env.observer.frames)[:-1]))
    finally:
        env.close()


def test_grasp_success_never_substitutes_for_rotation_gate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from probe_grasp_rotation_readiness import rotation_gate
    info=dict(is_success=True,stable=True,dropped=False,vision_age=0.,angle_error_deg=15.,
              displacement_m=.01,linear_speed_mps=0.,angular_speed_rps=0.)
    assert not rotation_gate(info,True)
    info['angle_error_deg']=7.
    assert rotation_gate(info,True)
    assert not rotation_gate(info,False)
    assert not rotation_gate(info,True,'penetration_limit')
