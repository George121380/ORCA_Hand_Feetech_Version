from pathlib import Path

from orca_feetech.pen_grasp import PenGraspEnv,GraspConfig


def test_signature_requires_identical_real_reset_physics_controls_and_rng(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from probe_grasp_exploration import reset_signature
    env=PenGraspEnv(GraspConfig(randomization=.3))
    try:
        env.reset(seed=2111000,options={'placement_index':0})
        original=reset_signature(env)
        env.step([.01]*16)
        assert reset_signature(env)!=original
        env.reset(seed=2111000,options={'placement_index':0})
        assert reset_signature(env)==original
        before=env.bias.copy()
        env.bias[0]+=1e-4
        assert reset_signature(env)!=original
        env.bias[:]=before
        assert reset_signature(env)==original
        env.np_random.random()
        assert reset_signature(env)!=original
    finally:
        env.close()


def test_failed_hold_requirements_are_not_conflated_with_penetration_abort(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from probe_grasp_exploration import failure_flags
    sample=dict(termination_reason='horizon',loaded_opposition=True,resting=True,hand_contact=True,
                final_500ms_orientation_span_deg=.1,final_500ms_position_span_m=.001)
    rows=[sample.copy() for _ in range(10)]
    rows[0]['loaded_opposition']=False
    flags=failure_flags(rows)
    assert flags['loaded_opposition_failed'] and not flags['rest_failed'] and not flags['hard_failure']
    rows[-1]['termination_reason']='penetration_limit'
    assert failure_flags(rows)==dict(hard_failure=True)
