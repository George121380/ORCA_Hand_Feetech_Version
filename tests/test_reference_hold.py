import copy
import json
from pathlib import Path

import numpy as np
import pytest

from orca_feetech.manipulation import ManipulationEnv, TaskConfig


@pytest.fixture
def hold_probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import probe_reference_hold
    return probe_reference_hold


def test_reference_contract_rejects_geometry_model_and_joint_range_mismatch(hold_probe):
    reference=json.loads((Path(__file__).parents[1]/'docs/sim2real/retarget-contact-probe.json').read_text())
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    try:
        hold_probe.validate_reference(reference,env)
        for field in ('geometry_sha256','model_sha256'):
            wrong=copy.deepcopy(reference)
            wrong[field]='incorrect'
            with pytest.raises(ValueError,match='contract mismatch'):
                hold_probe.validate_reference(wrong,env)
        wrong=copy.deepcopy(reference)
        wrong['trials'][0]['qpos'][0]=float(env.upper[0]+.1)
        with pytest.raises(ValueError,match='outside model ROM'):
            hold_probe.validate_reference(wrong,env)
        wrong=copy.deepcopy(reference)
        wrong['trials'][0]['object_quaternion_wxyz']=[0.,0,0,0]
        with pytest.raises(ValueError,match='Invalid object pose'):
            hold_probe.validate_reference(wrong,env)
    finally:
        env.close()


def test_hold_advances_free_object_dynamics_and_does_not_report_floor_support(hold_probe):
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    try:
        pose=np.array([.6,0,.8,1.,0,0,0])
        original=pose.copy()
        result=hold_probe.hold(env,env.hand_spec.neutral[:16],pose,.5)
        np.testing.assert_array_equal(pose,original)
        np.testing.assert_allclose(result['initial']['object_position_m'],pose[:3])
        assert result['initial']['linear_speed_mps']==0
        assert result['simulated_s']==pytest.approx(.5)
        assert len(result['samples'])==10
        assert result['end']['object_position_m'][2]<.6
        assert result['peak_displacement_m']>.2
        assert result['peak_linear_speed_mps']>1.
        assert not result['end']['hand_contact']
        assert not result['retained_at_end'] and not result['resting_last_200ms']
        assert result['numerical_failure'] is None
        assert not result['warnings']
        with pytest.raises(ValueError,match='0.5..2 seconds'):
            hold_probe.hold(env,env.hand_spec.neutral[:16],pose,3.)
    finally:
        env.close()


def test_snapshot_distinguishes_axial_roll_from_directed_shaft_motion(hold_probe):
    import mujoco
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    try:
        # The printable pen's shaft is +Y. A rotation about it changes SO(3),
        # but must not be reported as shaft-direction drift.
        env.data.qpos[env.oq:env.oq+3]=[.6,0,.8]
        env.data.qpos[env.oq+3:env.oq+7]=[np.cos(.3),0,np.sin(.3),0]
        mujoco.mj_forward(env.model,env.data)
        sample=hold_probe.snapshot(env,np.array([.6,0,.8]),np.eye(3),env.hand_spec.neutral)
        assert sample['rotation_change_deg']==pytest.approx(np.rad2deg(.6))
        assert sample['shaft_change_deg']==pytest.approx(0,abs=1e-6)
    finally:
        env.close()


def test_preload_target_does_not_teleport_actual_joint_state(hold_probe,monkeypatch):
    import mujoco
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    actual=(env.lower[:16]+env.upper[:16])/2
    target=actual+.03
    step=mujoco.mj_step
    checked=[]
    def observe_initial_state(model,data,**kwargs):
        if not checked:
            np.testing.assert_allclose(data.qpos[env.qadr[:16]],actual)
            np.testing.assert_allclose(data.ctrl[env.actuators[:16]],target)
            checked.append(True)
        return step(model,data,**kwargs)
    monkeypatch.setattr(mujoco,'mj_step',observe_initial_state)
    try:
        result=hold_probe.hold(env,actual,[.6,0,.8,1,0,0,0],.5,target=target)
        assert checked and result['simulated_s']==pytest.approx(.5)
        assert result['initial']['command_error_rms_rad']==pytest.approx(.03)
        assert result['peak_displacement_m']>.2  # Object remains dynamically free.
        for invalid in ([0.]*15,env.upper[:16]+.01,np.full(16,np.nan)):
            with pytest.raises(ValueError,match='16 finite joint positions'):
                hold_probe.hold(env,actual,[.6,0,.8,1,0,0,0],.5,target=invalid)
    finally:
        env.close()
