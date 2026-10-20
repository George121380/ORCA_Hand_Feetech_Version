import json
from pathlib import Path

import numpy as np
import pytest

from orca_feetech.manipulation import ManipulationEnv,TaskConfig


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import probe_pinch_workspace
    return probe_pinch_workspace


def test_empty_reach_physically_tracks_frozen_target_with_bounded_commands(probe):
    evidence=json.loads((Path(__file__).parents[1]/'docs/sim2real/grasp-acquisition-probe.json').read_text())
    target=evidence['cases'][0]['candidates'][0]['target_qpos_rad']
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    try:
        run=probe.empty_reach(env,target)
        assert run['reached'] and run['command_limits_met']
        assert run['simulated_seconds']==pytest.approx(6.5)
        assert run['target_error_rms_rad']<.03 and run['peak_self_penetration_m']<=.002
        assert np.max(np.abs(np.array(run['final_hand_qpos_rad'])[:16]-target))<.05
        assert all(s['command'][-1]==env.hand_spec.neutral[-1] for s in run['samples'])
        with pytest.raises(ValueError,match='within ROM'):
            probe.empty_reach(env,env.upper[:16]+.01)
    finally:
        env.close()


def test_new_manual_release_preserves_reached_hand_and_does_not_freeze_object(probe):
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    try:
        reach=probe.empty_reach(env,env.hand_spec.neutral[:16])
        assert reach['reached']
        dofs=env.model.jnt_dofadr[env.model.actuator_trnid[env.actuators,0]]
        env.data.qvel[dofs]=np.linspace(-.017,.017,17)  # Deliberately nonzero reached-hand sentinel.
        original_q=env.data.qpos[env.qadr].copy()
        original_v=env.data.qvel[dofs].copy()
        state=probe.manual_release_state(env,[.6,0,.8],np.eye(3))
        np.testing.assert_array_equal(env.data.qpos[env.qadr],original_q)
        np.testing.assert_array_equal(env.data.qvel[dofs],original_v)
        np.testing.assert_array_equal(env.data.qvel[env.ov:env.ov+6],np.zeros(6))
        run=probe.grasp.approach(env,state,np.asarray(reach['final_command_rad']),env.hand_spec.neutral[:16],'index')
        assert run['end']['object_position_m'][2]<.7 and run['end']['linear_speed_mps']>1
        assert not run['complete'] and not run['stable_loaded_pinch']
        with pytest.raises(ValueError,match='proper rotation'):
            probe.manual_release_state(env,[.6,0,.8],np.zeros((3,3)))
    finally:
        env.close()
