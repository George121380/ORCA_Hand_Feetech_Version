import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from orca_feetech.manipulation import ManipulationEnv, TaskConfig


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import probe_reference_preload
    return probe_reference_preload


def test_static_objective_is_history_independent_and_does_not_advance_physics(probe):
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    reference=json.loads((Path(__file__).parents[1]/'docs/sim2real/retarget-contact-probe.json').read_text())
    row=reference['trials'][0]
    q=np.asarray(row['qpos'])
    pose=np.r_[row['object_position_m'],row['object_quaternion_wxyz']]
    try:
        first,_=probe.instantaneous(env,q,pose,q)
        env.data.qacc_warmstart[:]=1000
        env.data.qvel[:]=1
        second,_=probe.instantaneous(env,q,pose,q)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(env.data.qpos[env.qadr[:16]],q)
        np.testing.assert_array_equal(env.data.qpos[env.oq:env.oq+7],pose)
        assert env.data.time==0 and np.count_nonzero(env.data.qvel)==0
        fit=probe.solve_preload(env,q,pose)
        target=np.asarray(fit['target_rad'])
        assert np.all(target>=env.lower[:16]) and np.all(target<=env.upper[:16])
        assert np.max(np.abs(target-q))<=.15+1e-12
        assert fit['optimizer_nfev']<=60 and fit['objective_forward_calls']<=1200
        assert fit['selected_objective_norm']<=fit['initial_objective_norm']
        assert env.data.time==0
    finally:
        env.close()


def test_frozen_input_hashes_and_unique_frame_sets_are_required(probe):
    reference=json.dumps(dict(trials=[dict(frame=1)])).encode()
    sha=hashlib.sha256(reference).hexdigest()
    hold=json.dumps(dict(status='complete',reference_sha256=sha,trials=[dict(frame=1)])).encode()
    audit=dict(status='complete',reference_sha256=sha,hold_report_sha256=hashlib.sha256(hold).hexdigest(),
               trials=[dict(frame=1)])
    probe.validate_inputs(reference,hold,json.dumps(audit).encode())
    audit['hold_report_sha256']='wrong'
    with pytest.raises(ValueError,match='match the reference exactly'):
        probe.validate_inputs(reference,hold,json.dumps(audit).encode())
    audit['hold_report_sha256']=hashlib.sha256(hold).hexdigest()
    audit['trials']*=2
    with pytest.raises(ValueError,match='Duplicate or mismatched'):
        probe.validate_inputs(reference,hold,json.dumps(audit).encode())
