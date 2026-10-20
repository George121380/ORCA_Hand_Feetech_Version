from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from orca_feetech.manipulation import ManipulationEnv, TaskConfig, yaw_rotation


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import probe_grasp_acquisition
    return probe_grasp_acquisition


def test_pinch_targets_preserve_shaft_station_and_opposed_radial_geometry(probe):
    tips=np.arange(15,dtype=float).reshape(5,3)/100
    original=tips.copy()
    position=np.array([.15,.05,.17])
    rotation=yaw_rotation(.7)
    targets=probe.pinch_targets(tips,position,rotation,1,-.040,.002)
    center=position-.040*rotation[:,1]
    np.testing.assert_allclose((targets[0]+targets[1])/2,center,atol=1e-12)
    assert np.linalg.norm(targets[0]-center)==pytest.approx(.010)
    assert (targets[0]-center)@rotation[:,1]==pytest.approx(0.,abs=1e-12)
    np.testing.assert_array_equal(targets[2:],tips[2:])
    np.testing.assert_array_equal(tips,original)
    for index,station,clearance in ((4,-.04,.002),(1,.1,.002),(1,-.04,.1)):
        with pytest.raises(ValueError):
            probe.pinch_targets(tips,position,rotation,index,station,clearance)


def test_loaded_opposition_uses_actual_solver_forces_and_named_distal_bodies(probe):
    model=mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".002" cone="pyramidal"/>
      <worldbody>
        <body name="object" pos="0 0 .0095"><freejoint/>
          <geom name="object_geom" type="box" size=".01 .01 .01" mass=".1" condim="4"/>
        </body>
        <body name="right_thumb_dp" pos="-.0155 0 .01"><geom type="box" size=".006 .02 .02"/></body>
        <body name="right_index_ip" pos=".0155 0 .01"><geom type="box" size=".006 .02 .02"/></body>
        <body name="right_palm"><geom type="plane" size="1 1 .1"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    env=SimpleNamespace(model=model,data=data,geoms=[model.geom('object_geom').id],body=model.body('object').id)
    result=probe.pinch_load(env,'index')
    assert result['normal_force_by_body_n']['right_thumb_dp']>0
    assert result['normal_force_by_body_n']['right_index_ip']>0
    assert result['minimum_thumb_target_normal_dot']==pytest.approx(-1.)
    assert result['loaded_opposition']
    assert not probe.pinch_load(env,'middle')['loaded_opposition']


def test_approach_preserves_real_initial_velocity_and_limits_commands(probe,monkeypatch):
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    try:
        _,command,_=probe.physical.prepare(env,1980000,1,5.)
        state=probe.physical.branch_state(env)
        before=state.copy()
        qpos,qvel=env.data.qpos.copy(),env.data.qvel.copy()
        assert np.linalg.norm(qvel[env.ov:env.ov+6])>1e-10
        step=mujoco.mj_step
        checked=[]
        def capture(model,data,**kwargs):
            if not checked:
                np.testing.assert_array_equal(data.qpos,qpos)
                np.testing.assert_array_equal(data.qvel,qvel)
                checked.append(True)
            return step(model,data,**kwargs)
        monkeypatch.setattr(mujoco,'mj_step',capture)
        baseline=probe.approach(env,state,command,command[:16],'index')
        assert checked and baseline['complete']
        assert baseline['simulated_seconds']==pytest.approx(7.)
        assert baseline['initial']['angular_speed_rps']==pytest.approx(np.linalg.norm(qvel[env.ov+3:env.ov+6]))
        assert not baseline['stable_loaded_pinch'] and not baseline['acceptance_met']
        np.testing.assert_array_equal(state,before)
        moving=probe.approach(env,state,command,(env.lower[:16]+env.upper[:16])/2,'middle')
        assert moving['samples'] and moving['command_limits_met']
        assert all(s['command'][-1]==env.hand_spec.neutral[-1] for s in moving['samples'])
        for invalid in ([0.]*15,env.upper[:16]+.01,np.full(16,np.nan)):
            with pytest.raises(ValueError,match='finite target inside'):
                probe.approach(env,state,command,invalid,'index')
    finally:
        env.close()
