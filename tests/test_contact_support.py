import mujoco
import numpy as np
import pytest

from orca_feetech.contact_support import gravity_support, hand_object_contacts


@pytest.mark.parametrize('support_body,accepted',[('right_palm',True),('right_tower',False),('floor_body',False)])
def test_engine_contact_sign_and_gravity_balance_exclude_floor_and_tower(support_body,accepted):
    # Use actual solved contact forces, not hand-written wrench expectations.
    model=mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".002" cone="pyramidal"/>
      <worldbody>
        <body name="object" pos="0 0 .031"><freejoint/>
          <geom name="object_geom" type="box" size=".01 .02 .03" mass=".1" condim="4"/>
        </body>
        <body name="{support_body}"><geom type="plane" size="1 1 .1" condim="4"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mujoco.mj_step(model,data,nstep=500)
    mujoco.mj_forward(model,data)
    contacts,excluded=hand_object_contacts(model,data,[model.geom('object_geom').id])
    body=model.body('object').id
    result=gravity_support(contacts,data.xipos[body],model.body_mass[body],model.opt.gravity)
    if accepted:
        assert contacts and not excluded
        assert result['gravity_wrench_feasible_5pct']
        np.testing.assert_allclose(result['solver_contact_force_world_n'],[0,0,.981],atol=1e-3)
        assert result['solver_gravity_wrench_relative_residual']<.005
        assert result['gravity_wrench_relative_residual']<1e-8
    else:
        assert not contacts and support_body in excluded
        assert not result['gravity_wrench_feasible_5pct']
        assert result['gravity_wrench_relative_residual']==pytest.approx(1.)


def test_side_contacts_need_friction_to_support_gravity_and_fit_is_not_actuator_proof():
    def pair(mu):
        return [dict(body='right_a',position_m=np.array([-.01,0,0]),frame=np.eye(3),sign=1.,
                     dimension=3,friction=np.array([mu,mu,0,0,0]),
                     solver_force_world_n=np.zeros(3),solver_torque_world_nm=np.zeros(3)),
                dict(body='right_b',position_m=np.array([.01,0,0]),frame=np.diag([-1,1,-1]),sign=1.,
                     dimension=3,friction=np.array([mu,mu,0,0,0]),
                     solver_force_world_n=np.zeros(3),solver_torque_world_nm=np.zeros(3))]
    frictionless=gravity_support(pair(0),np.zeros(3),.1,np.array([0,0,-9.81]))
    frictional=gravity_support(pair(1),np.zeros(3),.1,np.array([0,0,-9.81]))
    assert not frictionless['gravity_wrench_feasible_5pct']
    assert frictionless['gravity_wrench_relative_residual']==pytest.approx(1.)
    assert frictional['gravity_wrench_feasible_5pct']
    np.testing.assert_allclose(frictional['fitted_contact_force_world_n'],[0,0,.981],atol=1e-8)
    # Feasible hypothetical cone forces do not mean the current actuator
    # targets are generating them: this constructed pair has zero preload.
    assert frictional['solver_gravity_wrench_relative_residual']==pytest.approx(1.)
    assert frictional['required_normal_force_sum_n']>=.981-1e-8
    with pytest.raises(ValueError,match='finite'):
        gravity_support([],np.array([np.nan,0,0]),.1,np.array([0,0,-9.81]))
