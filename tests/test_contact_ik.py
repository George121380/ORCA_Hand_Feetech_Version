import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from orca_feetech.contact_ik import ContactKinematics,transformed_anchors,preload_bounds,contact_command


def test_rigid_contact_targets_and_ik_preserve_source_and_preload():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><joint type="hinge" axis="0 0 1" range="-90 90"/>
      <geom type="sphere" size=".1"/><body name="pad" pos="1 0 0"><geom type="sphere" size=".1"/></body>
      </body></worldbody></mujoco>''')
    source=mujoco.MjData(model)
    source.qpos[0]=.1
    source.qvel[0]=.17
    source.time=4.25
    mujoco.mj_forward(model,source)
    body=model.body('pad').id
    rotation=source.xmat[body].reshape(3,3)
    anchors=[dict(body_id=body,local_point=[0,0,0],local_normal=[1,0,0],
        world_point=source.xpos[body].copy(),world_normal=rotation[:,0].copy())]
    before=(source.qpos.copy(),source.qvel.copy(),source.time)
    command=np.array([.2])  # Preserve the original0.1rad command/actual preload.
    ik=ContactKinematics(model,source,[0],anchors,command,[-1],[1])
    delta=Rotation.from_euler('z',15,degrees=True).as_matrix()
    points,normals=transformed_anchors(anchors,[0,0,0],delta,[0,0,0])
    fit=ik.fit(points,normals,ik.initial,normal_targets=True)
    assert fit['success'] and max(fit['point_errors_m'])<1e-5
    assert max(fit['normal_errors_deg'])<.001
    assert fit['command_target'][0]-fit['actual_joint_target'][0]==pytest.approx(.1)
    np.testing.assert_array_equal(source.qpos,before[0])
    np.testing.assert_array_equal(source.qvel,before[1])
    assert source.time==before[2]


def test_preload_intersection_does_not_enlarge_command_or_actual_rom():
    actual=np.array([.9,-.9])
    command=np.array([.98,-.98])
    lower,upper=preload_bounds(actual,command,[-1,-1],[1,1])
    np.testing.assert_allclose(upper[0],.92)
    np.testing.assert_allclose(lower[1],-.92)
    for target in (lower,upper):
        moved=command+target-actual
        assert np.all(abs(moved)<=1+1e-12) and np.all(abs(target-actual)<=.3+1e-12)
    with pytest.raises(ValueError,match='outside fixed IK bounds'):
        preload_bounds([0],[1.1],[-1],[1])
    # A physical soft-limit residual is retained in the live state. Only the
    # proposed endpoint is bounded; never rewrite the actual initial joints.
    low,high=preload_bounds([1.01],[.99],[-1],[1])
    assert low[0]<high[0]==1.


def test_contact_curve_begins_at_existing_command_and_holds_last_two_seconds():
    initial=np.array([.2,.4])
    knots=initial+np.array([[.02,-.01],[.04,-.02],[.06,-.03],[.08,-.04]])
    np.testing.assert_allclose(contact_command(initial,knots,0),initial)
    for index,time in enumerate((1.25,2.5,3.75,5.)):
        np.testing.assert_allclose(contact_command(initial,knots,time),knots[index])
    for time in (5.1,6.,7.):
        np.testing.assert_allclose(contact_command(initial,knots,time),knots[-1])
