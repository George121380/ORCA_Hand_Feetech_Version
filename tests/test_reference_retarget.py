from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import probe_reference_retarget
    return probe_reference_retarget


def test_object_axis_mapping_preserves_oriented_shaft_across_hand_frames(probe):
    from orca_feetech.manipulation import yaw_rotation
    source=yaw_rotation(.37)
    target=yaw_rotation(-.61)
    q=np.array([np.cos(.2),0,np.sin(.2),0])
    result=probe.object_rotation(q,source,target)
    expected=target@source.T@probe.rotate(q,[1,0,0])
    np.testing.assert_allclose(result[:,1],expected,atol=1e-12)
    np.testing.assert_allclose(result.T@result,np.eye(3),atol=1e-12)
    assert np.linalg.det(result)==pytest.approx(1)


def test_hand_frame_is_equivariant_to_rigid_scene_movement(probe):
    from orca_feetech.manipulation import yaw_rotation
    bases=np.array([[.03,.02,0],[.03,.08,0],[0,.09,0],[-.03,.08,0],[-.05,.05,0]])
    wrist=np.array([0.,-.02,0])
    center,frame=probe.hand_frame(bases,wrist)
    rotation=yaw_rotation(.81)
    offset=np.array([.2,-.1,.3])
    transformed_center,transformed_frame=probe.hand_frame(bases@rotation.T+offset,wrist@rotation.T+offset)
    np.testing.assert_allclose(transformed_center,rotation@center+offset,atol=1e-12)
    np.testing.assert_allclose(transformed_frame,rotation@frame,atol=1e-12)
    with pytest.raises(ValueError,match='Degenerate'):
        probe.hand_frame(np.zeros((5,3)),np.zeros(3))


def test_actual_mjcf_ik_recovers_reachable_points_and_preserves_wrist(probe):
    hand=probe.HandKinematics()
    try:
        known=.8*hand.spec.neutral[:16]+.2*hand.spec.upper[:16]
        tips,proximal=hand.points(known)
        result,actual=hand.fit(tips,proximal,hand.spec.neutral[:16])
        assert np.all(result.x>=hand.spec.lower[:16]) and np.all(result.x<=hand.spec.upper[:16])
        np.testing.assert_allclose(actual,tips,atol=1e-4)
        assert hand.data.qpos[hand.env.qadr[-1]]==hand.spec.neutral[-1]
    finally:
        hand.close()
