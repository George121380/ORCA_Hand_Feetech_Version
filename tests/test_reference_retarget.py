from pathlib import Path
import hashlib
from io import BytesIO
import json
import xml.etree.ElementTree as ET
from zipfile import ZipFile

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


def test_proximal_ablation_preserves_default_fit_and_reports_valid_tip_only_fit(probe):
    hand=probe.HandKinematics()
    try:
        known=.8*hand.spec.neutral[:16]+.2*hand.spec.upper[:16]
        tips,proximal=hand.points(known)
        original,_=hand.fit(tips,proximal,hand.spec.neutral[:16])
        explicit,_=hand.fit(tips,proximal,hand.spec.neutral[:16],proximal_weight=.25)
        np.testing.assert_allclose(original.x,explicit.x,atol=1e-12)
        fit,actual=hand.fit(tips,proximal,hand.spec.neutral[:16],proximal_weight=0.)
        assert np.max(np.linalg.norm(actual-tips,axis=1))<.0005
        assert np.all(fit.x>=hand.spec.lower[:16]) and np.all(fit.x<=hand.spec.upper[:16])
        assert hand.data.qpos[hand.env.qadr[-1]]==hand.spec.neutral[-1]
        with pytest.raises(ValueError,match='proximal weight'):
            hand.fit(tips,proximal,hand.spec.neutral[:16],proximal_weight=.1)
    finally:
        hand.close()


def test_contact_refinement_reduces_real_mesh_overlap_without_moving_object(probe):
    hand=probe.HandKinematics()
    try:
        q=hand.spec.neutral[:16].copy()
        tips,proximal=hand.points(q)
        # Deliberately put the pen's centre on an actual robot fingertip.
        # The keypoint-only optimum is reachable, but physically intersects it.
        position=tips[1].copy()
        quaternion=np.array([1.,0,0,0])
        before=hand.contact_depths(position,quaternion)
        assert before.max()>.004
        result,actual=hand.fit(tips,proximal,q,(position,quaternion))
        after=hand.contact_depths(position,quaternion)
        assert before.shape==after.shape==(hand.model.ngeom,)
        assert after.max()<.001 and after.max()<before.max()/4
        assert np.max(np.linalg.norm(actual-tips,axis=1))<.015
        assert np.all(result.x>=hand.spec.lower[:16]) and np.all(result.x<=hand.spec.upper[:16])
        np.testing.assert_allclose(hand.data.qpos[hand.env.oq:hand.env.oq+7],
                                   np.r_[position,quaternion],atol=1e-12)
        assert hand.data.qpos[hand.env.qadr[-1]]==hand.spec.neutral[-1]
        assert after.max()==pytest.approx(max(x['penetration_m']
                                             for x in hand.contacts(position,quaternion)))
    finally:
        hand.close()


def test_self_contact_refinement_fixes_frozen_interfinger_overlap_without_moving_object(probe):
    # Frozen robot-reference frame 0: ring/pinky overlap was missed by the
    # object-only penalty. Read the tracked evidence, not local research data.
    report=json.loads((Path(__file__).parents[1]/'docs/sim2real/retarget-preload-probe.json').read_text())
    q=np.asarray(report['trials'][0]['actual_qpos_rad'])
    hand=probe.HandKinematics()
    pose=(np.array([.6,0,.8]),np.array([1.,0,0,0]))
    try:
        tips,proximal=hand.points(q)
        hand.set_object_pose(*pose)
        initial=hand.self_contact_depths()
        assert initial.shape==(hand.model.ngeom,) and initial.max()>.005
        assert any(set(c['bodies'])=={'right_ring_pp','right_pinky_pp'} for c in hand.self_contacts())
        result,_=hand.fit(tips,proximal,q,pose,self_contact_aware=True)
        hand.set_object_pose(*pose)
        after=hand.self_contact_depths()
        assert after.max()<.002 and after.max()<initial.max()/3
        assert np.all(result.x>=hand.spec.lower[:16]) and np.all(result.x<=hand.spec.upper[:16])
        np.testing.assert_allclose(hand.data.qpos[hand.env.oq:hand.env.oq+7],np.r_[*pose],atol=1e-12)
        assert hand.data.qpos[hand.env.qadr[-1]]==hand.spec.neutral[-1]
        assert hand.contact_depths(*pose).max()==0  # No object support was fabricated.
        with pytest.raises(ValueError,match='requires an object pose'):
            hand.fit(tips,proximal,q,self_contact_aware=True)
    finally:
        hand.close()


def test_source_robot_fk_checks_all_published_links_and_derives_actual_tip_sites(probe,tmp_path):
    import mujoco
    names=['thumb','index_finger','middle_finger','ring_finger','pinky']
    root=ET.Element('mujoco')
    wrist=ET.SubElement(ET.SubElement(root,'worldbody'),'body',name='r_wrist')
    body_names=['r_wrist']
    for finger,name in enumerate(names):
        parent=wrist
        joints=(['cmc_flex','cmc_abd','mcp','ip'] if finger==0 else ['mcp_flex','mcp_abd','pip','dip'])
        for index,part in enumerate(['proximal','proximal_abd','middle','distal']):
            body_name=f'r_{name}_{part}'
            pos=f'{finger*.02} .03 0' if index==0 else '0 .02 0'
            parent=ET.SubElement(parent,'body',name=body_name,pos=pos)
            ET.SubElement(parent,'joint',name=f'r_{name}_{joints[index]}',axis='0 0 1')
            ET.SubElement(parent,'geom',type='sphere',size='.003',mass='.01')
            body_names.append(body_name)
        ET.SubElement(parent,'site',name=f'r_{name}_tip',pos='0 .01 0')
    model_path=tmp_path/'source.xml'
    model_path.write_bytes(ET.tostring(root))
    model=mujoco.MjModel.from_xml_path(str(model_path))
    data=mujoco.MjData(model)
    q=np.arange(60).reshape(3,20)/100
    wrist_poses=np.array([[0,0,0,1,0,0,0],[.1,-.2,.3,np.cos(.2),0,0,np.sin(.2)],
                          [-.1,.1,.2,np.cos(.1),np.sin(.1),0,0]])
    links=[]
    tips=[]
    for joint_pose,pose in zip(q,wrist_poses):
        model.body_pos[model.body('r_wrist').id]=pose[:3]
        model.body_quat[model.body('r_wrist').id]=pose[3:]
        data.qpos[:]=joint_pose
        mujoco.mj_kinematics(model,data)
        links.append(data.xpos[[model.body(n).id for n in body_names]].copy())
        tips.append(data.site_xpos.copy())
    links=np.array(links)
    def archive(published_links):
        buf=BytesIO()
        np.savez(buf,reference_qpos=q,wrist_pose_w=wrist_poses,link_pos_w=published_links)
        raw=buf.getvalue()
        path=tmp_path/'source.zip'
        with ZipFile(path,'w') as z:
            z.writestr('fixture/01_single_axis/clip_001.npz',raw)
        return path,hashlib.sha256(raw).hexdigest()
    path,sha=archive(links)
    points,info=probe.source_robot_points(path,model_path,sha)
    assert points.shape==(3,21,3) and info['checked_frames']==3
    np.testing.assert_allclose(points[:,probe.TIP_IDS],tips,atol=1e-12)
    np.testing.assert_allclose(points[:,0],wrist_poses[:,:3],atol=1e-12)
    assert not info['source_physics_stepped'] and not info['source_joint_angles_copied_to_orca']
    with pytest.raises(ValueError,match='hash mismatch'):
        probe.source_robot_points(path,model_path,'wrong')
    links[-1,-1,0]+=.001  # The last frame must be checked, not only the first.
    path,sha=archive(links)
    with pytest.raises(ValueError,match='does not reproduce published links'):
        probe.source_robot_points(path,model_path,sha)
