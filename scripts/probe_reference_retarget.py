"""Bounded offline keyframe retargeting probe; not a physics rollout or policy.

Fits human reference fingertip/proximal points to the actual ORCA MJCF within
joint limits. Reports shape mismatch and contacts at the proposed pen pose.
No hardware imports, motor commands, training, or task acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
from zipfile import ZipFile

import mujoco
import numpy as np
from scipy.optimize import least_squares

from audit_wuji_motion import rotate
from orca_feetech.manipulation import TaskConfig, ManipulationEnv
from orca_feetech.model import JOINT_NAMES
from orca_feetech.objects import geometry_signature

FINGERS = ('thumb','index','middle','ring','pinky')
TIP_IDS = np.array([4,8,12,16,20])
BASE_IDS = np.array([1,5,9,13,17])
PROXIMAL_IDS = np.array([3,6,10,14,18])
# WUJI object shaft is+X; our canonical printable shaft is+Y. Columns
# express ORCA body axes in the source object frame (a proper -90deg yaw).
SOURCE_FROM_ORCA_BODY = np.array([[0.,1,0],[-1,0,0],[0,0,1]])


def hand_self_contacts(model,data):
    """Penetrating, model-enabled hand pairs, including finger/base contact."""
    rows=[]
    for c in data.contact:
        bodies=[model.body(model.geom_bodyid[g]).name or '' for g in (c.geom1,c.geom2)]
        if c.dist<0 and all(b.startswith('right_') for b in bodies):
            rows.append(dict(bodies=bodies,geoms=[int(c.geom1),int(c.geom2)],
                             penetration_m=float(-c.dist)))
    return rows


def object_rotation(source_quaternion,source_hand_rotation,target_hand_rotation):
    source_axes=np.array([rotate(source_quaternion,v) for v in np.eye(3)]).T
    return target_hand_rotation@source_hand_rotation.T@source_axes@SOURCE_FROM_ORCA_BODY


def hand_frame(bases, wrist):
    bases, wrist = np.asarray(bases,float), np.asarray(wrist,float)
    if bases.shape!=(5,3) or wrist.shape!=(3,) or not np.isfinite([*bases,wrist]).all():
        raise ValueError('Expected finite five bases and wrist')
    y = bases[2]-wrist
    if np.linalg.norm(y)<1e-6:
        raise ValueError('Degenerate wrist/middle axis')
    y /= np.linalg.norm(y)
    x = bases[1]-bases[3]
    x -= x.dot(y)*y
    if np.linalg.norm(x)<1e-6:
        raise ValueError('Degenerate lateral hand axis')
    x /= np.linalg.norm(x)
    return (bases[0]+bases[4])/2, np.column_stack([x,y,np.cross(x,y)])


class HandKinematics:
    def __init__(self):
        self.env=ManipulationEnv(TaskConfig(object='pen',angle=30,turns=2,horizon=480))
        self.model,self.data=self.env.model,self.env.data
        self.spec=self.env.hand_spec
        self.data.qpos[self.env.qadr]=self.spec.neutral
        mujoco.mj_forward(self.model,self.data)
        self.bases=np.array([self.model.body('right_'+f+'_mp').id for f in FINGERS])
        self.distal=np.array([self.model.body('right_'+f+('_dp' if f=='thumb' else '_ip')).id
                              for f in FINGERS])
        self.proximal=self.distal.copy()
        self.offsets=[]
        # A geometric proxy at the distal cap of each collision mesh. It is not
        # a measured fleshy contact point or a claim of force closure.
        for body in self.distal:
            vertices=[]
            for g in range(self.model.ngeom):
                if (self.model.geom_bodyid[g]!=body or not self.model.geom_contype[g]
                        or self.model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH):
                    continue
                mesh=self.model.geom_dataid[g]
                start=self.model.mesh_vertadr[mesh]
                count=self.model.mesh_vertnum[mesh]
                v=self.model.mesh_vert[start:start+count]
                world=v@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]
                vertices.append((world-self.data.xpos[body])@self.data.xmat[body].reshape(3,3))
            v=np.concatenate(vertices)
            cap=v[v[:,2]>=v[:,2].max()-.0015]
            self.offsets.append(cap.mean(axis=0))
        self.offsets=np.asarray(self.offsets)
        self.center,self.rotation=hand_frame(self.data.xpos[self.bases],
                                            self.data.xpos[self.model.body('right_palm').id])
        # Non-thumb segment lengths avoid relying on the bent neutral tip's
        # straight-line reach for the source-to-robot scale.
        self.finger_lengths=(np.linalg.norm(self.data.xpos[self.distal[1:]]-
                                           self.data.xpos[self.bases[1:]],axis=1)
                             +np.linalg.norm(self.offsets[1:],axis=1))

    def points(self,q):
        self.data.qpos[self.env.qadr[:16]]=q
        self.data.qpos[self.env.qadr[-1]]=self.spec.neutral[-1]
        mujoco.mj_kinematics(self.model,self.data)
        tips=np.array([self.data.xpos[b]+self.data.xmat[b].reshape(3,3)@v
                       for b,v in zip(self.distal,self.offsets)])
        return tips,self.data.xpos[self.proximal].copy()

    def fit(self,tips,proximal,previous,object_pose=None,*,self_contact_aware=False):
        if self_contact_aware and object_pose is None:
            raise ValueError('Self-contact refinement requires an object pose')
        def residual(q):
            actual,anchors=self.points(q)
            terms=[(actual-tips).ravel()/.01,
                   .25*(anchors-proximal).ravel()/.01,.02*(q-previous)]
            if object_pose is not None:
                # Fixed-length, per-geom residual: changing contact counts may
                # not change the least-squares vector's shape. Allow 0.5 mm
                # numerical overlap, then penalize each extra millimetre.
                depth=self.contact_depths(*object_pose)
                terms.append(np.maximum(depth-.0005,0.)/.001)
                if self_contact_aware:
                    terms.append(np.maximum(self.self_contact_depths()-.0005,0.)/.001)
            return np.concatenate(terms)
        result=least_squares(residual,np.clip(previous,self.spec.lower[:16]+1e-8,
                                               self.spec.upper[:16]-1e-8),
                             bounds=(self.spec.lower[:16],self.spec.upper[:16]),
                             max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        actual,_=self.points(result.x)
        return result,actual

    def set_object_pose(self,position,quaternion):
        self.data.qpos[self.env.oq:self.env.oq+3]=position
        self.data.qpos[self.env.oq+3:self.env.oq+7]=quaternion
        self.data.qvel[:]=0
        mujoco.mj_forward(self.model,self.data)

    def contact_depths(self,position,quaternion):
        self.set_object_pose(position,quaternion)
        depth=np.zeros(self.model.ngeom)
        for c in self.data.contact:
            if c.dist>=0 or not any(g in (c.geom1,c.geom2) for g in self.env.geoms):
                continue
            other=c.geom2 if c.geom1 in self.env.geoms else c.geom1
            if other not in self.env.geoms:
                depth[other]=max(depth[other],-c.dist)
        return depth

    def contacts(self,position,quaternion):
        self.set_object_pose(position,quaternion)
        rows=[]
        for c in self.data.contact:
            if c.dist>0 or not any(g in (c.geom1,c.geom2) for g in self.env.geoms):
                continue
            other=c.geom2 if c.geom1 in self.env.geoms else c.geom1
            body=self.model.body(self.model.geom_bodyid[other]).name
            rows.append(dict(body=body,penetration_m=float(-c.dist)))
        return rows

    def self_contacts(self):
        """Actual model-enabled hand/hand pairs, including finger/base contact.

        Does not enable previously filtered pairs or infer mesh distances.
        Call after mj_forward at the desired joint/object state.
        """
        return hand_self_contacts(self.model,self.data)

    def self_contact_depths(self):
        depths=np.zeros(self.model.ngeom)
        for contact in self.self_contacts():
            for geom in contact['geoms']:
                depths[geom]=max(depths[geom],contact['penetration_m'])
        return depths

    def close(self):
        self.env.close()


def source_clip(archive):
    group='01_single_axis'
    clip='clip_001'
    with ZipFile(archive) as z:
        indexes=[x for x in z.namelist() if x.endswith('/'+group+'/index.json')
                 and '/pen_spin/data/rawdata/clips/' in x]
        if len(indexes)!=1:
            raise ValueError('Expected fixed single-axis group')
        path=indexes[0]
        index=json.loads(z.read(path))
        entry=next(x for x in index['clips'] if x['clip_id']==clip)
        name=path.rsplit('/',1)[0]+'/'+entry['file']
        if z.getinfo(name).file_size>64*1024*1024 or index['frame_dt']!=.02:
            raise ValueError('Unexpected source size or frequency')
        raw=z.read(name)
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Reference hash mismatch')
        with np.load(BytesIO(raw),allow_pickle=False) as data:
            landmarks=data['mediapipe_landmarks_w'].astype(float)
            obj=data['object_pose_w'].astype(float)
        if (landmarks.shape[1:]!=(21,3) or obj.shape!=(len(landmarks),7)
                or not np.isfinite(landmarks).all() or not np.isfinite(obj).all()
                or np.any(np.linalg.norm(obj[:,3:],axis=1)<1e-6)):
            raise ValueError('Invalid reference arrays')
        return landmarks,obj,index['frame_dt'],entry['sha256']


def source_robot_points(archive,model_path,expected_sha):
    """Derive source robot landmarks by FK only; never run external Python code.

    Source links are checked against all published link_pos_w frames before
    using the model's fingertip sites. No source joint angles reach ORCA.
    """
    with ZipFile(archive) as z:
        names=[n for n in z.namelist() if n.endswith('/01_single_axis/clip_001.npz')]
        if len(names)!=1 or z.getinfo(names[0]).file_size>64*1024*1024:
            raise ValueError('Expected bounded fixed source clip')
        raw=z.read(names[0])
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Source robot clip hash mismatch')
    with np.load(BytesIO(raw),allow_pickle=False) as source:
        q=source['reference_qpos'].astype(float)
        wrist=source['wrist_pose_w'].astype(float)
        links=source['link_pos_w'].astype(float)
    if (q.ndim!=2 or q.shape[1]!=20 or wrist.shape!=(len(q),7)
            or links.shape!=(len(q),21,3) or not np.isfinite(q).all()
            or not np.isfinite(wrist).all() or not np.isfinite(links).all()
            or np.any(np.linalg.norm(wrist[:,3:],axis=1)<1e-6)):
        raise ValueError('Invalid source robot arrays')
    model=mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data=mujoco.MjData(model)
    names=['thumb','index_finger','middle_finger','ring_finger','pinky']
    expected_joints=['r_thumb_cmc_flex','r_thumb_cmc_abd','r_thumb_mcp','r_thumb_ip']
    expected_joints += [f'r_{f}_{j}' for f in names[1:] for j in ('mcp_flex','mcp_abd','pip','dip')]
    if model.nq!=20 or [model.joint(i).name for i in range(model.njnt)]!=expected_joints:
        raise ValueError('Unexpected source model joint contract')
    wrist_id=model.body('r_wrist').id
    bodies=np.array([model.body('r_wrist').id]+[model.body(f'r_{f}_{part}').id
        for f in names for part in ('proximal','proximal_abd','middle','distal')])
    sites=np.array([model.site(f'r_{f}_tip').id for f in names])
    points=np.empty((len(q),21,3))
    errors=[]
    for frame in range(len(q)):
        model.body_pos[wrist_id]=wrist[frame,:3]
        model.body_quat[wrist_id]=wrist[frame,3:]/np.linalg.norm(wrist[frame,3:])
        data.qpos[:]=q[frame]
        mujoco.mj_kinematics(model,data)
        actual=data.xpos[bodies]
        errors.append(np.linalg.norm(actual-links[frame],axis=1))
        points[frame,0]=actual[0]
        for finger in range(5):
            # Four human-layout slots per digit: base, next flexion joint,
            # distal flexion joint, actual model tip site. Abduction links
            # are intentionally not mistaken for fingertips.
            start=1+4*finger
            points[frame,start:start+3]=actual[[start,start+2,start+3]]
            points[frame,start+3]=data.site_xpos[sites[finger]]
    maximum=float(np.max(errors))
    if maximum>1e-4:
        raise ValueError(f'Source FK does not reproduce published links: {maximum}m')
    return points,dict(model_xml_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        checked_frames=len(q),maximum_link_position_error_m=maximum,
        mean_link_position_error_m=float(np.mean(errors)),source_joint_order=expected_joints,
        point_mapping='wrist; per digit: proximal origin, middle origin, distal origin, model tip site',
        source_physics_stepped=False,source_joint_angles_copied_to_orca=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=32)
    p.add_argument('--contact-aware',action='store_true',
                   help='Refine each baseline keyframe with a fixed object pose and penetration penalty')
    p.add_argument('--self-contact-aware',action='store_true',
                   help='Also penalize model-enabled hand/hand overlap; requires --contact-aware')
    p.add_argument('--object-z-offset-mm',type=int,choices=[0,-10,-20],default=0,
                   help='Bounded reference-placement diagnostic; moves only the object reference in world Z')
    p.add_argument('--wuji-model',type=Path,
                   help='Optional raw WUJI Hand2 XML: derive source robot tip sites instead of human landmarks')
    args=p.parse_args()
    if not 2<=args.frames<=64:
        p.error('Use2..64 evenly sampled keyframes')
    if args.object_z_offset_mm and not args.contact_aware:
        p.error('Placement candidates require --contact-aware to refit the hand')
    if args.self_contact_aware and not args.contact_aware:
        p.error('Self-contact refinement requires --contact-aware')
    points,objects,dt,sha=source_clip(args.archive)
    source_fk=None
    if args.wuji_model:
        points,source_fk=source_robot_points(args.archive,args.wuji_model,sha)
    hand=HandKinematics()
    source_lengths=np.array([sum(np.linalg.norm(np.diff(f,axis=0),axis=1))
                             for f in points[0].reshape(21,3)[5:].reshape(4,4,3)])
    scale=float(np.median(hand.finger_lengths/source_lengths))
    ids=np.unique(np.linspace(0,len(points)-1,args.frames).round().astype(int))
    report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='running',
        purpose='Offline keyframe geometry probe; not physical execution, training, deployment or acceptance',
        runtime=dict(system=platform.system(),python=platform.python_version(),
                     mujoco=mujoco.__version__,numpy=np.__version__),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        is_physics_rollout=False,acceptance_met=False,clip='01_single_axis/clip_001',
        source_url='https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.9.27',
        reference_attribution='Copyright2026 Wuji Technology Co., Ltd.; see docs/sim2real/third-party/wuji/',
        source_sha256=sha,source_frame_dt_s=dt,source_frames=len(points),
        sampled_frames=len(ids),maximum_optimizer_nfev_per_frame=60,
        contact_refinement=dict(enabled=args.contact_aware,maximum_nfev_per_frame=60,
                                penetration_allowance_m=.0005,residual_scale_m=.001,
                                optimize_object_pose=False,self_contact_aware=args.self_contact_aware),
        model_sha256=hand.spec.source_sha256,
        geometry_sha256=hashlib.sha256(json.dumps(geometry_signature('pen'),sort_keys=True).encode()).hexdigest(),
        source_from_orca_object_body=SOURCE_FROM_ORCA_BODY.tolist(),
        reference_object_translation_m=[0.,0.,args.object_z_offset_mm/1000],
        reference_point_source='wuji_robot_sites' if source_fk else 'human_mediapipe',
        source_robot_fk=source_fk,
        joint_names=JOINT_NAMES,scale=scale,fingertip_proxy_offsets_m=hand.offsets.tolist(),
        alignment='Per-frame source hand basis mapped to neutral ORCA hand basis; one bone-length scale',
        limitations=['Contacts are static mesh intersections, not support forces or physical replay.',
                     'Overlap penalties are soft; neither contact forces nor joint/actuator balance are enforced.',
                     'Hand/hand overlap is penalized only with the explicit --self-contact-aware option.',
                     'Source motions may need different grasp topology, object alignment and retiming.',
                     'Fingertips are distal mesh-cap proxies, not calibrated real contact points.',
                     'No policy or training reset asset may use these states without a separate physics check.'],
        trials=[])
    args.output.mkdir(parents=True,exist_ok=False)
    def save():
        temp=args.output/'probe.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n')
        temp.replace(args.output/'probe.json')
    save()
    previous=hand.spec.neutral[:16].copy()
    try:
        for frame in ids:
            center,rotation=hand_frame(points[frame,BASE_IDS],points[frame,0])
            targets=hand.center+scale*((points[frame]-center)@rotation)@hand.rotation.T
            result,actual=hand.fit(targets[TIP_IDS],targets[PROXIMAL_IDS],previous)
            # Baseline warm starts stay identical in both modes. Refinement
            # never changes the next frame's baseline fit or the object pose.
            previous=result.x.copy()
            position=hand.center+scale*((objects[frame,:3]-center)@rotation)@hand.rotation.T
            position+=np.array([0.,0.,args.object_z_offset_mm/1000])
            quat=np.empty(4)
            mapped=object_rotation(objects[frame,3:],rotation,hand.rotation)
            mujoco.mju_mat2Quat(quat,mapped.ravel())
            baseline_error=np.linalg.norm(actual-targets[TIP_IDS],axis=1)
            baseline_contacts=hand.contacts(position,quat)
            baseline_self=hand.self_contacts()
            baseline=dict(qpos=result.x.tolist(),optimizer_nfev=result.nfev,
                          tip_rms_m=float(np.sqrt(np.mean(baseline_error**2))),
                          deepest_penetration_m=max([x['penetration_m'] for x in baseline_contacts],default=0.),
                          self_contacts=baseline_self,
                          deepest_self_penetration_m=max([x['penetration_m'] for x in baseline_self],default=0.))
            if args.contact_aware:
                result,actual=hand.fit(targets[TIP_IDS],targets[PROXIMAL_IDS],previous,(position,quat),
                                       self_contact_aware=args.self_contact_aware)
            contacts=hand.contacts(position,quat)
            self_contacts=hand.self_contacts()
            error=np.linalg.norm(actual-targets[TIP_IDS],axis=1)
            report['trials'].append(dict(frame=int(frame),time_s=float(frame*dt),qpos=result.x.tolist(),
                optimizer_success=bool(result.success),optimizer_nfev=result.nfev,baseline=baseline,
                tip_errors_m=error.tolist(),tip_rms_m=float(np.sqrt(np.mean(error**2))),
                object_position_m=position.tolist(),object_quaternion_wxyz=quat.tolist(),
                contacts=contacts,deepest_penetration_m=max([x['penetration_m'] for x in contacts],default=0.),
                self_contacts=self_contacts,
                deepest_self_penetration_m=max([x['penetration_m'] for x in self_contacts],default=0.),
                tower_contact=any(x['body']=='right_tower' for x in contacts)))
            save()
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise
    finally:
        hand.close()
    report['summary']=dict(mean_tip_rms_m=float(np.mean([x['tip_rms_m'] for x in report['trials']])),
        worst_tip_error_m=max(max(x['tip_errors_m']) for x in report['trials']),
        frames_with_penetration_over_2mm=sum(x['deepest_penetration_m']>.002 for x in report['trials']),
        frames_without_object_contact=sum(not x['contacts'] for x in report['trials']),
        frames_with_tower_contact=sum(x['tower_contact'] for x in report['trials']))
    report['summary']['deepest_penetration_m']=max(x['deepest_penetration_m'] for x in report['trials'])
    report['summary']['optimizer_converged_frames']=sum(x['optimizer_success'] for x in report['trials'])
    report['summary']['baseline_mean_tip_rms_m']=float(np.mean([x['baseline']['tip_rms_m'] for x in report['trials']]))
    report['summary']['baseline_deepest_penetration_m']=max(x['baseline']['deepest_penetration_m'] for x in report['trials'])
    report['summary']['deepest_self_penetration_m']=max(x['deepest_self_penetration_m'] for x in report['trials'])
    report['summary']['frames_with_self_penetration_over_2mm']=sum(x['deepest_self_penetration_m']>.002 for x in report['trials'])
    report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
