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

    def fit(self,tips,proximal,previous):
        def residual(q):
            actual,anchors=self.points(q)
            return np.concatenate([(actual-tips).ravel()/.01,
                                   .25*(anchors-proximal).ravel()/.01,.02*(q-previous)])
        result=least_squares(residual,np.clip(previous,self.spec.lower[:16]+1e-8,
                                               self.spec.upper[:16]-1e-8),
                             bounds=(self.spec.lower[:16],self.spec.upper[:16]),
                             max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        actual,_=self.points(result.x)
        return result,actual

    def contacts(self,position,quaternion):
        self.data.qpos[self.env.oq:self.env.oq+3]=position
        self.data.qpos[self.env.oq+3:self.env.oq+7]=quaternion
        self.data.qvel[:]=0
        mujoco.mj_forward(self.model,self.data)
        rows=[]
        for c in self.data.contact:
            if c.dist>0 or not any(g in (c.geom1,c.geom2) for g in self.env.geoms):
                continue
            other=c.geom2 if c.geom1 in self.env.geoms else c.geom1
            body=self.model.body(self.model.geom_bodyid[other]).name
            rows.append(dict(body=body,penetration_m=float(-c.dist)))
        return rows

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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=32)
    args=p.parse_args()
    if not 2<=args.frames<=64:
        p.error('Use2..64 evenly sampled keyframes')
    points,objects,dt,sha=source_clip(args.archive)
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
        source_sha256=sha,source_frame_dt_s=dt,source_frames=len(points),
        sampled_frames=len(ids),maximum_optimizer_nfev_per_frame=60,
        model_sha256=hand.spec.source_sha256,
        geometry_sha256=hashlib.sha256(json.dumps(geometry_signature('pen'),sort_keys=True).encode()).hexdigest(),
        source_from_orca_object_body=SOURCE_FROM_ORCA_BODY.tolist(),
        joint_names=JOINT_NAMES,scale=scale,fingertip_proxy_offsets_m=hand.offsets.tolist(),
        alignment='Per-frame source hand basis mapped to neutral ORCA hand basis; one bone-length scale',
        limitations=['No contact objective in IK; contacts are static mesh intersections, not support forces.',
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
            previous=result.x.copy()
            position=hand.center+scale*((objects[frame,:3]-center)@rotation)@hand.rotation.T
            quat=np.empty(4)
            mapped=object_rotation(objects[frame,3:],rotation,hand.rotation)
            mujoco.mju_mat2Quat(quat,mapped.ravel())
            contacts=hand.contacts(position,quat)
            error=np.linalg.norm(actual-targets[TIP_IDS],axis=1)
            report['trials'].append(dict(frame=int(frame),time_s=float(frame*dt),qpos=result.x.tolist(),
                optimizer_success=bool(result.success),optimizer_nfev=result.nfev,
                tip_errors_m=error.tolist(),tip_rms_m=float(np.sqrt(np.mean(error**2))),
                object_position_m=position.tolist(),object_quaternion_wxyz=quat.tolist(),
                contacts=contacts,deepest_penetration_m=max([x['penetration_m'] for x in contacts],default=0.),
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
    report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
