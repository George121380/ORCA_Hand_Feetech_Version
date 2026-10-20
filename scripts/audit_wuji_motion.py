"""Read-only audit of the official WUJI reference archive; never command a robot.

This measures the source hand's sampled reference, before ORCA retargeting.
Numerical derivatives include recording/retargeting noise and are not measured
motor velocities or proof of physical feasibility. No third-party code runs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def rotate(q, v):
    """Unit wxyz quaternion applied to vectors with matching leading shapes."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    uv = np.cross(q[..., 1:], v)
    return v + 2*(q[..., :1]*uv + np.cross(q[..., 1:], uv))


def wrist_local(pose, points):
    inverse = pose[:, 3:].astype(float).copy()
    inverse[:, 1:] *= -1
    return rotate(inverse[:, None, :], points-pose[:, None, :3])


def audit(archive):
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
        purpose="Reference data audit before retargeting; not ORCA training or policy evaluation",
        source="https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.9.27",
        reference_attribution="Copyright2026 Wuji Technology Co., Ltd.; Apache-2.0 (official asset archive)",
        reference_license_and_notice="docs/sim2real/third-party/wuji/",
        source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        derivative_definition="Unsmoothed finite differences of reference_qpos; pooled over all joints and times",
        limits_compared=dict(velocity_rad_s=.8,acceleration_rad_s2=2.),
        limitations=["Source hand has20 finger joints; ORCA has16. No joint or contact mapping is validated.",
                     "Raw derivatives include reference noise; do not infer actual robot speed or a safe retiming factor.",
                     "Retiming can change dynamic contact feasibility; a slow reference must be revalidated in physics."],
        groups=[])
    preview = None
    with ZipFile(archive) as z:
        indexes = sorted(p for p in z.namelist() if '/pen_spin/data/rawdata/clips/' in p
                         and p.endswith('/index.json'))
        if len(indexes) != 5:
            raise ValueError("Expected the five groups in the official2026.9.27 archive")
        for path in indexes:
            index = json.loads(z.read(path))
            dt = index['frame_dt']
            if dt != .02 or len(index['clips']) != index['clip_count']:
                raise ValueError("Unexpected reference frequency or clip count")
            velocities, accelerations, clips = [], [], []
            for entry in index['clips']:
                if Path(entry['file']).name != entry['file']:
                    raise ValueError("Clip name must be a single path component")
                name = path.rsplit('/',1)[0]+'/'+entry['file']
                if z.getinfo(name).file_size > 64*1024*1024:
                    raise ValueError("Unexpectedly large clip")
                raw = z.read(name)
                if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                    raise ValueError("Clip hash mismatch")
                with np.load(BytesIO(raw),allow_pickle=False) as data:
                    q = data['reference_qpos'].astype(float)
                    n = len(q)
                    if q.shape != (n,20) or n < 3 or not np.isfinite(q).all():
                        raise ValueError("Expected a finiteNx20 reference")
                    v = np.diff(q,axis=0)/dt
                    a = np.diff(v,axis=0)/dt
                    velocities.append(np.abs(v).ravel())
                    accelerations.append(np.abs(a).ravel())
                    clips.append(dict(id=entry['clip_id'],sha256=entry['sha256'],frames=n,
                                      sample_time_span_s=(n-1)*dt))
                    if index['group_id']=='02_continuous_single_axis' and entry['clip_id']=='clip_001':
                        wrist, obj, links = (data[k] for k in ['wrist_pose_w','object_pose_w','link_pos_w'])
                        if wrist.shape!=(n,7) or obj.shape!=(n,7) or links.shape!=(n,21,3):
                            raise ValueError("Unexpected reference preview shapes")
                        if not all(np.isfinite(x).all() for x in [wrist,obj,links]):
                            raise ValueError("Nonfinite preview")
                        for pose in [wrist,obj]:
                            if not np.allclose(np.linalg.norm(pose[:,3:],axis=1),1.,atol=1e-3):
                                raise ValueError("Nonunit reference quaternion")
                        axis = rotate(obj[:,3:],np.broadcast_to([1.,0,0],(n,3)))
                        ends = obj[:,None,:3] + axis[:,None,:]*np.array([-.1,.1])[None,:,None]
                        ids = np.linspace(0,n-1,80).round().astype(int)
                        preview = dict(label="WUJI reference clip; not an ORCA rollout",
                            clip='02_continuous_single_axis/clip_001',sha256=entry['sha256'],
                            units='mm in wrist frame; pen markers are virtual axis points at+-100mm',
                            times_s=(ids*dt).round(3).tolist(),
                            hand_points_mm=(wrist_local(wrist,links)[ids]*1000).round(2).tolist(),
                            pen_axis_points_mm=(wrist_local(wrist,ends)[ids]*1000).round(2).tolist())
            v,a=np.concatenate(velocities),np.concatenate(accelerations)
            report['groups'].append(dict(id=index['group_id'],clips=clips,clip_count=len(clips),
                reference_dt_s=dt,velocity_abs_p99_rad_s=float(np.percentile(v,99)),
                velocity_abs_max_rad_s=float(v.max()),fraction_joint_deltas_above_0_8=float(np.mean(v>.8)),
                acceleration_abs_p99_rad_s2=float(np.percentile(a,99))))
    report['total_clips']=sum(g['clip_count'] for g in report['groups'])
    if report['total_clips']!=40 or preview is None:
        raise ValueError("Incomplete official reference archive")
    report['preview']=preview
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','preview')},indent=2))


if __name__=='__main__':
    main()
