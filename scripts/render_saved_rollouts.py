"""Render every saved continuation in order, without any new physics steps.

Geometry is reconstructed from full recorded qpos/qvel. Labels come only from
the original native task result, never from the renderer's force calculation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from orca_feetech.model import ModelSpec
from orca_feetech.pen_grasp import PenGraspEnv,GraspConfig
from manipulation import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--still',type=Path)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    if report['status']!='complete':
        raise ValueError('Render a completed immutable report only')
    cases=[c for c in report['cases'] if 'end' in c]
    if not cases:
        raise ValueError('No recorded continuations')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise ValueError('Refuse overwriting an existing video')
    def no_physics(*args,**kwargs):
        raise RuntimeError('Saved-state rendering may not integrate physics')
    original_step=mujoco.mj_step
    mujoco.mj_step=no_physics
    env=PenGraspEnv(GraspConfig(randomization=.3))
    frames=0
    metadata=[]
    try:
        with imageio.get_writer(args.output,fps=20,codec='libx264',quality=6,macro_block_size=16) as writer:
            for sequence,brief in enumerate(cases):
                path=Path(brief['path'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=brief['sha256']:
                    raise ValueError('Saved case hash mismatch')
                row=json.loads(path.read_text())
                result='PASS' if row['native_task_success'] else 'FAIL'
                color=(90,235,130) if row['native_task_success'] else (245,120,100)
                label=row['role'].replace('selected-delta-development-transfer','transfer').replace('frozen-partial-reference','reference')
                for tick,sample in enumerate(row['samples']):
                    env.data.qpos[:]=sample['qpos_all']
                    env.data.qvel[:]=sample['qvel_all']
                    env.data.time=sample['time_s']
                    mujoco.mj_forward(env.model,env.data)
                    position,rotation=env.pose()
                    np.testing.assert_allclose(position,sample['position'],atol=1e-10,rtol=0)
                    np.testing.assert_allclose(rotation,sample['rotation'],atol=1e-10,rtol=0)
                    canvas=np.full((560,640,3),22,dtype=np.uint8)
                    canvas[48:528]=env.render()
                    cv2.putText(canvas,'ORCA v1 | recorded simulation | 15-degree task',(9,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1)
                    cv2.putText(canvas,f'{sequence+1}/{len(cases)}  case{row["index"]} {label}  final {result}',(9,39),cv2.FONT_HERSHEY_SIMPLEX,.46,color,1)
                    dot=sample['minimum_thumb_target_normal_dot']
                    dot_label='none' if dot is None else f'{dot:.3f}'
                    cv2.putText(canvas,f't={(tick+1)*.05:.2f}/7.00s  SO3={sample["angle_error_deg"]:.2f}deg  normal dot={dot_label}',(9,542),cv2.FONT_HERSHEY_SIMPLEX,.44,(235,235,235),1)
                    cv2.putText(canvas,'Development trajectories | original gates | not a trained-policy evaluation',(9,557),cv2.FONT_HERSHEY_SIMPLEX,.38,(190,190,190),1)
                    writer.append_data(canvas)
                    frames+=1
                    if args.still and sequence==1 and tick==len(row['samples'])-1:
                        args.still.parent.mkdir(parents=True,exist_ok=True)
                        imageio.imwrite(args.still,canvas)
                metadata.append(dict(index=row['index'],seed=row['seed'],role=row['role'],native_task_success=row['native_task_success'],
                    frames=len(row['samples']),case_sha256=brief['sha256']))
    finally:
        env.close()
        mujoco.mj_step=original_step
    write(args.output.with_suffix('.json'),dict(render_only=True,additional_physics_steps=0,
        scope='All recorded continuations in original order; geometry only, labels from original task. No force re-evaluation or policy/hardware acceptance.',
        report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_source_sha256=ModelSpec.load().source_sha256,mujoco_version=mujoco.__version__,
        fps=20,frames=frames,duration_seconds=frames/20,cases=metadata,video_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()))


if __name__=='__main__':
    main()
