"""Bounded contact-normal/gravity-wrench audit against frozen physical holds."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from orca_feetech import contact_support
from orca_feetech.manipulation import ManipulationEnv, TaskConfig
from probe_reference_hold import geometry_hash, validate_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--hold-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    raw=args.reference.read_bytes()
    reference=json.loads(raw)
    hold_raw=args.hold_report.read_bytes()
    holds=json.loads(hold_raw)
    if holds.get('status')!='complete' or holds.get('reference_sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Frozen hold report must match the reference exactly')
    env=ManipulationEnv(TaskConfig(object='pen',randomization=0))
    try:
        validate_reference(reference,env)
        hold_by_frame={row['frame']:row for row in holds['trials']}
        if set(hold_by_frame)!={row['frame'] for row in reference['trials']}:
            raise ValueError('Reference/hold frame sets differ')
        args.output.mkdir(parents=True,exist_ok=False)
        report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='running',
            purpose='Instantaneous contact/gravity diagnostic compared with already-frozen holds; not policy acceptance',
            is_physics_rollout=False,acceptance_met=False,randomization_applied=False,
            maximum_synthetic_states=128,reference_sha256=hashlib.sha256(raw).hexdigest(),
            hold_report_sha256=hashlib.sha256(hold_raw).hexdigest(),
            model_sha256=env.hand_spec.source_sha256,geometry_sha256=geometry_hash(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            support_module_sha256=hashlib.sha256(Path(contact_support.__file__).read_bytes()).hexdigest(),
            source_url=reference.get('source_url'),reference_attribution=reference.get('reference_attribution'),
            runtime=dict(system=platform.system(),python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__),
            protocol=dict(cone='model pyramidal, including sliding/torsional/rolling components as present',
                torque_length_m=.1,gravity_relative_residual_threshold=.05,
                solver_wrench='mj_forward at synthetic zero-velocity state with nominal position targets',
                physics_steps=0,external_wrenches_applied=False,normal_force_upper_bound=None),
            documentation='https://mujoco.readthedocs.io/en/stable/computation/#contact',
            limitations=['Gravity-wrench feasibility is not force closure or disturbance robustness.',
                'Cone fit ignores actuator limits, compliance, joint torque balance and dynamics.',
                'Instantaneous solver forces at inserted states are not measured hardware forces.',
                'A moving reference phase need not be statically supportable; no reference is approved here.'],trials=[])
        def save():
            temp=args.output/'probe.tmp'
            temp.write_text(json.dumps(report,indent=2)+'\n')
            temp.replace(args.output/'probe.json')
        save()
        for row in reference['trials']:
            pose=np.r_[row['object_position_m'],row['object_quaternion_wxyz']]
            modes={}
            for mode in ('baseline','refined'):
                command=env.hand_spec.neutral.copy()
                command[:16]=row['baseline']['qpos'] if mode=='baseline' else row['qpos']
                mujoco.mj_resetData(env.model,env.data)
                env.data.qpos[env.qadr]=command
                env.data.qpos[env.oq:env.oq+7]=pose
                env.data.ctrl[env.actuators]=command
                mujoco.mj_forward(env.model,env.data)
                contacts,excluded=contact_support.hand_object_contacts(env.model,env.data,env.geoms)
                metrics=contact_support.gravity_support(contacts,env.data.xipos[env.body],
                    env.model.body_mass[env.body],env.model.opt.gravity)
                metrics.update(excluded_contact_bodies=excluded,
                    initial_penetration_m=max((c['penetration_m'] for c in contacts),default=0.),
                    contact_geometry=[dict(body=c['body'],position_m=c['position_m'].tolist(),
                        normal_on_object=(c['sign']*c['frame'][0]).tolist(),dimension=c['dimension'],
                        friction=c['friction'].tolist()) for c in contacts],
                    frozen_hold={k:hold_by_frame[row['frame']][mode][k] for k in
                        ('retained_at_end','resting_last_200ms','shaft_preserved_last_200ms','full_pose_preserved_last_200ms')})
                modes[mode]=metrics
            report['trials'].append(dict(frame=row['frame'],**modes))
            save()
        report['summary']={}
        for mode in ('baseline','refined'):
            rows=[r[mode] for r in report['trials']]
            feasible=[r for r in rows if r['gravity_wrench_feasible_5pct']]
            report['summary'][mode]=dict(states=len(rows),gravity_cone_feasible=len(feasible),
                cone_solver_failures=sum(not r['cone_fit_converged'] for r in rows),
                solver_wrench_near_gravity_5pct=sum(r['solver_gravity_wrench_relative_residual']<=.05 for r in rows),
                median_gravity_cone_relative_residual=float(np.median([r['gravity_wrench_relative_residual'] for r in rows])),
                median_solver_gravity_relative_residual=float(np.median([r['solver_gravity_wrench_relative_residual'] for r in rows])),
                cone_feasible_and_frozen_shaft_held=sum(r['frozen_hold']['shaft_preserved_last_200ms'] for r in feasible),
                cone_feasible_but_frozen_shaft_not_held=sum(not r['frozen_hold']['shaft_preserved_last_200ms'] for r in feasible))
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(report['summary'],indent=2))
    finally:
        env.close()


if __name__=='__main__':
    main()
