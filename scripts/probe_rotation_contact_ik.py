"""At most18 physical preparations/rollouts and48 isolated contact IK fits.

Twelve privileged proposals on three previously seen development states.
No live state injection, neural training, formal credit, retry or hardware.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from orca_feetech.contact_ik import ContactKinematics,transformed_anchors,contact_command
from orca_feetech.contact_support import hand_object_contacts
from orca_feetech.manipulation import DT
from orca_feetech.pen_rotate import PenRotateEnv,RotationConfig,PREFIX_SHA256
from manipulation import write
from probe_rotation_authority import contacts
from probe_rotation_goal_response import paired_signature


ROOT=Path(__file__).resolve().parents[1]


def plan_file():
    path=ROOT/'configs/training/pen-rotate-contact-ik-diagnostic-v1.json'
    plan=json.loads(path.read_text())
    if (plan['rotation_fingerprint']!=RotationConfig().fingerprint() or plan['prefix_checkpoint_sha256']!=PREFIX_SHA256
            or plan['maximum_cases']!=18 or plan['maximum_continuations']!=15 or plan['extensions']!=0
            or plan['knot_seconds']!=[1.25,2.5,3.75,5.] or plan['ik']['maximum_fits']!=48
            or plan['ik']['max_nfev_per_fit']!=60 or plan['ik']['joint_delta_limit_rad']!=.3
            or [(s['seed'],s['placement_index'],s['direction']) for s in plan['states']]
            !=[(2511000,0,1),(2511001,1,1),(2511002,2,1)]
            or [(c['normal_targets'],c['lift_m']) for c in plan['conditions']]
            !=[(False,0.),(True,0.),(False,.003),(True,.003)]):
        raise ValueError('Changed contact diagnostic protocol; register a separate experiment')
    return plan,hashlib.sha256(path.read_bytes()).hexdigest()


def anchors_from_prefix(env):
    carrier=env.carrier
    rows,excluded=hand_object_contacts(carrier.model,carrier.data,carrier.geoms)
    if excluded or not env.info()['loaded_opposition']:
        raise ValueError('Require the actual loaded thumb/middle prefix; no substituted contact state')
    weight=env.info()['object_weight_n']
    candidates=[]
    for contact in rows:
        normal=contact['sign']*contact['frame'][0]
        force=max(0.,float(contact['solver_force_world_n']@normal))
        if force<=0:
            continue
        candidates.append(dict(contact=contact,normal=normal,force=force))
    selected=[]
    for body in ('right_thumb_dp','right_middle_ip'):
        group=[c for c in candidates if c['contact']['body']==body]
        if not group:
            raise ValueError('Missing required loaded contact')
        selected.append(max(group,key=lambda c:c['force']))
    for finger in ('index','ring','pinky'):
        group=[c for c in candidates if c['contact']['body'].startswith('right_'+finger+'_') and c['force']>=.05*weight]
        if group:
            selected.append(max(group,key=lambda c:c['force']))
    anchors=[]
    for chosen in selected:
        c,normal=chosen['contact'],chosen['normal']
        body=carrier.model.body(c['body']).id
        rotation=carrier.data.xmat[body].reshape(3,3)
        anchors.append(dict(body_id=body,body=c['body'],world_point=c['position_m'].tolist(),
            world_normal=normal.tolist(),normal_force_n=chosen['force'],
            local_point=(rotation.T@(c['position_m']-carrier.data.xpos[body])).tolist(),
            local_normal=(rotation.T@normal).tolist()))
    return anchors


def hypothetical_geometry(carrier,scratch,position,rotation):
    # Only an isolated planning MjData is edited, never carrier.data.
    if scratch is carrier.data:
        raise RuntimeError('Refuse hypothetical object editing on live execution data')
    quaternion=np.empty(4)
    mujoco.mju_mat2Quat(quaternion,np.asarray(rotation).ravel())
    scratch.qpos[carrier.oq:carrier.oq+7]=np.r_[position,quaternion]
    mujoco.mj_forward(carrier.model,scratch)
    self_depth=object_depth=0.
    excluded=False
    for c in scratch.contact:
        if c.dist>=0:
            continue
        names=[carrier.model.body(carrier.model.geom_bodyid[g]).name or '' for g in (c.geom1,c.geom2)]
        if c.geom1 in carrier.geoms or c.geom2 in carrier.geoms:
            object_depth=max(object_depth,-float(c.dist))
            other=names[1] if c.geom1 in carrier.geoms else names[0]
            excluded|=not other.startswith('right_') or other=='right_tower'
        elif all(n.startswith('right_') for n in names):
            self_depth=max(self_depth,-float(c.dist))
    return dict(self_penetration_m=self_depth,object_penetration_m=object_depth,excluded_support=excluded,
                hypothetical_only=True)


def propose(env,anchors,condition):
    carrier=env.carrier
    before=paired_signature(carrier)
    ik=ContactKinematics(carrier.model,carrier.data,carrier.qadr[:16],anchors,carrier.codec.q[:16],
        carrier.lower[:16],carrier.upper[:16])
    position,rotation=env.pose()
    world_increment=Rotation.from_matrix(env.goal@rotation.T).as_rotvec()
    previous=ik.initial.copy()
    fits=[]
    for fraction in (.25,.5,.75,1.):
        delta=Rotation.from_rotvec(fraction*world_increment).as_matrix()
        translation=np.array([0,0,fraction*condition['lift_m']])
        points,normals=transformed_anchors(anchors,position,delta,translation)
        fit=ik.fit(points,normals,previous,normal_targets=condition['normal_targets'])
        fit.update(fraction=fraction,hypothetical_geometry=hypothetical_geometry(carrier,ik.data,
            position+translation,delta@rotation))
        fits.append(fit)
        previous=np.asarray(fit['actual_joint_target'])
    if paired_signature(carrier)!=before:
        raise RuntimeError('Contact planning changed live physics/controller/camera/RNG/model state')
    commands=np.asarray([fit['command_target'] for fit in fits])
    if (not np.isfinite(commands).all() or np.any(commands<carrier.lower[:16]-1e-10)
            or np.any(commands>carrier.upper[:16]+1e-10)
            or np.max(np.abs(commands-carrier.codec.q[:16]))>.3+1e-10):
        raise ValueError('IK command proposal is nonfinite or exceeds original bounds')
    return dict(condition=condition,fits=fits,command_knots=commands.tolist(),
                planning_live_state_unchanged=True,planning_signature=before,
                all_optimizers_converged=all(f['success'] for f in fits))


def run_continuation(env,state,knots,expected_signature,role,output,report,save):
    number=len(report['cases'])
    if number>=18:
        raise RuntimeError('Contact experiment budget exhausted')
    report['active_case']=dict(index=number,role=role,**state)
    save()
    obs,initial=env.reset(seed=state['seed'],options=dict(placement_index=state['placement_index'],direction=1))
    signature=paired_signature(env.carrier)
    initial_command=env.carrier.codec.q.copy()
    previous_command=initial_command.copy()
    previous_velocity=env.carrier.codec.velocity.copy()
    start_position,start_rotation=env.pose()
    release_origin=env.carrier.origin.copy()
    start_time=float(env.carrier.data.time)
    path=output/'cases'/f'case-{number:04d}.json'
    row=dict(index=number,role=role,**state,initial=initial,prefix_signature=signature,
        start_position=start_position.tolist(),start_rotation=start_rotation.tolist(),goal_rotation=env.goal.tolist(),
        start_command=initial_command.tolist(),inherited_command_velocity=previous_velocity.tolist(),
        command_knots=np.asarray(knots).tolist(),initial_contacts=contacts(env),samples=[],
        peak_command_velocity_rad_s=0.,peak_command_acceleration_rad_s2=0.)
    write(path,row)
    try:
        if signature!=expected_signature:
            raise RuntimeError('Independently executed prefix differs from planning prefix; no retry')
        info=initial
        if initial['prefix_grasp_success']:
            for tick in range(140):
                target=contact_command(initial_command[:16],knots,(tick+1)*DT)
                action=np.clip((target-env.carrier.codec.q[:16])/(env.config.max_velocity*DT),-1,1)
                input_observation=obs.copy()
                obs,reward,done,truncated,info=env.step(action)
                position,rotation=env.pose()
                command=env.carrier.codec.q.copy()
                velocity=(command[:16]-previous_command[:16])/DT
                row['peak_command_velocity_rad_s']=max(row['peak_command_velocity_rad_s'],float(max(abs(velocity))))
                row['peak_command_acceleration_rad_s2']=max(row['peak_command_acceleration_rad_s2'],
                    float(max(abs(velocity-previous_velocity)))/DT)
                command_ok=bool(np.all(command>=env.carrier.lower-1e-10)
                    and np.all(command<=env.carrier.upper+1e-10)
                    and abs(command[-1]-env.carrier.hand_spec.neutral[-1])<1e-10
                    and row['peak_command_velocity_rad_s']<=.8+1e-9
                    and row['peak_command_acceleration_rad_s2']<=2.+1e-9)
                row['samples'].append(dict(tick=tick+1,observation=input_observation.tolist(),
                    action=action.tolist(),position_target=target.tolist(),command=command.tolist(),
                    command_limits_met=command_ok,position=position.tolist(),rotation=rotation.tolist(),
                    qpos_all=env.carrier.data.qpos.tolist(),qvel_all=env.carrier.data.qvel.tolist(),
                    time_s=float(env.carrier.data.time),contacts=contacts(env),**info))
                previous_command,previous_velocity=command,velocity
                if (tick+1)%20==0:
                    write(path,row)
                if done or truncated or not command_ok:
                    break
        np.testing.assert_array_equal(env.carrier.origin,release_origin)
        complete=bool(initial['prefix_grasp_success'] and len(row['samples'])==140
            and info['termination_reason'] in ('horizon','success') and not info['dropped'])
        duration=float(env.carrier.data.time-start_time)
        if complete and not np.isclose(duration,7.,atol=1e-9,rtol=0):
            raise RuntimeError('Continuation duration differs from the original7s')
        command_ok=all(s['command_limits_met'] for s in row['samples'])
        row.update(end=info,simulated_seconds=duration,full_uncensored_duration=complete,
            command_limits_met=command_ok,native_task_success=bool(complete and command_ok and info['is_success']))
        write(path,row)
    except BaseException as error:
        row['error']=repr(error)
        write(path,row)
        raise
    brief={k:v for k,v in row.items() if k not in ('samples','initial_contacts')}
    brief.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report['cases'].append(brief)
    report['active_case']=None
    report['progress']=dict(cases=len(report['cases']),maximum_cases=18,
        continuations=sum('end' in c for c in report['cases']),maximum_continuations=15)
    save()
    print(json.dumps(dict(case=number,role=role,seed=state['seed'],ticks=len(row['samples']),
        end_error_deg=info['angle_error_deg'],reason=info['termination_reason'],
        native_task_success=row['native_task_success'])),flush=True)
    return brief


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-report',type=Path,required=True)
    parser.add_argument('--trajectory-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan,plan_sha=plan_file()
    parent_raw=args.parent_report.read_bytes()
    trajectory_raw=args.trajectory_report.read_bytes()
    if (hashlib.sha256(parent_raw).hexdigest()!=plan['parent_report_sha256']
            or hashlib.sha256(trajectory_raw).hexdigest()!=plan['trajectory_report_sha256']):
        raise ValueError('Require immutable authority and trajectory reports')
    parent=json.loads(parent_raw)
    trajectory=json.loads(trajectory_raw)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'cases').mkdir()
    report=dict(status='running',timestamp=datetime.now(timezone.utc).isoformat(),plan=plan,plan_sha256=plan_sha,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        acceptance_met=False,continuous_rotation_accepted=False,states=[],cases=[],active_case=None)
    def save():
        report['updated_at']=datetime.now(timezone.utc).isoformat()
        write(args.output/'probe.json',report)
    save()
    env=PenRotateEnv(RotationConfig())
    model=env.carrier.model
    controlled=set(map(int,model.actuator_trnid[env.carrier.actuators,0]))
    hinges={j for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE}
    if controlled!=hinges or len(controlled)!=17 or model.neq!=1:
        raise RuntimeError('Pinned model changed: refuse silently ignoring passive/uncontrolled hinges')
    report['compiled_model']=dict(nq=model.nq,nv=model.nv,nu=model.nu,neq=model.neq,
        controlled_finger_hinges=16,additional_passive_hinges=0,
        joints=[model.joint(j).name for j in range(model.njnt)])
    try:
        for state in plan['states']:
            role='planning-prefix'
            number=len(report['cases'])
            report['active_case']=dict(index=number,role=role,**state)
            save()
            obs,initial=env.reset(seed=state['seed'],options=dict(placement_index=state['placement_index'],direction=1))
            signature=paired_signature(env.carrier)
            prep=dict(index=number,role=role,**state,initial=initial,prefix_signature=signature,
                qpos=env.carrier.data.qpos.tolist(),qvel=env.carrier.data.qvel.tolist(),
                command=env.carrier.codec.q.tolist(),contacts=contacts(env))
            path=args.output/'cases'/f'case-{number:04d}.json'
            write(path,prep)
            report['cases'].append(dict(prep,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if not initial['prefix_grasp_success']:
                raise RuntimeError('Planning prefix failed; preserve it and stop without retry')
            if state['seed']==2511000 and signature!=parent['cases'][0]['prefix_signature']:
                raise RuntimeError('Physical planning prefix differs from immutable parent')
            anchors=anchors_from_prefix(env)
            state_report=dict(**state,prefix_signature=signature,anchors=anchors,proposals=[])
            report['states'].append(state_report)
            for condition in plan['conditions']:
                before=paired_signature(env.carrier)
                try:
                    proposal=propose(env,anchors,condition)
                except ValueError as error:
                    proposal=dict(condition=condition,skipped=True,reason=repr(error))
                if paired_signature(env.carrier)!=before:
                    raise RuntimeError('Failed or completed planner mutated live state')
                state_report['proposals'].append(proposal)
                save()
            initial_command=env.carrier.codec.q[:16].copy()
            baseline=run_continuation(env,state,np.tile(initial_command,(4,1)),signature,
                'zero-offset-reference',args.output,report,save)
            if state['seed']==2511000 and baseline['end']!=trajectory['cases'][0]['end']:
                raise RuntimeError('Zero-offset continuation differs from frozen nonlinear-search reference')
            for proposal in state_report['proposals']:
                role=proposal['condition']['name']
                if proposal.get('skipped'):
                    report['cases'].append(dict(index=len(report['cases']),role=role,**state,
                        skipped=True,reason=proposal['reason'],native_task_success=False))
                    save()
                    continue
                run_continuation(env,state,proposal['command_knots'],signature,role,args.output,report,save)
        assert len(report['cases'])==18
        proposals=[c for c in report['cases'] if c['role'] not in ('planning-prefix','zero-offset-reference')]
        references=[c for c in report['cases'] if c['role']=='zero-offset-reference']
        report.update(status='complete',active_case=None,finished_at=datetime.now(timezone.utc).isoformat(),
            summary=dict(total_cases=18,planning_preparations=3,proposals=12,
                skipped_proposals=sum(c.get('skipped',False) for c in proposals),
                proposal_successes=sum(c['native_task_success'] for c in proposals),
                reference_successes=sum(c['native_task_success'] for c in references),
                all_three_states_and_four_conditions_retained=True,formal_or_continuous_acceptance=False))
        save()
        print(json.dumps(report['summary'],indent=2),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise
    finally:
        env.close()


if __name__=='__main__':
    main()
