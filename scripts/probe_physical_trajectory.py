"""Bounded privileged trajectory search from ordinary settled ORCA resets.

Four nominal cases, two smooth target segments and fixed CEM budgets. Each
candidate restores a whole physical branch state once, preserving velocities.
There is no intermediate reference-state injection or trained/deployable actor.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from orca_feetech.manipulation import DT, ManipulationEnv, TaskConfig, rotation_error, yaw_rotation
from orca_feetech.model import ModelSpec
from orca_feetech import reference_control
from orca_feetech.reference_control import BrakingActionCodec, CONTROLLER_ID
import probe_reference_hold as hold_probe
import probe_reference_retarget as retarget
from probe_reference_replay import command_kinematics


SEEDS=(1970000,1970001,1970002,1970003)
HELDOUT_SEEDS=(1971000,1971001)
POPULATION=32
GENERATIONS=4
ELITES=8
OFFSET_LIMIT=.3
SEGMENT_SECONDS=1.5
DURATION=4.
STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION


def save_report(path,report):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(path)


def branch_state(env):
    state=np.empty(mujoco.mj_stateSize(env.model,STATE_SPEC))
    mujoco.mj_getState(env.model,env.data,state,STATE_SPEC)
    return state


def smooth_target(initial,offsets,t,lower,upper):
    offsets=np.asarray(offsets,float)
    if offsets.shape!=(2,16) or not np.isfinite(offsets).all() or np.max(np.abs(offsets))>OFFSET_LIMIT+1e-12:
        raise ValueError('Expected two finite 16-joint offsets within 0.3 radians')
    knots=np.clip(initial[:16]+offsets,lower[:16],upper[:16])
    if t<=SEGMENT_SECONDS:
        start,end=initial[:16],knots[0]
        phase=t/SEGMENT_SECONDS
    else:
        start,end=knots
        phase=(t-SEGMENT_SECONDS)/SEGMENT_SECONDS
    phase=float(np.clip(phase,0,1))
    blend=phase**3*(10+phase*(-15+6*phase))
    return start+blend*(end-start)


def geometry(env):
    self_depth=max((c['penetration_m'] for c in retarget.hand_self_contacts(env.model,env.data)),default=0.)
    object_depth=0.
    excluded=False
    for c in env.data.contact:
        if c.dist<0 and any(g in (c.geom1,c.geom2) for g in env.geoms):
            object_depth=max(object_depth,-float(c.dist))
            other=c.geom2 if c.geom1 in env.geoms else c.geom1
            name=env.model.body(env.model.geom_bodyid[other]).name or 'world'
            excluded |= not name.startswith('right_') or name=='right_tower'
    return self_depth,object_depth,bool(excluded)


def prepare(env,seed,direction,extra_hold_seconds):
    if extra_hold_seconds not in (0.,5.):
        raise ValueError('Use a fixed zero or five second extra hold')
    _,reset_info=env.reset(seed=seed,options={'direction':direction})
    command=env.codec.q.copy()
    origin,rotation=env.pose()
    initial=hold_probe.snapshot(env,origin,rotation,command)
    samples=[]
    self_peak,object_peak,excluded=geometry(env)
    for _ in range(round(extra_hold_seconds/DT)):
        env.data.ctrl[env.actuators]=command
        mujoco.mj_step(env.model,env.data,nstep=25)
        mujoco.mj_forward(env.model,env.data)
        samples.append(hold_probe.snapshot(env,origin,rotation,command))
        a,b,bad=geometry(env)
        self_peak=max(self_peak,a)
        object_peak=max(object_peak,b)
        excluded |= bad
    return reset_info,command,dict(seconds=extra_hold_seconds,initial=initial,
        end=samples[-1] if samples else initial,samples=samples,
        peak_self_penetration_m=self_peak,peak_object_penetration_m=object_peak,
        excluded_object_support=excluded,velocities_rewritten=False)


def trajectory(env,state,initial_command,offsets,direction,*,fine_collision_audit=False):
    if direction not in (-1,1):
        raise ValueError('Direction must be -1 or 1')
    # Branching a simulator is initialization of an independent candidate,
    # not a reference-state correction within its subsequent physical motion.
    mujoco.mj_setState(env.model,env.data,state,STATE_SPEC)
    # Warning counters are diagnostic metadata, not part of mjSTATE_INTEGRATION.
    # A warning from a different candidate must not taint this fresh branch.
    for warning in env.data.warning:
        warning.number=0
        warning.lastinfo=0
    mujoco.mj_forward(env.model,env.data)
    origin,initial_rotation=env.pose()
    goal=yaw_rotation(np.deg2rad(10)*direction)@initial_rotation
    start_time=float(env.data.time)
    codec=BrakingActionCodec(env.config,initial_command)
    initial=hold_probe.snapshot(env,origin,initial_rotation,initial_command)
    samples=[]
    self_peak,object_peak,excluded=geometry(env)
    numerical_failure=None
    for tick in range(round(DURATION/DT)):
        target=smooth_target(initial_command,offsets,(tick+1)*DT,env.lower,env.upper)
        action=np.clip((target-codec.q[:16])/(env.config.max_velocity*DT),-1,1)
        command=codec.step(action)
        env.data.ctrl[env.actuators]=command
        if fine_collision_audit:
            for _ in range(25):
                mujoco.mj_step(env.model,env.data)
                # Contacts from mj_step are evaluated at its pre-integration
                # state. Reading them does not run another forward solve.
                a,b,bad=geometry(env)
                self_peak=max(self_peak,a)
                object_peak=max(object_peak,b)
                excluded |= bad
        else:
            mujoco.mj_step(env.model,env.data,nstep=25)
        mujoco.mj_forward(env.model,env.data)
        a,b,bad=geometry(env)
        self_peak=max(self_peak,a)
        object_peak=max(object_peak,b)
        excluded |= bad
        if (not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all()
                or abs(env.data.time-start_time-(tick+1)*DT)>1e-8):
            numerical_failure='Non-finite state or solver time reset'
            break
        sample=hold_probe.snapshot(env,origin,initial_rotation,command)
        _,rotation=env.pose()
        relative=rotation@initial_rotation.T
        sample.update(command=command.tolist(),
            target_rotation_error_deg=float(np.rad2deg(rotation_error(rotation,goal))),
            signed_yaw_progress_deg=float(direction*np.rad2deg(np.arctan2(relative[1,0],relative[0,0]))))
        samples.append(sample)
        if sample['object_position_m'][2]<.10 or sample['displacement_m']>.12 or excluded:
            break
    warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(env.data.warning) if w.number}
    if warnings:
        numerical_failure=numerical_failure or 'MuJoCo solver warning'
    effective=command_kinematics(initial_command,samples,env.config)
    command_ok=bool(effective['peak_velocity_rad_s']<=.8+1e-8 and effective['peak_acceleration_rad_s2']<=2+1e-8)
    complete=len(samples)==round(DURATION/DT)
    feasible=bool(complete and not numerical_failure and not excluded and command_ok
                  and self_peak<=.002 and object_peak<=.002)
    final=samples[-10:] if samples else [dict(target_rotation_error_deg=180,displacement_m=1.,
        linear_speed_mps=1.,angular_speed_rps=100.,retained=False)]
    physical_cost=float(np.mean([(s['target_rotation_error_deg']/10)**2+(s['displacement_m']/.04)**2
        +.05*(s['linear_speed_mps']/.08)**2+.05*(s['angular_speed_rps']/1.)**2
        +25*(not s['retained']) for s in final]))
    cost=physical_cost+.001*float(np.mean((np.asarray(offsets)/OFFSET_LIMIT)**2))
    if not feasible:
        cost+=10000+100*(max(self_peak,object_peak)/.002)**2
    preserved=bool(feasible and len(final)==10 and all(s['resting'] and s['target_rotation_error_deg']<2
        and s['signed_yaw_progress_deg']>=8 for s in final))
    return dict(score=cost,physical_cost=physical_cost,initial=initial,end=samples[-1] if samples else initial,
        samples=samples,simulated_seconds=float(env.data.time-start_time),
        complete=complete,numerical_failure=numerical_failure,warnings=warnings,
        feasible_diagnostic=feasible,diagnostic_motion_preserved=preserved,acceptance_met=False,
        collision_sample_interval_s=.002 if fine_collision_audit else DT,
        peak_self_penetration_m=self_peak,peak_object_penetration_m=object_peak,
        excluded_object_support=excluded,effective_command_kinematics=effective,command_limits_met=command_ok)


def brief(run):
    return {k:v for k,v in run.items() if k!='samples'}


def search_case(case_index,output,extra_hold_seconds=0.):
    seed=SEEDS[case_index]
    direction=1 if case_index%2==0 else -1
    env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
    path=Path(output)/f'case-{case_index}.json'
    report=dict(status='running',seed=seed,direction=direction,goal_angle_deg=10,
                search_candidates=[],generations=[])
    try:
        reset_info,command,preparation=prepare(env,seed,direction,extra_hold_seconds)
        state=branch_state(env)
        report.update(reset_info=reset_info,preparation=preparation,initial_state=state.tolist(),initial_command_rad=command.tolist(),
            initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest())
        zero=np.zeros((2,16))
        baseline=trajectory(env,state,command,zero,direction)
        best=zero.copy()
        best_run=baseline
        report.update(baseline=brief(baseline),baseline_score=baseline['score'])
        mean=np.zeros(32)
        sigma=np.full(32,.10)
        rng=np.random.default_rng(seed)
        started=time.monotonic()
        for generation in range(GENERATIONS):
            candidates=np.clip(rng.normal(mean,sigma,(POPULATION,32)),-OFFSET_LIMIT,OFFSET_LIMIT)
            candidates[0]=best.ravel()
            scores=[]
            for index,candidate in enumerate(candidates):
                run=trajectory(env,state,command,candidate.reshape(2,16),direction)
                scores.append(run['score'])
                report['search_candidates'].append(dict(generation=generation,index=index,
                    offsets_rad=candidate.reshape(2,16).tolist(),result=brief(run)))
                if run['score']<best_run['score']:
                    best,best_run=candidate.reshape(2,16).copy(),run
            elite=candidates[np.argsort(scores,kind='stable')[:ELITES]]
            mean=elite.mean(axis=0)
            sigma=np.maximum(elite.std(axis=0),.015)
            report['generations'].append(dict(generation=generation,best_score=best_run['score'],
                best_final_error_deg=best_run['end'].get('target_rotation_error_deg'),
                feasible_candidates=sum(r['result']['feasible_diagnostic'] for r in report['search_candidates'][-POPULATION:])))
            save_report(path,report)
            print(json.dumps(dict(case=case_index,**report['generations'][-1])),flush=True)
        # Selection is frozen before finer collision audit and held-out reset
        # transfer. Those results never choose or update the plan.
        report.update(selected_offsets_rad=best.tolist(),selected_search_result=brief(best_run),
            baseline_validation=trajectory(env,state,command,zero,direction,fine_collision_audit=True),
            selected_validation=trajectory(env,state,command,best,direction,fine_collision_audit=True),
            elapsed_search_and_validation_s=time.monotonic()-started,
            total_search_rollouts=1+POPULATION*GENERATIONS,status='complete')
        save_report(path,report)
        return report
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save_report(path,report)
        raise
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extra-hold-seconds',type=float,choices=[0.,5.],default=0.,
                   help='Fixed physical hold after ordinary reset, before defining this diagnostic target')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(timestamp=datetime.now(timezone.utc).isoformat(),status='running',acceptance_met=False,
        purpose='Privileged nominal trajectory feasibility diagnostic; no trained policy, formal test or deployment',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        collision_script_sha256=hashlib.sha256(Path(retarget.__file__).read_bytes()).hexdigest(),
        hold_script_sha256=hashlib.sha256(Path(hold_probe.__file__).read_bytes()).hexdigest(),
        controller_module_sha256=hashlib.sha256(Path(reference_control.__file__).read_bytes()).hexdigest(),
        model_sha256=ModelSpec.load().source_sha256,
        geometry_sha256=hold_probe.geometry_hash(),controller=CONTROLLER_ID,
        runtime=dict(system=platform.system(),python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__),
        protocol=dict(search_seeds=list(SEEDS),heldout_reset_seeds=list(HELDOUT_SEEDS),workers=4,
            extra_hold_seconds=args.extra_hold_seconds,
            population=POPULATION,generations=GENERATIONS,elites=ELITES,
            maximum_search_rollouts=4*(1+POPULATION*GENERATIONS),maximum_fine_validation_rollouts=12,
            seconds_per_rollout=DURATION,position_offset_limit_rad=OFFSET_LIMIT,
            segments=2,seconds_per_segment=SEGMENT_SECONDS,final_target_hold_seconds=1.,
            parameterization='Two quintic smoothstep position targets, clipped knots within original ROM',
            selection='Fixed CEM objective; finer audits and held-out reset replay never choose the plan',
            objective='Last 0.5s mean squared SO3/10deg, displacement/4cm; 0.05 squared speed/0.08mps and angularspeed/1rps; 25 if unsupported; 0.001 normalized offset energy; infeasible +10000+100*(peakdepth/2mm)^2',
            command_velocity_limit_rad_s=.8,command_acceleration_limit_rad_s2=2.,
            collision_threshold_m=.002,diagnostic_preservation='Last 0.5s: full SO3 error <2deg, signed yaw >=8deg, physical reset rest thresholds and retained; whole rollout feasible',
            initial_state='Whole mjSTATE_INTEGRATION from ordinary physical settle, velocities preserved',
            state_injections_during_rollout=0,randomization=0.,vision_delay_and_noise=False),
        limitations=['Four optimized nominal cases and two held-out resets are not a policy success rate.',
            'The optimizer reads privileged simulation truth and branches the exact model.',
            'Search collisions sampled at 20Hz; selected/baseline and held-out replays audit every 2ms.',
            'No learned actor, normal-reset asset, real actuator/vision validation or continuous-rotation acceptance.'],
        cases=[],heldout_replays=[])
    save_report(args.output/'probe.json',report)
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(search_case,i,str(args.output),args.extra_hold_seconds):i for i in range(4)}
            for future in as_completed(futures):
                report['cases'].append(future.result())
                report['cases'].sort(key=lambda r:r['seed'])
                save_report(args.output/'probe.json',report)
        env=ManipulationEnv(TaskConfig(object='pen',angle=10,randomization=0))
        try:
            for index,seed in enumerate(HELDOUT_SEEDS):
                direction=1 if index==0 else -1
                info,command,preparation=prepare(env,seed,direction,args.extra_hold_seconds)
                state=branch_state(env)
                for case in report['cases']:
                    if case['direction']==direction:
                        run=trajectory(env,state,command,case['selected_offsets_rad'],direction,fine_collision_audit=True)
                        report['heldout_replays'].append(dict(seed=seed,source_seed=case['seed'],reset_info=info,
                                                              preparation=preparation,run=run))
                        save_report(args.output/'probe.json',report)
        finally:
            env.close()
        report['summary']=dict(search_cases=4,
            baseline_preserved=sum(c['baseline_validation']['diagnostic_motion_preserved'] for c in report['cases']),
            selected_preserved=sum(c['selected_validation']['diagnostic_motion_preserved'] for c in report['cases']),
            selected_feasible=sum(c['selected_validation']['feasible_diagnostic'] for c in report['cases']),
            selected_mean_final_error_deg=float(np.mean([c['selected_validation']['end']['target_rotation_error_deg'] for c in report['cases']])),
            heldout_preserved=sum(r['run']['diagnostic_motion_preserved'] for r in report['heldout_replays']),
            heldout_replays=len(report['heldout_replays']),heldout_distinct_resets=2,
            all_fine_command_limits_met=all(r['command_limits_met'] for r in
                [*[c[mode] for c in report['cases'] for mode in ('baseline_validation','selected_validation')],
                 *[h['run'] for h in report['heldout_replays']]]))
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        save_report(args.output/'probe.json',report)
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
