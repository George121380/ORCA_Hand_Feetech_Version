"""One CUDA actor-only2000-update pilot. Zero additional physics/PPO steps."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import torch
from stable_baselines3 import PPO

from manipulation import write,fresh_ppo
from orca_feetech.manipulation import FRAME_SIZE,HISTORY,limits
from orca_feetech.model import JOINT_NAMES
from orca_feetech.objects import specification
from orca_feetech.paths import provenance
from orca_feetech.pen_rotate import (PenRotateEnv,RotationConfig,TASK_ID_ROTATE,
    PREFIX_SHA256,validate_grasp_prefix,rotation_contract,load_rotation_config)
from orca_feetech.policy_imitation import censored_action_loss
from orca_feetech.recorded_rotation_audit import audit_recorded_rotation
from orca_feetech.rotation_demonstrations import demonstration_admission,eligible_demo
from orca_feetech.training_supervision import checkpoint_steps


ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_pairs(report_path,audit_path):
    report=json.loads(report_path.read_text())
    audit=json.loads(audit_path.read_text())
    if (report.get('status')!='complete' or report.get('kind')!='rotation_training_demonstrations_v1'
            or report['plan_sha256']!='39a2ce2ab1b0f768f50396cc09bd7a8f0cd57f6764ccf8814721dd2a7b8d8aad'
            or report['is_formal_evaluation'] is not False or audit.get('status')!='complete'
            or audit.get('report_sha256')!=digest(report_path) or audit.get('allocations_verified')!=112
            or audit.get('additional_physics_steps')!=0
            or not demonstration_admission(report['cases'])['imitation_admitted']):
        raise ValueError('Require the frozen admitted training corpus and matching complete native audit')
    lower,upper=limits(RotationConfig())
    groups={}
    proof=[]
    for case in report['cases']:
        if case.get('skipped'):
            continue
        path=report_path.parent/'cases'/f"case-{case['index']:04d}.json"
        if digest(path)!=case['sha256']:
            raise ValueError('Changed successful or failed case file')
        if not eligible_demo(case):
            continue
        row=json.loads(path.read_text())
        checked=audit_recorded_rotation(row,lower,upper)
        if not checked['native_task_success'] or checked['observable_action_pairs']!=140:
            raise ValueError('Ineligible raw demonstration')
        key=(case['seed'],case['placement_index'],case['direction'])
        groups.setdefault(key,[]).append((np.asarray([s['observation'] for s in row['samples']],np.float32),
                                         np.asarray([s['action'] for s in row['samples']],np.float32)))
        proof.append(dict(index=case['index'],seed=case['seed'],direction=case['direction'],sha256=digest(path)))
    arrays={k:(np.concatenate([p[0] for p in v]),np.concatenate([p[1] for p in v])) for k,v in groups.items()}
    return arrays,dict(report_sha256=digest(report_path),audit_sha256=digest(audit_path),
                       admission=demonstration_admission(report['cases']),trajectories=proof)


def fit_actor(model,groups):
    policy=model.policy
    parameters=dict(policy.named_parameters())
    actor={n:p for n,p in parameters.items() if n.startswith(('mlp_extractor.policy_net.','action_net.'))}
    if len(actor)!=6 or model.observation_space.shape!=(296,) or model.action_space.shape!=(16,):
        raise ValueError('Unexpected actor architecture/inputs')
    original_flags={n:p.requires_grad for n,p in parameters.items()}
    frozen={n:p.detach().clone() for n,p in parameters.items() if n not in actor}
    for n,p in parameters.items():
        p.requires_grad_(n in actor)
    values=[(torch.as_tensor(o,device=model.device),torch.as_tensor(a,device=model.device)) for o,a in groups.values()]
    lengths=np.asarray([len(o) for o,a in values])
    offsets=np.r_[0,np.cumsum(lengths)[:-1]]
    all_observations=torch.cat([o for o,a in values])
    all_actions=torch.cat([a for o,a in values])
    policy.set_training_mode(False)
    rng=np.random.default_rng(0)
    torch.manual_seed(0)
    optimizer=torch.optim.Adam(actor.values(),lr=1e-4,eps=1e-5)
    def loss(o,a):
        return censored_action_loss(policy.get_distribution(o).distribution.mean,a)
    def metrics():
        with torch.no_grad():
            return [float(torch.stack([loss(o[i:i+256],a[i:i+256])*len(o[i:i+256])
                for i in range(0,len(o),256)]).sum()/len(o)) for o,a in values]
    try:
        before=metrics()
        for _ in range(2000):
            states=rng.integers(len(values),size=256)
            indices=torch.as_tensor(offsets[states]+rng.integers(0,lengths[states]),device=model.device)
            error=loss(all_observations[indices],all_actions[indices])
            if not torch.isfinite(error):
                raise ValueError('Nonfinite actor imitation loss')
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.values()),.5,error_if_nonfinite=True)
            optimizer.step()
        if not all(torch.equal(parameters[n],v) for n,v in frozen.items()):
            raise RuntimeError('Imitation changed frozen critic/log_std')
        if (model.num_timesteps!=0 or model._n_updates!=0 or policy.optimizer.state
                or not all(torch.isfinite(p).all() for p in parameters.values())):
            raise RuntimeError('Nonfinite checkpoint or nonzero PPO steps/updates/optimizer history')
        return dict(updates=2000,batch_size=256,learning_rate=1e-4,seed=0,loss_before=before,
            loss_after=metrics(),state_keys=list(groups),samples_by_state=[len(o) for o,a in values],
            critic_and_log_std_bitwise_unchanged=True,ppo_optimizer_empty=True,
            additional_environment_steps=0,ppo_updates=0)
    finally:
        for n,p in parameters.items():
            p.requires_grad_(original_flags[n])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    plan_path=ROOT/'configs/training/pen-rotate-imitation-v1.json'
    plan=json.loads(plan_path.read_text())
    if ((plan['updates'],plan['batch_size'],plan['learning_rate'],plan['seed'],plan['candidates'],plan['extensions'])
            !=(2000,256,1e-4,0,1,0)):
        raise ValueError('Changed finite imitation protocol')
    groups,data_proof=load_training_pairs(args.report,args.audit)
    validate_grasp_prefix(args.source)
    torch.set_num_threads(1)
    check=torch.ones((8,8),device='cuda')
    if float((check@check).sum())!=512.:
        raise RuntimeError('CUDA matrix execution failed')
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=RotationConfig()
    meta=dict(task_id=TASK_ID_ROTATE,task=asdict(cfg),fingerprint=cfg.fingerprint(),
        task_contract=rotation_contract(),joint_names=JOINT_NAMES,history=HISTORY,frame_size=FRAME_SIZE,
        upstream=provenance(),geometry=specification('pen'),seed=0,num_envs=1,device='cuda',
        actual_steps=0,requested_steps=0,trained=False,
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        actor_initialization=dict(source_task='orca-v1-pen-grasp-v1',source_sha256=PREFIX_SHA256,
            source_steps=200704,copy='All tensors into fresh rotation container; then actor-only BC',
            training_report_sha256=data_proof['report_sha256'],imitation_plan_sha256=digest(plan_path)))
    env=PenRotateEnv(cfg)
    try:
        model=fresh_ppo(env,SimpleNamespace(seed=0,device='cuda',num_envs=1,learning_rate=1e-4,
            entropy_coef=.01,target_kl=.03,initial_log_std=-4.,output=args.output))
        source=PPO.load(args.source,device='cpu')
        model.policy.load_state_dict(source.policy.state_dict(),strict=True)
        if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
            raise RuntimeError('Expected a fresh zero-step target optimizer')
        initial=args.output/'initial'
        initial.mkdir()
        model.save(initial/'policy.zip')
        write(initial/'config.json',dict(meta,artifact_role='before_bc'))
        proof=fit_actor(model,groups)
        proof.update(status='complete',plan=plan,plan_sha256=digest(plan_path),datasets=data_proof,
            source_sha256=PREFIX_SHA256,timestamp=datetime.now(timezone.utc).isoformat(),
            cuda_device=torch.cuda.get_device_name(0),torch_version=torch.__version__,
            is_formal_evaluation=False,acceptance_met=False,continuous_rotation_accepted=False)
        model.save(args.output/'policy.zip')
        meta.update(trained=True,training_kind='offline_actor_imitation_only',offline_imitation=proof)
        write(args.output/'config.json',meta)
        load_rotation_config(args.output/'policy.zip')
        if checkpoint_steps(args.output/'policy.zip',check_weights=True)!=0:
            raise RuntimeError('Offline imitation acquired an incorrect environment step count')
        proof['checkpoint_sha256']=digest(args.output/'policy.zip')
        write(args.output/'imitation.json',proof)
        print(json.dumps({k:v for k,v in proof.items() if k!='datasets'},indent=2),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
