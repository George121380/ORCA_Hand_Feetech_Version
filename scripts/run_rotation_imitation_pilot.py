"""One audited BC fit and two28-state development evaluations; no retries/PPO."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from manipulation import write


ROOT=Path(__file__).resolve().parents[1]


def development_comparison(source,candidate):
    a,b=source['trials'],candidate['trials']
    if len(a)!=28 or len(b)!=28:
        raise ValueError('Expected all28 reserved development trials')
    for i,(x,y) in enumerate(zip(a,b)):
        expected=dict(seed=3211000+i,placement_index=i%7,direction=-1 if (i//7)%2==0 else 1)
        if (any(x[k]!=v or y[k]!=v for k,v in expected.items())
                or any(x[k]!=y[k] for k in ('initial_position','initial_rotation','goal_rotation',
                                             'prefix_grasp_success','prefix_steps','prefix_end_reason'))):
            raise ValueError('Development trials do not have matched original physical prefixes/goals')
    old=sum(x['is_success'] for x in a)
    new=sum(y['is_success'] for y in b)
    drops=sum(y['dropped'] for y in b)
    directions={str(d):sum(y['is_success'] and y['direction']==d for y in b) for d in (-1,1)}
    return dict(source_successes=old,candidate_successes=new,success_gain=new-old,candidate_drops=drops,
        successes_per_direction=directions,bounded_ppo_admitted=bool(new>=4 and new-old>=3
            and drops==0 and min(directions.values())>0),
        gained=sum(not x['is_success'] and y['is_success'] for x,y in zip(a,b)),
        lost=sum(x['is_success'] and not y['is_success'] for x,y in zip(a,b)),
        is_formal_evaluation=False,acceptance_met=False,continuous_rotation_accepted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    plan_path=ROOT/'configs/training/pen-rotate-imitation-v1.json'
    plan=json.loads(plan_path.read_text())
    if (plan['development']!=dict(seed=3211000,episodes=28,checkpoints=['before_bc','after_bc'])
            or plan['bounded_ppo_admission']!=dict(minimum_candidate_successes=4,minimum_success_gain=3,
                maximum_drops=0,both_directions_successful=True)):
        raise ValueError('Changed development allocation/admission protocol')
    job=dict(status='running',phase='initializing',plan=plan,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        started_at=datetime.now(timezone.utc).isoformat(),commands=[],is_formal_evaluation=False,
        acceptance_met=False,continuous_rotation_accepted=False)
    def save():
        job['updated_at']=datetime.now(timezone.utc).isoformat()
        write(args.output/'pilot.json',job)
    def run(phase,argv,timeout):
        job['phase']=phase
        command=dict(phase=phase,argv=argv,status='running')
        job['commands'].append(command)
        save()
        with (args.output/(phase+'.log')).open('x') as log:
            with subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT) as process:
                command['pid']=process.pid
                save()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError(f'Fixed{phase} wall-clock budget exhausted; no retry') from None
                command.update(returncode=process.returncode,status='complete' if process.returncode==0 else 'failed')
                save()
                if process.returncode:
                    raise RuntimeError(f'{phase} exited{process.returncode}; preserve log and stop')
    save()
    try:
        model=args.output/'model'
        run('imitation',[sys.executable,str(ROOT/'scripts/imitate_rotation_demonstrations.py'),
            '--report',str(args.report.resolve()),'--audit',str(args.audit.resolve()),
            '--source',str(args.source.resolve()),'--output',str(model)],180)
        for label,checkpoint in (('development-before',model/'initial/policy.zip'),
                                 ('development-after',model/'policy.zip')):
            run(label,[sys.executable,str(ROOT/'scripts/pen_rotate.py'),'evaluate','--checkpoint',str(checkpoint),
                '--output',str(args.output/label),'--seed','3211000','--episodes','28'],900)
        job['comparison']=development_comparison(
            json.loads((args.output/'development-before/metrics.json').read_text()),
            json.loads((args.output/'development-after/metrics.json').read_text()))
        job.update(status='complete',phase='complete',finished_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(job['comparison'],indent=2),flush=True)
    except BaseException as error:
        job.update(status='failed',error=repr(error))
        save()
        raise


if __name__=='__main__':
    main()
