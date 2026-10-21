"""Verify all demonstration allocations, hashes, native labels and deployable inputs."""
import argparse
import hashlib
import json
from pathlib import Path

from manipulation import write
from orca_feetech.manipulation import limits
from orca_feetech.pen_rotate import RotationConfig
from orca_feetech.recorded_rotation_audit import audit_recorded_rotation
from orca_feetech.rotation_demonstrations import training_states,demonstration_admission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    if (report['status']!='complete' or report['kind']!='rotation_training_demonstrations_v1'
            or report['is_formal_evaluation'] is not False or len(report['cases'])!=112
            or report['plan']['rotation_fingerprint']!=RotationConfig().fingerprint()):
        raise ValueError('Require a complete training-only rotation demonstration report')
    expected=[(s,role) for s in training_states() for role in (
        'planning-prefix','middle-abd-delta-+0.00','middle-abd-delta--0.08','middle-abd-delta-+0.08')]
    lower,upper=limits(RotationConfig())
    results=[]
    signature=None
    for index,(case,(state,role)) in enumerate(zip(report['cases'],expected)):
        if case['index']!=index or case['role']!=role or any(case[k]!=v for k,v in state.items()):
            raise ValueError('Missing/replaced/reordered seed or variant')
        if case.get('skipped'):
            if role=='planning-prefix' or case.get('native_task_success') or not case['reason']:
                raise ValueError('Invalid skipped allocation')
            continue
        path=args.report.parent/'cases'/f'case-{index:04d}.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=case['sha256']:
            raise ValueError('Recorded case hash mismatch')
        row=json.loads(path.read_text())
        if role=='planning-prefix':
            signature=row['prefix_signature']
            continue
        if row['prefix_signature']!=signature:
            raise ValueError('Continuation prefix differs from independently executed planning prefix')
        audit=audit_recorded_rotation(row,lower,upper)
        if audit['native_task_success']!=case['native_task_success']:
            raise ValueError('Report does not match raw success')
        results.append(dict(index=index,**state,role=role,**audit))
    admission=demonstration_admission(report['cases'])
    if any(report['summary'][k]!=v for k,v in admission.items()):
        raise ValueError('Dataset admission differs from fixed distinct-state gate')
    proof=dict(status='complete',report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        allocations_verified=112,continuations_verified=len(results),
        observable_action_pairs=sum(r['observable_action_pairs'] for r in results),
        admission=admission,cases=results,additional_physics_steps=0,
        limitation='Recomputes20Hz windows and checks recorded cumulative2ms peaks; no new simulation or hardware')
    if args.output.exists():
        raise FileExistsError(args.output)
    write(args.output,proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='cases'},indent=2))


if __name__=='__main__':
    main()
