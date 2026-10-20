"""Fixed 28-case zero-action baseline for the separately versioned grasp task."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from orca_feetech.pen_grasp import PenGraspEnv,GraspConfig,grasp_contract,ASSET_SHA256
from manipulation import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--randomization',type=float,choices=[0.,.3],default=0.)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=GraspConfig(randomization=args.randomization)
    first_seed=2110000 if args.randomization==0 else 2111000
    report=dict(status='running',timestamp=datetime.now(timezone.utc).isoformat(),acceptance_met=False,
        fingerprint=config.fingerprint(),task_contract=grasp_contract(),asset_sha256=ASSET_SHA256,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(system=platform.system(),python=platform.python_version(),mujoco=mujoco.__version__),
        protocol=dict(seeds=list(range(first_seed,first_seed+28)),placement='episode%7;4 per bucket',
            randomization=args.randomization,baseline='zero actions; all fixed placement buckets with continuous release jitter',
            maximum_episodes=28,no_initial_state_retries=True),cases=[])
    def save():
        write(args.output/'probe.json',report)
    save()
    env=PenGraspEnv(config)
    try:
        for episode in range(28):
            _,initial=env.reset(seed=first_seed+episode,options={'placement_index':episode%7})
            record=dict(seed=first_seed+episode,initial=initial,initial_object_rotation=env.pose()[1].tolist(),
                reached_hand_qpos_rad=env.reached_hand_qpos.tolist(),reached_hand_qvel_rad_s=env.reached_hand_qvel.tolist(),samples=[])
            report['cases'].append(record)
            for tick in range(140):
                _,reward,done,truncated,info=env.step(np.zeros(16))
                record['samples'].append(dict(tick=tick+1,reward=reward,command=env.codec.q.tolist(),**info))
                if done or truncated:
                    break
            record['end']=info
            save()
        report['summary']=dict(episodes=28,successes=sum(r['end']['is_success'] for r in report['cases']),
            drops=sum(r['end']['dropped'] for r in report['cases']),
            invalid_initializations=sum(r['initial']['reset_failure'] is not None for r in report['cases']),
            by_placement=[dict(index=i,episodes=4,successes=sum(r['end']['is_success'] for r in report['cases']
                if r['initial']['placement_index']==i)) for i in range(7)])
        report.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(report['summary'],indent=2))
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise
    finally:
        env.close()


if __name__=='__main__':
    main()
