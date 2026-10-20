"""Training-only feedback audit of previously rescued initial states.

The expert residual is an oracle lookup selected by earlier simulation search,
not an available deployment input. Interpolation results cannot qualify a
policy or count as an independent success rate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import ManipulationEnv
from orca_feetech.policy_imitation import load_demonstrations
from probe_action_residuals import rollout


class FeedbackBlend:
    def __init__(self, source, candidate, residual, fraction):
        if fraction not in (0., .25, .5, .75, 1.):
            raise ValueError("Use a frozen feedback fraction")
        self.source, self.candidate = source, candidate
        self.residual, self.fraction = np.asarray(residual,dtype=np.float32), fraction

    def predict(self, obs, deterministic=True):
        if not deterministic:
            raise ValueError("Diagnostic rollouts are deterministic")
        if self.fraction == 0:
            return self.candidate.predict(obs,deterministic=True)
        expert = np.clip(self.source.predict(obs,deterministic=True)[0]+self.residual,-1,1)
        if self.fraction == 1:
            return expert, None
        candidate = self.candidate.predict(obs,deterministic=True)[0]
        return ((1-self.fraction)*candidate+self.fraction*expert).astype(np.float32), None


def initialize(source, candidate):
    import torch
    from stable_baselines3 import PPO
    global SOURCE, CANDIDATE, ENV
    torch.set_num_threads(1)
    SOURCE, CANDIDATE = PPO.load(source,device="cpu"), PPO.load(candidate,device="cpu")
    ENV = ManipulationEnv(load_config(Path(source))[0])


def audit_trial(record):
    rows = []
    for fraction in (0., .25, .5, .75, 1.):
        policy = FeedbackBlend(SOURCE,CANDIDATE,record["residual"],fraction)
        result, _, _ = rollout(ENV,policy,record["seed"],np.zeros(16,np.float32))
        if result["initial"] != record["initial"]:
            raise ValueError("Feedback audit changed the frozen initial state/goal")
        result["expert_fraction"] = fraction
        rows.append(result)
    return dict(seed=record["seed"],expert_residual=record["residual"],feedback=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--reports",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    config, meta = load_config(args.source)
    candidate_config, candidate_meta = load_config(args.candidate)
    if config.fingerprint() != candidate_config.fingerprint():
        parser.error("Source and candidate must share the exact canonical task")
    _, data_proof = load_demonstrations(args.reports,args.source,meta["fingerprint"])
    proof = candidate_meta.get("offline_imitation",{})
    if (proof.get("source_sha256") != hashlib.sha256(args.source.read_bytes()).hexdigest()
            or [p["sha256"] for p in proof.get("datasets",[])] != [p["sha256"] for p in data_proof]):
        parser.error("Candidate did not use this exact frozen training dataset/source")
    records = []
    for path in args.reports:
        for trial in json.loads(path.read_text())["trials"]:
            records.extend(candidate for candidate in trial["candidates"] if candidate["success"])
    if not 1 <= len(records) <= 64 or len({r["seed"] for r in records}) != len(records):
        parser.error("Use1..64 distinct previously rescued training initials")
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),status="running",
        purpose="Training-only oracle feedback sensitivity; no selection, learning or formal acceptance",
        is_formal_evaluation=False,acceptance_met=False,source=str(args.source.resolve()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        candidate=str(args.candidate.resolve()),candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        rescued_initials=len(records),fractions=[0.,.25,.5,.75,1.],maximum_rollouts=len(records)*5,
        maximum_control_steps=len(records)*5*config.horizon,trials=[])
    def save():
        temporary = args.output/"audit.tmp"
        temporary.write_text(json.dumps(report,indent=2)+"\n")
        temporary.replace(args.output/"audit.json")
    save()
    with ProcessPoolExecutor(max_workers=4,initializer=initialize,
                             initargs=(str(args.source),str(args.candidate))) as pool:
        futures = [pool.submit(audit_trial,r) for r in records]
        for future in as_completed(futures):
            report["trials"].append(future.result())
            report["trials"].sort(key=lambda t:t["seed"])
            save()
    report["summary"] = [dict(expert_fraction=fraction,successes=sum(
        row["success"] for trial in report["trials"] for row in trial["feedback"]
        if row["expert_fraction"]==fraction),episodes=len(records),drops=sum(
        row["dropped"] for trial in report["trials"] for row in trial["feedback"]
        if row["expert_fraction"]==fraction)) for fraction in report["fractions"]]
    report["oracle_replay_valid"] = report["summary"][-1]["successes"] == len(records)
    report.update(status="complete",ended_at=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps({k:v for k,v in report.items() if k!="trials"},indent=2))


if __name__ == "__main__":
    main()
