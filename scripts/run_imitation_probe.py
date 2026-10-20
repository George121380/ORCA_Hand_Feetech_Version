"""Resumable bounded imitation experiment, using the existing watchdog service.

Training-only collection and one fixed offline candidate, then a paired fresh
development comparison. Never consumes a formal test or marks acceptance.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from orca_feetech.checkpoint import load_config
from orca_feetech.training_supervision import checkpoint_steps
from run_training_job import ROOT, archive_partial, now, save, sha256
from select_checkpoints import cached_metrics


def continuation_gate(source, candidate):
    a, b = source["trials"], candidate["trials"]
    if (len(a) != 200 or len(b) != 200
            or [t["seed"] for t in a] != list(range(1020000,1020200))
            or any(x["seed"] != y["seed"] or x["direction"] != y["direction"]
                   or x["initial_position"] != y["initial_position"]
                   or x["initial_rotation"] != y["initial_rotation"]
                   or x["goal_rotation"] != y["goal_rotation"] for x,y in zip(a,b))):
        raise ValueError("Development comparison is not paired on the frozen fresh seeds")
    old, new = sum(t["is_success"] for t in a), sum(t["is_success"] for t in b)
    return dict(source_successes=old, candidate_successes=new,
                gained=sum(not x["is_success"] and y["is_success"] for x,y in zip(a,b)),
                lost=sum(x["is_success"] and not y["is_success"] for x,y in zip(a,b)),
                candidate_drops=sum(t["dropped"] for t in b),
                continuation_recommended=new >= 150 and new-old >= 10 and sum(t["dropped"] for t in b) <= 2,
                is_formal_evaluation=False, acceptance_met=False)


class Experiment:
    def __init__(self, source, prior_report, state_dir):
        self.source, self.prior = Path(source).resolve(), Path(prior_report).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.output = self.state_dir/"job"
        self.started = time.monotonic()
        _, meta = load_config(self.source)
        self.protocol = dict(kind="imitation_probe", source=str(self.source), source_sha256=sha256(self.source),
            config_sha256=sha256(self.source.parent/"config.json"), fingerprint=meta["fingerprint"],
            prior_report=str(self.prior), prior_report_sha256=sha256(self.prior),
            collection=dict(seed=32,episodes=128,candidates=16,workers=4,timeout=600),
            imitation=dict(updates=512,batch_size=256,learning_rate=3.75e-5,timeout=180),
            development=dict(seed=1020000,episodes=200,timeout=180),
            continuation=dict(minimum_successes=150,minimum_gain=10,maximum_drops=2),
            formal_test=None, maximum_attempts=3)

    def persist(self):
        save(self.output/"job.json", self.job)
        status = ("needs_review" if self.job["status"] == "failed" else
                  "needs_review" if self.job.get("comparison",{}).get("continuation_recommended") else
                  "quality_gate_failed" if self.job["status"] == "complete" else "monitoring")
        save(self.state_dir/"watchdog.json", dict(kind="imitation_probe",status=status,
             active_job=str(self.output),stage_name="35-imitation-development",supervisor_pid=os.getpid(),
             stage_results=[],updated_at=now(),restarts=self.job["attempts"]-1,error=self.job.get("error")))
        save(self.output/"model/progress.json", dict(total_steps=0,elapsed_s=time.monotonic()-self.started,
             evaluations_completed=self.job.get("evaluations_completed",0),evaluations_total=2,candidates_total=1))

    def command(self, name, argv, timeout):
        self.job["phase"] = name
        with (self.output/(name+".log")).open("a") as log:
            process = subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry = dict(phase=name,argv=argv,pid=process.pid,started_at=now(),status="running")
            self.job["commands"].append(entry)
            self.persist()
            deadline = time.monotonic()+timeout
            try:
                while True:
                    try:
                        code = process.wait(timeout=min(10,max(.01,deadline-time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        self.persist()
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f"{name} exceeded its fixed{timeout}s budget") from None
            except BaseException:
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                entry.update(status="failed",ended_at=now())
                raise
            entry.update(status="complete" if code == 0 else "failed",returncode=code,ended_at=now())
            self.persist()
            if code:
                raise RuntimeError(f"{name} exited{code}; inspect its preserved log")

    def evaluate(self, name, checkpoint):
        candidate = dict(checkpoint=str(checkpoint),sha256=sha256(checkpoint),
                         config_sha256=sha256(checkpoint.parent/"config.json"))
        output = self.output/name
        result = cached_metrics(output,candidate,200,1020000)
        if result is None:
            archive_partial(output)
            self.command(name,[sys.executable,str(ROOT/"scripts/manipulation.py"),"evaluate",
                "--checkpoint",str(checkpoint),"--output",str(output),"--episodes","200","--seed","1020000"],180)
            (output/"checkpoint.sha256").write_text(candidate["sha256"]+"\n")
            result = cached_metrics(output,candidate,200,1020000)
            if result is None:
                raise ValueError("Development metrics do not match the frozen checkpoint/seeds")
        self.job["evaluations_completed"] += 1
        self.persist()
        return result

    def run(self):
        self.state_dir.mkdir(parents=True,exist_ok=True)
        with (self.state_dir/".experiment.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.output.mkdir(exist_ok=True)
            (self.output/"model").mkdir(exist_ok=True)
            frozen = self.state_dir/"protocol.json"
            if frozen.exists() and json.loads(frozen.read_text()) != self.protocol:
                raise ValueError("Frozen imitation protocol changed")
            save(frozen,self.protocol)
            previous = json.loads((self.output/"job.json").read_text()) if (self.output/"job.json").exists() else {}
            if previous.get("status") == "complete":
                return previous
            self.job = dict(kind="imitation_probe",status="running",phase="initializing",pid=os.getpid(),
                attempts=previous.get("attempts",0)+1,started_at=previous.get("started_at",now()),
                arguments=self.protocol,commands=previous.get("commands",[]),acceptance_met=False,
                target_total_steps=0,current_model="model",evaluations_completed=0)
            self.persist()
            try:
                if self.job["attempts"] > 3:
                    raise RuntimeError("Imitation process restart budget exhausted")
                collected = self.output/"training128"
                report = json.loads((collected/"probe.json").read_text()) if (collected/"probe.json").exists() else {}
                valid = (report.get("status") == "complete" and report.get("checkpoint_sha256") == self.protocol["source_sha256"]
                         and report.get("config_sha256") == self.protocol["config_sha256"]
                         and report.get("seed") == 32 and report.get("episodes") == 128)
                if not valid:
                    archive_partial(collected)
                    self.command("collect-training128",[sys.executable,str(ROOT/"scripts/probe_action_residuals.py"),
                        "--checkpoint",str(self.source),"--output",str(collected),"--seed","32","--episodes","128",
                        "--search-seed","0","--workers","4"],600)
                reports = [self.prior,collected/"probe.json"]
                candidate = self.output/"imitation"
                try:
                    proof = json.loads((candidate/"imitation.json").read_text())
                    valid = (proof["source_sha256"] == self.protocol["source_sha256"]
                        and [p["sha256"] for p in proof["datasets"]] == [sha256(p) for p in reports]
                        and (candidate/"checkpoint.sha256").read_text().strip() == sha256(candidate/"policy.zip")
                        and (candidate/"config.sha256").read_text().strip() == sha256(candidate/"config.json")
                        and (candidate/"imitation.sha256").read_text().strip() == sha256(candidate/"imitation.json"))
                except (OSError,ValueError,KeyError):
                    valid = False
                if not valid:
                    archive_partial(candidate)
                    self.command("offline-imitation",[sys.executable,str(ROOT/"scripts/imitate_residual_policy.py"),
                        "--source",str(self.source),"--output",str(candidate),"--reports",*[str(p) for p in reports],
                        "--device","cuda"],180)
                    (candidate/"checkpoint.sha256").write_text(sha256(candidate/"policy.zip")+"\n")
                    (candidate/"config.sha256").write_text(sha256(candidate/"config.json")+"\n")
                    (candidate/"imitation.sha256").write_text(sha256(candidate/"imitation.json")+"\n")
                checkpoint_steps(candidate/"policy.zip",check_weights=True)
                source_result = self.evaluate("development-source",self.source)
                candidate_result = self.evaluate("development-imitation",candidate/"policy.zip")
                self.job["comparison"] = continuation_gate(source_result,candidate_result)
                save(self.output/"comparison.json",self.job["comparison"])
                self.job.update(status="complete",phase="complete",ended_at=now())
            except BaseException as error:
                self.job.update(status="failed",error=repr(error),ended_at=now())
                raise
            finally:
                self.persist()
            return self.job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--prior-report",type=Path,required=True)
    parser.add_argument("--state-dir",type=Path,required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this bounded optimization experiment on the Linux training host")
    print(json.dumps(Experiment(args.source,args.prior_report,args.state_dir).run(),indent=2))


if __name__ == "__main__":
    main()
