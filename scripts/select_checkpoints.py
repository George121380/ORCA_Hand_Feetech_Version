"""Bounded, resumable checkpoint selection; no learning or hardware access.

Freeze the manifest before scoring. Screen saved checkpoints on development,
confirm only the top three on fresh development seeds, freeze one winner,
then (only after development passes) run one untouched formal test block.
Uses the existing watchdog service and heartbeat state format.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile

from orca_feetech.checkpoint import load_config
from orca_feetech.training_supervision import checkpoint_steps
from run_training_job import ROOT, archive_partial, now, save, selection_score, sha256


def validate_protocol(protocol):
    candidates = protocol["candidates"]
    if not 1 <= len(candidates) <= 48 or not 1 <= protocol["workers"] <= 4:
        raise ValueError("Selection permits 1..48 candidates and 1..4 workers")
    if not 1 <= protocol["top_k"] <= min(3, len(candidates)):
        raise ValueError("Confirmation permits at most three candidates")
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    blocks = []
    for split, expected in (("screen", 40), ("confirmation", 200), ("test", 100)):
        seed, count = protocol[split]["seed"], protocol[split]["episodes"]
        if type(seed) is not int or seed < 20000 or type(count) is not int or count != expected:
            raise ValueError("Use explicit development40/200 and formal100 seed blocks")
        if any(max(seed, s) < min(seed+count, s+n) for s, n in blocks):
            raise ValueError("Development and formal blocks overlap")
        blocks.append((seed, count))


def policy_signature(path):
    """Conservatively deduplicate identical serialized weights and architecture.

    Outer ZIP timestamps, optimizer state and training counters cannot change
    inference. Different policy classes/kwargs remain distinct candidates.
    """
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
        payload = dict(weights=hashlib.sha256(archive.read("policy.pth")).hexdigest(),
                       policy_class=data.get("policy_class"), policy_kwargs=data.get("policy_kwargs"),
                       use_sde=data.get("use_sde"))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def manifest(directories, *, screen_seed, confirmation_seed, test_seed):
    candidates, models, fingerprint = [], {}, None
    for directory in sorted(Path(d).resolve() for d in directories):
        for path in sorted(directory.glob("*.zip")):
            if path.stem not in ("policy", "best_model") and not path.stem.startswith("checkpoint_"):
                continue
            config, metadata = load_config(path)
            if (config.object != "pen" or config.angle != 35 or config.turns != 1
                    or config.horizon != 240 or config.randomization != 1):
                raise ValueError("This protocol is limited to canonical pen35/r1 checkpoints")
            if fingerprint is not None and fingerprint != metadata["fingerprint"]:
                raise ValueError("Candidate task contracts differ")
            fingerprint = metadata["fingerprint"]
            steps = checkpoint_steps(path, check_weights=True)
            digest = sha256(path)
            signature = policy_signature(path)
            if signature in models:
                models[signature].setdefault("equivalent_checkpoints", []).append(
                    dict(checkpoint=str(path), sha256=digest))
                continue
            candidate = dict(id=digest, checkpoint=str(path), sha256=digest, policy_signature=signature,
                             config_sha256=sha256(path.parent/"config.json"), steps=steps)
            candidates.append(candidate)
            models[signature] = candidate
    protocol = dict(schema=1, kind="checkpoint_selection", fingerprint=fingerprint,
                    candidates=candidates, workers=4, top_k=min(3, len(candidates)),
                    screen=dict(seed=screen_seed, episodes=40),
                    confirmation=dict(seed=confirmation_seed, episodes=200),
                    test=dict(seed=test_seed, episodes=100), video=True,
                    deduplication="serialized_policy_and_architecture_v1",
                    purpose="Development-only selection before one frozen independent formal test")
    validate_protocol(protocol)
    return protocol


def cached_metrics(output, candidate, episodes, seed, baseline=None):
    try:
        data = json.loads((output/"metrics.json").read_text())
        valid = ((output/"checkpoint.sha256").read_text().strip() == candidate["sha256"]
                 and sha256(output/"config.json") == candidate["config_sha256"]
                 and data["episodes"] == episodes
                 and data["policy"] == (baseline or candidate["checkpoint"])
                 and [t["seed"] for t in data["trials"]] == list(range(seed, seed+episodes)))
        return data if valid else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def adopt_development_cache(protocol, previous_job, state_dir):
    """Copy only exact development caches into a new, separately frozen run."""
    previous_job, state_dir = Path(previous_job), Path(state_dir)
    if (state_dir/"protocol.json").exists() or (state_dir/"job/job.json").exists():
        raise ValueError("Import caches before starting the new protocol")
    counts = dict(screen=0, confirmation=0)
    for split in counts:
        block = protocol[split]
        for candidate in protocol["candidates"]:
            source = previous_job/split/candidate["id"]
            if cached_metrics(source, candidate, block["episodes"], block["seed"]):
                target = state_dir/"job"/split/candidate["id"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
                counts[split] += 1
    state_dir.mkdir(parents=True, exist_ok=True)
    save(state_dir/"cache-import.json", dict(previous_job=str(previous_job), imported=counts,
                                           formal_caches_imported=0))
    return counts


class SelectionJob:
    def __init__(self, protocol, state_dir, evaluator=None):
        validate_protocol(protocol)
        self.protocol, self.state_dir = protocol, Path(state_dir)
        self.output = self.state_dir/"job"
        self.lock = threading.RLock()
        self.evaluator = evaluator
        self.started = time.monotonic()

    def persist(self):
        with self.lock:
            save(self.output/"job.json", self.job)
            completed = self.job.get("evaluations_completed", 0)
            progress = dict(total_steps=0, elapsed_s=time.monotonic()-self.started,
                            evaluations_completed=completed,
                            evaluations_total=len(self.protocol["candidates"])+self.protocol["top_k"]+4,
                            candidates_total=len(self.protocol["candidates"]))
            save(self.output/"model/progress.json", progress)
            done = self.job["status"] == "complete"
            status = ("complete" if self.job["acceptance_met"] else "quality_gate_failed") if done else (
                "needs_review" if self.job["status"] == "failed" else "monitoring")
            save(self.state_dir/"watchdog.json", dict(
                kind="checkpoint_selection", status=status, active_job=str(self.output),
                stage_name="35-checkpoint-selection", stage_results=(
                    [dict(stage="35-checkpoint-selection", test=self.job.get("test"),
                          acceptance_met=self.job["acceptance_met"])] if done else []),
                supervisor_pid=os.getpid(), updated_at=now(), restarts=self.job["attempts"]-1,
                quality_retries=0, error=self.job.get("error")))

    def evaluate(self, candidate, output, episodes, seed, baseline=None, video=False):
        result = cached_metrics(output, candidate, episodes, seed, baseline)
        if result is None:
            if sha256(Path(candidate["checkpoint"])) != candidate["sha256"]:
                raise ValueError("Candidate checkpoint changed after manifest freeze")
            if sha256(Path(candidate["checkpoint"]).parent/"config.json") != candidate["config_sha256"]:
                raise ValueError("Candidate metadata changed after manifest freeze")
            archive_partial(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            if self.evaluator:
                result = self.evaluator(candidate, output, episodes, seed, baseline, video)
                output.mkdir(exist_ok=False)
                shutil.copy2(Path(candidate["checkpoint"]).parent/"config.json", output/"config.json")
                save(output/"metrics.json", result)
            else:
                argv = [sys.executable, str(ROOT/"scripts/manipulation.py"), "evaluate",
                        "--checkpoint", candidate["checkpoint"], "--output", str(output),
                        "--episodes", str(episodes), "--seed", str(seed)]
                if baseline:
                    argv += ["--baseline", baseline]
                if video:
                    argv += ["--video"]
                with output.with_suffix(".log").open("a") as log:
                    process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    entry = dict(argv=argv, pid=process.pid, status="running", started_at=now())
                    with self.lock:
                        self.job["commands"].append(entry)
                        self.persist()
                    try:
                        code = process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise RuntimeError(f"Evaluation exceeded 180s: {output}") from None
                    with self.lock:
                        entry.update(status="complete" if code == 0 else "failed", ended_at=now(), returncode=code)
                        self.persist()
                    if code:
                        raise RuntimeError(f"Evaluation exited {code}: {output.with_suffix('.log')}")
                result = json.loads((output/"metrics.json").read_text())
            (output/"checkpoint.sha256").write_text(candidate["sha256"]+"\n")
            if cached_metrics(output, candidate, episodes, seed, baseline) is None:
                raise ValueError("Evaluation result does not match frozen candidate/seed block")
        with self.lock:
            self.job["evaluations_completed"] += 1
            self.persist()
        return result

    def batch(self, split, candidates):
        self.job["phase"] = split
        self.persist()
        block = self.protocol[split]
        with ThreadPoolExecutor(max_workers=self.protocol["workers"]) as pool:
            futures = {pool.submit(self.evaluate, c, self.output/split/c["id"],
                                   block["episodes"], block["seed"]): c for c in candidates}
            results = [dict(candidate=futures[f], metrics=f.result()) for f in as_completed(futures)]
        results.sort(key=lambda r: r["candidate"]["id"])
        results.sort(key=lambda r: selection_score(r["metrics"]), reverse=True)
        save(self.output/(split+"-ranking.json"), [dict(
            candidate=r["candidate"], metrics={k:v for k,v in r["metrics"].items() if k != "trials"},
            is_formal_evaluation=False) for r in results])
        return results

    def run(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ownership = (self.state_dir/".selection.lock").open("a")
        fcntl.flock(self.ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.output.mkdir(exist_ok=True)
        (self.output/"model").mkdir(exist_ok=True)
        frozen = self.state_dir/"protocol.json"
        if frozen.exists():
            if json.loads(frozen.read_text()) != self.protocol:
                self.ownership.close()
                raise ValueError("Frozen selection protocol changed")
        else:
            save(frozen, self.protocol)
        state = self.output/"job.json"
        previous = json.loads(state.read_text()) if state.exists() else {}
        if previous.get("status") == "complete":
            self.ownership.close()
            return previous
        attempts = previous.get("attempts", 0)+1
        self.job = dict(kind="checkpoint_selection", status="running", phase="screen", pid=os.getpid(),
                        attempts=attempts, arguments=self.protocol, target_total_steps=0,
                        commands=previous.get("commands", []), started_at=previous.get("started_at", now()),
                        current_model="model", evaluations_completed=0, acceptance_met=False)
        try:
            if attempts > 3:
                raise RuntimeError("Selection restart budget exhausted")
            self.persist()
            screening = self.batch("screen", self.protocol["candidates"])
            finalists = [r["candidate"] for r in screening[:self.protocol["top_k"]]]
            confirmation = self.batch("confirmation", finalists)
            winner, metrics = confirmation[0]["candidate"], confirmation[0]["metrics"]
            selected = self.output/"selected"
            selected.mkdir(exist_ok=True)
            if (selected/"policy.zip").exists() and sha256(selected/"policy.zip") != winner["sha256"]:
                raise ValueError("Frozen winner changed on recovery")
            shutil.copy2(winner["checkpoint"], selected/"policy.zip")
            metadata = json.loads((Path(winner["checkpoint"]).parent/"config.json").read_text())
            metadata.update(actual_steps=winner["steps"], checkpoint_actual_steps=winner["steps"],
                            selected_from=winner["checkpoint"], checkpoint_selection=dict(
                                protocol_sha256=sha256(frozen), is_formal_evaluation=False))
            save(selected/"config.json", metadata)
            self.job.update(selected_from=winner["checkpoint"], selected_sha256=winner["sha256"],
                            validation=dict({k:v for k,v in metrics.items() if k != "trials"},
                                            is_formal_evaluation=False))
            self.persist()
            if metrics["success_rate"] < .8 or metrics["drop_rate"] > .1:
                self.job["error"] = "Fresh development confirmation below threshold; formal test not consumed"
            else:
                chosen = dict(winner, checkpoint=str(selected/"policy.zip"),
                              config_sha256=sha256(selected/"config.json"))
                block = self.protocol["test"]
                for baseline in (None, "hold", "random"):
                    name = "test-"+(baseline or "policy")
                    self.job["phase"] = name
                    self.persist()
                    result = self.evaluate(chosen, self.output/name, block["episodes"], block["seed"], baseline)
                    if baseline is None:
                        self.job.update(test={k:v for k,v in result.items() if k != "trials"},
                                        acceptance_met=result["acceptance_met"])
                if self.protocol["video"]:
                    self.job["phase"] = "video"
                    self.persist()
                    self.evaluate(chosen, self.output/"video", 5, block["seed"], video=True)
            self.job.update(status="complete", phase="complete", ended_at=now())
        except BaseException as error:
            self.job.update(status="failed", error=str(error), ended_at=now())
            raise
        finally:
            self.persist()
            self.ownership.close()
        return self.job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run checkpoint rollouts on the Linux training host")
    protocol = json.loads(args.manifest.read_text())
    validate_protocol(protocol)
    for candidate in protocol["candidates"]:
        path = Path(candidate["checkpoint"])
        config, metadata = load_config(path)
        if (config.object != "pen" or config.angle != 35 or config.turns != 1
                or config.horizon != 240 or config.randomization != 1
                or metadata["fingerprint"] != protocol["fingerprint"] or sha256(path) != candidate["sha256"]
                or sha256(path.parent/"config.json") != candidate["config_sha256"]
                or (candidate.get("policy_signature") is not None
                    and policy_signature(path) != candidate["policy_signature"])
                or checkpoint_steps(path, check_weights=True) != candidate["steps"]):
            raise ValueError("Candidate differs from frozen manifest")
    job = SelectionJob(protocol, args.state_dir).run()
    print(json.dumps({k:v for k,v in job.items() if k not in ("arguments", "commands")}), flush=True)


if __name__ == "__main__":
    main()
