"""Persistent Linux watchdog for existing training jobs and the pen curriculum.

Use a systemd user service. Checkpoint recovery retains the original step budget;
quality-gate failures are reported separately from crashed or stalled processes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

from orca_feetech.training_supervision import (failure_kind, process_argv, progress_sample,
                                              running_processes, wrapper_alive)
from run_training_job import ROOT, archive_partial, now, save
from orca_feetech.manipulation import TaskConfig


def tail(path, limit=16000):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size-limit))
            return stream.read().decode(errors="replace")
    except OSError:
        return ""


def load_plan(path):
    if path is None:
        return None
    plan = json.loads(path.read_text())
    keys = {"name", "angle", "turns", "horizon", "steps", "randomization"}
    if not isinstance(plan, list) or not 1 <= len(plan) <= 10:
        raise ValueError("Plan requires 1..10 stages")
    names = set()
    for stage in plan:
        if not isinstance(stage, dict) or set(stage) != keys:
            raise ValueError("Plan stage must contain name, angle, turns, horizon, steps, randomization")
        name = stage["name"]
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", name)
                or "-quality-" in name or name in names):
            raise ValueError("Plan stage names must be unique safe directory names")
        names.add(name)
        if any(type(stage[k]) is not int for k in ("steps", "turns", "horizon")) or stage["steps"] < 1:
            raise ValueError("Plan budgets, turns and horizons must be positive integers")
        TaskConfig(object="pen", **{k: stage[k] for k in ("angle", "turns", "horizon", "randomization")})
    return plan


def validate_plan_seeds(initial, plan, quality_retries):
    blocks = []
    for index in range(len(plan)):
        blocks.append((initial["selection_seed"] + 1000*index, initial["selection_episodes"],
                       f"development stage {index}"))
        for attempt in range(quality_retries + 1):
            blocks.append((initial["test_seed"] + 1000*index + 100000*attempt,
                           initial["test_episodes"], f"test stage {index} attempt {attempt}"))
    for i, (start, count, label) in enumerate(blocks):
        for other, size, other_label in blocks[:i]:
            if max(start, other) < min(start+count, other+size):
                raise ValueError(f"Plan seed blocks overlap: {label} / {other_label}")


def stage_options(initial, index, plan=None):
    options = initial.copy()
    if plan is not None:
        stage = plan[index]
        options.update({k: v for k, v in stage.items() if k != "name"})
        options.update(selection_seed=initial["selection_seed"] + 1000*index,
                       test_seed=initial["test_seed"] + 1000*index)
        return stage["name"], options
    name, angle, turns, horizon, steps = [
        ("initial", 15, 1, 240, initial["steps"]),
        ("15-r1", 15, 1, 240, 3000000), ("45-r1", 45, 1, 240, 3000000),
        ("90-r1", 90, 1, 240, 3000000), ("90x4-r1", 90, 4, 800, 10000000)][index]
    options.update(angle=angle, turns=turns, horizon=horizon, steps=steps,
                   randomization=initial["randomization"] if index == 0 else 1,
                   selection_seed=initial["selection_seed"] + 1000*index,
                   test_seed=initial["test_seed"] + 1000*index)
    return name, options


def job_command(options, output, resume=None):
    command = [sys.executable, str(ROOT / "scripts/run_training_job.py"), "--output", str(output)]
    if options.get('task_family') is not None:
        command += ['--task-family',options['task_family']]
    for key in ("object", "angle", "turns", "horizon", "randomization", "steps", "num_envs", "device",
                "seed", "device_config", "mass_kg", "friction", "selection_episodes", "selection_seed",
                "test_episodes", "test_seed", "checkpoint_every", "eval_every", "learning_rate",
                "entropy_coef", "target_kl", "initial_log_std", "negative_goal_probability",
                "minimum_goal_angle"):
        if options.get(key) is not None:
            command += ["--" + key.replace("_", "-"), str(options[key])]
    if resume:
        command += ["--resume", str(resume)]
    if options.get("prefix_reset_checkpoint") is not None:
        command += ["--prefix-reset-checkpoint",str(options["prefix_reset_checkpoint"]),
                    "--prefix-reset-probability",str(options.get("prefix_reset_probability",.75))]
    if options.get("video"):
        command += ["--video"]
    if options.get('zero_initial_action_mean'):
        command += ['--zero-initial-action-mean']
    if options.get('initialize_policy') and not resume:
        command += ['--initialize-policy',str(options['initialize_policy'])]
    return command


def quality_retry_options(initial, index, attempt, steps, plan=None):
    """Bounded same-skill fine tuning; never advance a failed skill or reuse its test set."""
    name, options = (("initial", initial.copy()) if index == 0 and plan is None
                     else stage_options(initial, index, plan))
    options.update(steps=steps, learning_rate=max(1e-5, initial.get("learning_rate", 3e-4) / 2),
                   entropy_coef=min(.001, initial.get("entropy_coef", .01)))
    # The development set remains fixed. Each retry has its own formal test
    # block, disjoint from every course stage and every prior retry.
    options["test_seed"] += 100000 * attempt
    return f"{name}-quality-{attempt}", options


def terminate_verified(state, output):
    """Signal only the identified job's process group or exact worker command."""
    targets = running_processes(state, output)
    if not targets:
        return True
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 3)):
        live = running_processes(state, output)
        if not live:
            return True
        if wrapper_alive(state, output) and os.getpgid(state["pid"]) == state["pid"]:
            try:
                os.killpg(state["pid"], sig)
            except ProcessLookupError:
                pass
        else:
            for pid in live:
                # Recheck immediately before signalling to avoid PID reuse.
                if pid in running_processes(state, output):
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and running_processes(state, output):
            time.sleep(.5)
    return not running_processes(state, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-job", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    curriculum = parser.add_mutually_exclusive_group()
    curriculum.add_argument("--course", action="store_true")
    curriculum.add_argument("--plan", type=Path, help="Explicit ordered pen curriculum JSON")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--stall-seconds", type=float, default=300)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=30)
    parser.add_argument("--quality-retries", type=int, default=0,
                        help="Bounded extra training pilots on an unfinished skill (no promotion)")
    parser.add_argument("--quality-retry-steps", type=int, default=2000000)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("The watchdog requires Linux /proc and systemd")
    if min(args.poll_seconds, args.stall_seconds, args.retry_seconds) <= 0 or args.max_restarts < 0:
        parser.error("Invalid watchdog timing/retry limits")
    if not 0 <= args.quality_retries <= 3 or args.quality_retry_steps < 1:
        parser.error("Invalid bounded quality-pilot budget")
    plan = load_plan(args.plan)
    args.initial_job = args.initial_job.resolve()
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "watchdog.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another watchdog owns this state directory", flush=True)
        return
    path = args.state_dir / "watchdog.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["initial_job"] != str(args.initial_job) or state["course"] != args.course:
            raise ValueError("State directory belongs to a different job or monitoring plan")
        if state.get("plan") != plan:
            raise ValueError("Curriculum differs from the persisted monitoring plan")
        if (state.get("quality_retries", 0) != args.quality_retries
                or state.get("quality_retry_steps", 2000000) != args.quality_retry_steps):
            raise ValueError("Quality pilot budget differs from the persisted monitoring plan")
        if state["status"] in ("complete", "quality_gate_failed", "recovery_exhausted", "needs_review"):
            print(json.dumps(state, indent=2), flush=True)
            return
    else:
        initial = json.loads((args.initial_job / "job.json").read_text())["arguments"]
        if initial.get('task_family') in ('pen-grasp-v1','pen-rotate-v1') and (args.course or plan is not None):
            parser.error('Grasp and single-target rotation pilots cannot enter an automatic curriculum')
        if args.course and (initial["object"] != "pen" or initial["angle"] != 15
                            or initial["turns"] != 1 or initial["randomization"] != .3):
            parser.error("Pen course must start with pen/15deg/r0.3")
        if plan is not None:
            if initial["object"] != "pen" or any(initial[k] != v for k, v in plan[0].items() if k != "name"):
                parser.error("Initial job does not match the first explicit plan stage")
            validate_plan_seeds(initial, plan, args.quality_retries)
        state = dict(status="monitoring", initial_job=str(args.initial_job), course=args.course,
                     plan=plan,
                     initial_arguments=initial, active_job=str(args.initial_job), stage_index=0,
                     stage_name=plan[0]["name"] if plan else "initial",
                     restarts=0, stage_results=[], started_at=now(),
                     last_progress_at=time.time(), sample=None, pending_restart=None,
                     quality_attempt=0, quality_retries=args.quality_retries,
                     quality_retry_steps=args.quality_retry_steps)

    def persist():
        state.update(updated_at=now(), supervisor_pid=os.getpid(), poll_seconds=args.poll_seconds,
                     stall_seconds=args.stall_seconds, max_restarts=args.max_restarts)
        save(path, state)
        link = args.state_dir / "current-job"
        temporary = args.state_dir / "current-job.tmp"
        if temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(state["active_job"], target_is_directory=True)
        temporary.replace(link)

    def event(kind, **details):
        value = dict(time=now(), event=kind, stage=state["stage_name"])
        value.update(details)
        with (args.state_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    def launch(command):
        state.update(launched_argv=command, launched_at=time.time())
        persist()
        with (args.state_dir / "worker-console.log").open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        state.update(launched_pid=child.pid, sample=None, last_progress_at=time.time(), pending_restart=None)
        persist()

    event("monitor_started", job=state["active_job"], poll_seconds=args.poll_seconds)
    persist()
    while True:
        job = Path(state["active_job"])
        state_file = job / "job.json"
        if not state_file.exists():
            if process_argv(state.get("launched_pid", 0)) == state.get("launched_argv"):
                time.sleep(args.poll_seconds)
                continue
            if (job.is_relative_to(args.state_dir) and state.get("launched_argv")
                    and state["restarts"] < args.max_restarts):
                state["restarts"] += 1
                archive_partial(job)
                event("startup_retry", attempt=state["restarts"])
                launch(state["launched_argv"])
                time.sleep(args.poll_seconds)
                continue
            state.update(status="needs_review", error="Job did not create its state file")
            event("missing_job_state", job=str(job))
            persist()
            return
        try:
            current = json.loads(state_file.read_text())
            sample = progress_sample(job, current)
        except (OSError, ValueError, KeyError) as error:
            event("unreadable_state", error=str(error))
            time.sleep(args.poll_seconds)
            continue
        state.update(job_status=current["status"], job_phase=current.get("phase"), progress=sample)
        if sample != state["sample"]:
            state.update(sample=sample, last_progress_at=time.time())
        if current["status"] == "complete":
            result = dict(stage=state["stage_name"], job=str(job), test=current.get("test"),
                          acceptance_met=current["acceptance_met"])
            state["stage_results"].append(result)
            event("stage_finished", **result)
            if not current["acceptance_met"]:
                attempt = state.get("quality_attempt", 0) + 1
                if attempt <= args.quality_retries:
                    name, options = quality_retry_options(state["initial_arguments"],
                                                           state["stage_index"], attempt,
                                                           args.quality_retry_steps, plan)
                    options["num_envs"] = current.get("effective_num_envs", options["num_envs"])
                    output = args.state_dir / name
                    # A quality retry remains on the current course stage.
                    # Keep the failed job, test and selected model intact.
                    state.update(quality_attempt=attempt, active_job=str(output), restarts=0,
                                 pending_restart=None, status="monitoring")
                    event("quality_pilot_starting", job=str(output), attempt=attempt,
                          steps=args.quality_retry_steps, test_seed=options["test_seed"],
                          learning_rate=options["learning_rate"], entropy_coef=options["entropy_coef"])
                    launch(job_command(options, output, job / "selected/policy.zip"))
                    time.sleep(args.poll_seconds)
                    continue
                state.update(status="quality_gate_failed", error="Policy finished but did not meet acceptance")
                event("quality_budget_exhausted", attempts=state.get("quality_attempt", 0),
                      test=current.get("test"), job=str(job))
                persist()
                return
            last_stage = len(plan)-1 if plan is not None else 4 if args.course else 0
            if state["stage_index"] == last_stage:
                state.update(status="complete", selected_checkpoint=str(job / "selected/policy.zip"))
                persist()
                return
            index = state["stage_index"] + 1
            name, options = stage_options(state["initial_arguments"], index, plan)
            options["num_envs"] = current.get("effective_num_envs", options["num_envs"])
            output = args.state_dir / name
            state.update(stage_index=index, stage_name=name, active_job=str(output), restarts=0,
                         pending_restart=None, quality_attempt=0, status="monitoring")
            event("stage_starting", job=str(output))
            launch(job_command(options, output, job / "selected/policy.zip"))
            time.sleep(args.poll_seconds)
            continue
        live = running_processes(current, job)
        launching = (process_argv(state.get("launched_pid", 0)) == state.get("launched_argv")
                     and time.time() - state.get("launched_at", 0) < 120)
        stalled = time.time() - state["last_progress_at"] > args.stall_seconds
        if (live or launching) and not stalled:
            state.update(status="monitoring", live_pids=live)
            persist()
            time.sleep(args.poll_seconds)
            continue
        if state["pending_restart"] is None:
            if state["restarts"] >= args.max_restarts:
                state.update(status="recovery_exhausted", error="Automatic restart limit reached")
                event("recovery_exhausted", job=str(job))
                persist()
                return
            reason = "stalled" if stalled and live else "process_exited"
            if live and not terminate_verified(current, job):
                state.update(status="needs_review", error="Verified workers did not stop; no duplicate launched")
                persist()
                return
            text = str(current.get("error", "")) + tail(job / (current.get("phase", "train") + ".log"))
            kind = failure_kind(text)
            state["restarts"] += 1
            pending = dict(reason=reason, kind=kind,
                           after=time.time() + min(300, args.retry_seconds * 2**(state["restarts"]-1)))
            state.update(status="recovery_wait", pending_restart=pending)
            event("recovery_scheduled", reason=reason, failure_kind=kind, attempt=state["restarts"])
            persist()
        pending = state["pending_restart"]
        if time.time() < pending["after"]:
            time.sleep(args.poll_seconds)
            continue
        if shutil.disk_usage(ROOT).free < 2 * 1024**3:
            state["status"] = "waiting_for_disk"
            persist()
            time.sleep(args.poll_seconds)
            continue
        if current["arguments"]["device"] == "cuda":
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15)
            free = [int(line.strip()) for line in gpu.stdout.splitlines() if line.strip().isdigit()]
            if gpu.returncode or not any(value >= 1024 for value in free):
                state["status"] = "waiting_for_gpu"
                persist()
                time.sleep(args.poll_seconds)
                continue
        command = [sys.executable, str(ROOT / "scripts/run_training_job.py"), "--output", str(job),
                   "--continue-existing"]
        if pending["kind"] == "memory":
            workers = max(1, current.get("effective_num_envs", current["arguments"]["num_envs"]) // 2)
            command += ["--recovery-num-envs", str(workers)]
            event("workers_reduced", num_envs=workers)
        event("restarting", job=str(job), attempt=state["restarts"], argv=command)
        launch(command)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
