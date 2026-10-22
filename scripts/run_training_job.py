"""Run one Linux training stage, select on validation seeds, then evaluate once.

Run inside tmux. A completed job can still fail the manipulation acceptance gate;
job.json distinguishes process completion from policy quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import pickle
import shutil
import subprocess
import sys
import zipfile

from orca_feetech.training_supervision import (checkpoint_steps, latest_checkpoint,
                                              running_processes, training_target)

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def selection_score(metrics):
    return (metrics["success_rate"], -metrics["drop_rate"], -metrics["mean_final_error_deg"])


def grasp_acceptance(test,hold):
    comparison=dict(matched_hold_success_rate=hold['success_rate'],matched_hold_drop_rate=hold['drop_rate'],
        success_at_least_hold=test['success_rate']>=hold['success_rate'],
        drops_no_more_than_hold=test['drop_rate']<=hold['drop_rate'])
    accepted=bool(test['acceptance_met'] and comparison['success_at_least_hold'] and comparison['drops_no_more_than_hold'])
    return accepted,comparison


def archive_partial(path):
    if not path.exists():
        return
    index = 1
    while path.with_name(path.name + f"-aborted-{index}").exists():
        index += 1
    path.rename(path.with_name(path.name + f"-aborted-{index}"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--task-family',choices=['legacy-v7','v8','pen-grasp-v1','pen-rotate-v1'],default='legacy-v7')
    parser.add_argument("--object", choices=["pen", "cube"], default="pen")
    parser.add_argument("--angle", type=float, default=15)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--randomization", type=float, default=.3)
    parser.add_argument("--steps", type=int, default=3000000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument('--initialize-policy',type=Path)
    parser.add_argument("--device-config", type=Path)
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--friction", type=float, default=1.)
    parser.add_argument("--selection-episodes", type=int, default=40)
    parser.add_argument("--selection-seed", type=int, default=30000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=20000)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100000)
    parser.add_argument("--eval-every", type=int, default=250000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=.01)
    parser.add_argument("--target-kl", type=float, default=.03)
    parser.add_argument("--initial-log-std", type=float, default=0.)
    parser.add_argument('--zero-initial-action-mean',action='store_true')
    parser.add_argument('--relative-goal-features',action='store_true')
    parser.add_argument("--reset-log-std", type=float)
    parser.add_argument("--negative-goal-probability", type=float, default=.5)
    parser.add_argument("--minimum-goal-angle", type=float)
    parser.add_argument("--prefix-reset-checkpoint",type=Path)
    parser.add_argument("--prefix-reset-probability",type=float,default=.75)
    parser.add_argument("--continue-existing", action="store_true",
                        help="Recover an interrupted job without repeating completed evaluations")
    parser.add_argument("--recovery-num-envs", type=int,
                        help="Reduce workers on recovery after an out-of-memory failure")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Long training jobs belong on the Linux training host")
    args.output = args.output.resolve()
    state_path = args.output / "job.json"
    if args.continue_existing:
        state = json.loads(state_path.read_text())
        if state["status"] == "complete":
            print(json.dumps(state, indent=2))
            return
        if running_processes(state, args.output):
            parser.error("This job still has a live wrapper or worker; refusing duplicate training")
        for key, value in state["arguments"].items():
            if key not in ("continue_existing", "recovery_num_envs"):
                setattr(args, key, Path(value) if value is not None and key in
                        ("output", "resume", "initialize_policy", "device_config", "prefix_reset_checkpoint") else value)
        args.num_envs = args.recovery_num_envs or state.get("effective_num_envs", args.num_envs)
        state.update(status="running", pid=os.getpid(), recovered_at=now(),
                     resumptions=state.get("resumptions", 0) + 1, effective_num_envs=args.num_envs)
    else:
        state = dict(status="running", started_at=now(), pid=os.getpid(), hostname=platform.node(),
                     arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     commands=[], acceptance_met=False)
    from orca_feetech.task_families import task_family
    family=task_family(args.task_family)
    if args.initialize_policy is not None:
        if args.task_family!='pen-rotate-v1' or args.resume is not None:
            parser.error('Fresh rotation initialization only; cannot combine with resume')
        from orca_feetech.pen_rotate import validate_grasp_prefix
        validate_grasp_prefix(args.initialize_policy)
    if args.task_family!='legacy-v7':
        family.config_class(object=args.object,angle=args.angle,turns=args.turns,
                            horizon=args.horizon,randomization=args.randomization)
        if args.prefix_reset_checkpoint is not None:
            parser.error('New task families reject legacy prefix teachers')
    if args.task_family=='pen-grasp-v1' and (args.minimum_goal_angle is not None or args.negative_goal_probability!=.5):
        parser.error('Grasp task has no directional or goal-angle curriculum')
    if min(args.steps, args.num_envs, args.selection_episodes, args.test_episodes,
           args.checkpoint_every, args.eval_every) < 1:
        parser.error("Budgets must be positive")
    if args.minimum_goal_angle is not None and not 10 <= args.minimum_goal_angle <= args.angle:
        parser.error("Training minimum goal angle must be between 10 and task angle")
    if not 0 <= args.prefix_reset_probability <= 1:
        parser.error("Prefix probability must be in[0,1]")
    if args.prefix_reset_checkpoint is not None:
        if args.minimum_goal_angle is not None:
            parser.error("Prefix reset and goal-angle mixture cannot be combined")
        for key,path in (("prefix_reset_source_sha256",args.prefix_reset_checkpoint),
                         ("prefix_reset_config_sha256",args.prefix_reset_checkpoint.parent/"config.json")):
            value = sha256(path)
            if state.setdefault(key,value) != value:
                parser.error("Frozen prefix teacher or its metadata changed")
    selection_seeds = set(range(args.selection_seed, args.selection_seed + args.selection_episodes))
    test_seeds = set(range(args.test_seed, args.test_seed + args.test_episodes))
    if min(args.selection_seed, args.test_seed) < 20000 or selection_seeds & test_seeds:
        parser.error("Selection/test seeds must be disjoint and >=20000")
    if not args.continue_existing:
        args.output.mkdir(parents=True, exist_ok=False)
    job_lock = (args.output / ".job.lock").open("a")
    try:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another process already owns this training job")
    state.setdefault("target_total_steps", training_target(state["arguments"]))
    state.setdefault("model_directories", ["model"])
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("MUJOCO_GL", "egl")

    def run(name, command):
        entry = dict(name=name, argv=command, started_at=now(), status="running")
        state["phase"] = name
        state["commands"].append(entry)
        save(state_path, state)
        print(now(), name, flush=True)
        with (args.output / (name + ".log")).open("a") as log:
            if name.startswith("train"):
                link = args.output / "current-training.log"
                if link.is_symlink():
                    link.unlink()
                if not link.exists():
                    link.symlink_to(name + ".log")
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            entry["pid"] = process.pid
            save(state_path, state)
            code = process.wait()
        entry.update(status="complete" if code == 0 else "failed", returncode=code, ended_at=now())
        save(state_path, state)
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect {args.output / (name + '.log')}")

    python = [sys.executable, str(ROOT / 'scripts' / family.cli_script)]

    def evaluate_phase(name, checkpoint, episodes, seed, baseline=None, video=False):
        output = args.output / name
        digest = sha256(checkpoint)
        if (output / "metrics.json").exists() and (output / "checkpoint.sha256").exists():
            result = json.loads((output / "metrics.json").read_text())
            if ((output / "checkpoint.sha256").read_text().strip() == digest
                    and result["episodes"] == episodes
                    and [r["seed"] for r in result["trials"]] == list(range(seed, seed + episodes))
                    and result["policy"] == (baseline or str(checkpoint))):
                return result
        archive_partial(output)
        command = python + ["evaluate", "--checkpoint", str(checkpoint), "--episodes", str(episodes),
                            "--seed", str(seed), "--output", str(output)]
        if baseline:
            command += ["--baseline", baseline]
        if video:
            command += ["--video"]
        run(name, command)
        (output / "checkpoint.sha256").write_text(digest + "\n")
        return json.loads((output / "metrics.json").read_text())

    def finished_model():
        for name in reversed(state["model_directories"]):
            checkpoint = args.output / name / "policy.zip"
            if not checkpoint.is_file():
                continue
            try:
                _, meta = family.load_config(checkpoint)
                if meta["trained"] and checkpoint_steps(checkpoint, check_weights=True) >= state["target_total_steps"]:
                    return checkpoint.parent
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError, RuntimeError, pickle.UnpicklingError):
                continue
        return None

    try:
        model = finished_model() if args.continue_existing else None
        if model is None:
            checkpoint_kwargs={} if args.task_family=='legacy-v7' else dict(config_loader=family.load_config)
            source, steps = (latest_checkpoint([args.output / p for p in state["model_directories"]], args.resume,**checkpoint_kwargs)
                             if args.continue_existing else (args.resume, checkpoint_steps(args.resume) if args.resume else 0))
            if args.resume and source is None:
                raise RuntimeError("No valid checkpoint for this resumed stage; refusing to restart its entire history")
            name = "model"
            if (args.output / name).exists():
                index = 1
                while (args.output / f"model-recovery-{index}").exists():
                    index += 1
                name = f"model-recovery-{index}"
                state["model_directories"].append(name)
            model = args.output / name
            state.update(current_model=name, recovery_checkpoint=str(source) if source else None)
            remaining = max(1, state["target_total_steps"] - steps)
            command = python + ["train", "--output", str(model), "--steps", str(remaining)]
            if source:
                command += ["--resume", str(source)]
            elif args.initialize_policy is not None:
                command += ['--initialize-policy',str(args.initialize_policy)]
            if (args.reset_log_std is not None
                    and (source is None or not Path(source).resolve().is_relative_to(args.output))):
                command += ["--reset-log-std", str(args.reset_log_std)]
            for key in ("object", "angle", "turns", "horizon", "randomization", "num_envs", "device",
                        "seed", "device_config", "mass_kg", "friction", "checkpoint_every", "eval_every",
                        "learning_rate", "entropy_coef", "target_kl", "initial_log_std",
                        "minimum_goal_angle"):
                value = getattr(args, key)
                if value is not None:
                    command += ["--" + key.replace("_", "-"), str(value)]
            command += ["--negative-goal-probability", str(args.negative_goal_probability)]
            if args.zero_initial_action_mean:
                command += ['--zero-initial-action-mean']
            if args.relative_goal_features:
                command += ['--relative-goal-features']
            if args.prefix_reset_checkpoint is not None:
                command += ["--prefix-reset-checkpoint",str(args.prefix_reset_checkpoint),
                            "--prefix-reset-probability",str(args.prefix_reset_probability)]
            run("train" if name == "model" else name.replace("model", "train", 1), command)
        state["current_model"] = str(model.relative_to(args.output))
        candidates = []
        selected = args.output / "selected"
        preserve_selection = (args.continue_existing and (selected / "config.json").is_file()
                              and (selected / "policy.zip").is_file() and state.get("selected_from")
                              and state.get("selected_sha256") == sha256(selected / "policy.zip"))
        if not preserve_selection:
            for directory in state["model_directories"]:
                for name in ("policy", "best_model"):
                    checkpoint = args.output / directory / (name + ".zip")
                    if not checkpoint.is_file():
                        continue
                    try:
                        checkpoint_steps(checkpoint, check_weights=True)
                    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError, RuntimeError, pickle.UnpicklingError):
                        continue
                    prefix = "validation-" + ("" if directory == "model" else directory + "-") + name
                    metrics = evaluate_phase(prefix, checkpoint, args.selection_episodes, args.selection_seed)
                    candidates.append((checkpoint, metrics))
            chosen, metrics = max(candidates, key=lambda item: selection_score(item[1]))
            archive_partial(selected)
            selected.mkdir()
            shutil.copy2(chosen, selected / "policy.zip")
            metadata = json.loads((chosen.parent / "config.json").read_text())
            actual = checkpoint_steps(chosen)
            metadata.update(selected_from=str(chosen), checkpoint_actual_steps=actual,
                            actual_steps=actual, trained=True, training_stage_target=state["target_total_steps"])
            save(selected / "config.json", metadata)
            state.update(selected_from=str(chosen), selected_sha256=sha256(selected / "policy.zip"),
                         validation={k: v for k, v in metrics.items() if k != "trials"})
            save(state_path, state)
        for baseline in ((None,'hold','random','grasp') if args.task_family=='pen-rotate-v1' else (None,'hold','random')):
            name = "test-" + (baseline or "policy")
            evaluate_phase(name, selected / "policy.zip", args.test_episodes, args.test_seed, baseline=baseline)
        test = json.loads((args.output / "test-policy/metrics.json").read_text())
        scalar={k:v for k,v in test.items() if k!='trials'}
        if args.task_family=='pen-grasp-v1':
            hold=json.loads((args.output/'test-hold/metrics.json').read_text())
            accepted,comparison=grasp_acceptance(test,hold)
            scalar.update(policy_sample_threshold_met=test['acceptance_met'],baseline_comparison=comparison,
                          acceptance_met=accepted)
        if args.task_family=='pen-rotate-v1':
            comparisons={}
            accepted=test['acceptance_met']
            for baseline in ('hold','grasp'):
                reference=json.loads((args.output/f'test-{baseline}/metrics.json').read_text())
                passed,comparison=grasp_acceptance(test,reference)
                comparisons[baseline]=comparison
                accepted=accepted and passed
            scalar.update(policy_sample_threshold_met=test['acceptance_met'],baseline_comparisons=comparisons,
                          acceptance_met=bool(accepted),continuous_rotation_accepted=False)
        state.update(test=scalar,acceptance_met=scalar['acceptance_met'])
        if args.video:
            # Fixed first five TEST episodes, without selecting successful trials.
            try:
                evaluate_phase("video", selected / "policy.zip", min(5, args.test_episodes), args.test_seed, video=True)
            except RuntimeError as error:
                state["video_error"] = str(error)
        state.update(status="complete", phase="complete", ended_at=now())
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=str(error), ended_at=now())
        raise
    finally:
        save(state_path, state)
        print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
