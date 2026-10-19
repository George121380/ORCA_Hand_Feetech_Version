"""Run one Linux training stage, select on validation seeds, then evaluate once.

Run inside tmux. A completed job can still fail the manipulation acceptance gate;
job.json distinguishes process completion from policy quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def selection_score(metrics):
    return (metrics["success_rate"], -metrics["drop_rate"], -metrics["mean_final_error_deg"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--device-config", type=Path)
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--friction", type=float, default=1.)
    parser.add_argument("--selection-episodes", type=int, default=40)
    parser.add_argument("--selection-seed", type=int, default=30000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=20000)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Long training jobs belong on the Linux training host")
    if min(args.steps, args.num_envs, args.selection_episodes, args.test_episodes) < 1:
        parser.error("Budgets must be positive")
    selection_seeds = set(range(args.selection_seed, args.selection_seed + args.selection_episodes))
    test_seeds = set(range(args.test_seed, args.test_seed + args.test_episodes))
    if min(args.selection_seed, args.test_seed) < 20000 or selection_seeds & test_seeds:
        parser.error("Selection/test seeds must be disjoint and >=20000")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", started_at=now(), pid=os.getpid(), hostname=platform.node(),
                 arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 commands=[], acceptance_met=False)
    state_path = args.output / "job.json"
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
        with (args.output / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            entry["pid"] = process.pid
            save(state_path, state)
            code = process.wait()
        entry.update(status="complete" if code == 0 else "failed", returncode=code, ended_at=now())
        save(state_path, state)
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect {args.output / (name + '.log')}")

    python = [sys.executable, str(ROOT / "scripts/manipulation.py")]
    model = args.output / "model"
    command = python + ["train", "--output", str(model)]
    for key in ("object", "angle", "turns", "horizon", "randomization", "steps", "num_envs", "device",
                "seed", "resume", "device_config", "mass_kg", "friction"):
        value = getattr(args, key)
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    try:
        run("train", command)
        candidates = []
        for name in ("policy", "best_model"):
            checkpoint = model / (name + ".zip")
            if not checkpoint.is_file():
                continue
            output = args.output / ("validation-" + name)
            run("validation-" + name, python + ["evaluate", "--checkpoint", str(checkpoint),
                "--episodes", str(args.selection_episodes), "--seed", str(args.selection_seed),
                "--output", str(output)])
            candidates.append((checkpoint, json.loads((output / "metrics.json").read_text())))
        chosen, metrics = max(candidates, key=lambda item: selection_score(item[1]))
        selected = args.output / "selected"
        selected.mkdir()
        shutil.copy2(chosen, selected / "policy.zip")
        metadata = json.loads((model / "config.json").read_text())
        with zipfile.ZipFile(chosen) as archive:
            checkpoint_data = json.loads(archive.read("data"))
        metadata.update(selected_from=chosen.name, checkpoint_actual_steps=checkpoint_data["num_timesteps"])
        save(selected / "config.json", metadata)
        state.update(selected_from=str(chosen), validation={k: v for k, v in metrics.items() if k != "trials"})
        for baseline in (None, "hold", "random"):
            name = "test-" + (baseline or "policy")
            command = python + ["evaluate", "--checkpoint", str(selected / "policy.zip"),
                       "--episodes", str(args.test_episodes), "--seed", str(args.test_seed),
                       "--output", str(args.output / name)]
            if baseline:
                command += ["--baseline", baseline]
            run(name, command)
        test = json.loads((args.output / "test-policy/metrics.json").read_text())
        state.update(test={k: v for k, v in test.items() if k != "trials"},
                     acceptance_met=test["acceptance_met"])
        if args.video:
            # Fixed first five TEST episodes, without selecting successful trials.
            try:
                run("video", python + ["evaluate", "--checkpoint", str(selected / "policy.zip"),
                    "--episodes", "5", "--seed", str(args.test_seed), "--video",
                    "--output", str(args.output / "video")])
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
