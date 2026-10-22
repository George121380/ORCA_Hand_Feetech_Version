"""Record bounded, unfiltered intermediate rollouts for the user's review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.project, text=True).strip()
    if revision != plan["native_source_revision"]:
        raise ValueError("Native source must match the registered revision")
    if plan["maximum_rollouts"] != 2 * plan["episodes_per_checkpoint"]:
        raise ValueError("Review budget must cover exactly two checkpoints")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.plan, args.output / "plan.json")
    shutil.copy2(__file__, args.output / "recorder.py")
    reference_path = args.output / "reference-metrics.json"
    shutil.copy2(plan["reference_metrics"], reference_path)
    reference = read(reference_path)["trials"][:plan["episodes_per_checkpoint"]]
    seeds = list(range(plan["seed"], plan["seed"] + len(reference)))
    if [r["seed"] for r in reference] != seeds:
        raise ValueError("Reference must contain the registered fixed first seeds")
    state = dict(status="recording", started_at=datetime.now(timezone.utc).isoformat(),
                 native_revision=revision, plan_sha256=digest(args.plan),
                 recorder_sha256=digest(__file__),
                 reference_sha256=digest(reference_path), models={})
    status_path = args.output / "review.json"
    write(status_path, state)
    env = dict(os.environ, ORCA_WORKSPACE=str(args.project),
               PYTHONPATH=str(args.project / "src"), MUJOCO_GL="egl",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               ORCA_GRASP_CHECKPOINT=plan["grasp_checkpoint"])
    try:
        for name, checkpoint in [("comparison", plan["comparison_checkpoint"]),
                                 ("pure", plan["pure_checkpoint"])]:
            source = Path(checkpoint)
            frozen = args.output / (name + "-frozen")
            frozen.mkdir()
            source_hash = digest(source)
            if name == "comparison" and source_hash != plan["comparison_checkpoint_sha256"]:
                raise ValueError("Comparison checkpoint changed")
            shutil.copy2(source, frozen / "policy.zip")
            shutil.copy2(source.parent / "config.json", frozen / "config.json")
            if digest(frozen / "policy.zip") != source_hash or digest(source) != source_hash:
                raise ValueError("Checkpoint changed while freezing")
            output = args.output / name
            command = ["timeout", "--signal=TERM", "--kill-after=10s",
                       str(plan["timeout_seconds_per_checkpoint"]), sys.executable,
                       str(args.project / "scripts/pen_rotate.py"), "evaluate",
                       "--checkpoint", str(frozen / "policy.zip"),
                       "--episodes", str(plan["episodes_per_checkpoint"]),
                       "--seed", str(plan["seed"]), "--output", str(output), "--video"]
            entry = dict(source=str(source), checkpoint_sha256=source_hash,
                         config_sha256=digest(frozen / "config.json"), argv=command,
                         status="running", started_at=datetime.now(timezone.utc).isoformat())
            state["models"][name] = entry
            write(status_path, state)
            with (args.output / (name + ".log")).open("w") as log:
                result = subprocess.run(command, cwd=args.project, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            entry["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{name} recording failed; no automatic retry")
            metrics = read(output / "metrics.json")
            trials = metrics["trials"]
            initial_fields = ["seed", "direction", "initial_position", "initial_rotation",
                              "goal_rotation", "reset_attempts", "settle_seconds",
                              "reset_linear_speed_mps", "reset_angular_speed_rps"]
            paired = len(trials) == len(reference) and all(
                all(t[key] == r[key] for key in initial_fields)
                for t, r in zip(trials, reference))
            entry.update(status="complete", exact_reference_initial_pairing=paired,
                         metrics_sha256=digest(output / "metrics.json"),
                         video_sha256=digest(output / "evaluation.mp4"),
                         summary={k: v for k, v in metrics.items() if k != "trials"},
                         ended_at=datetime.now(timezone.utc).isoformat())
            write(status_path, state)
            if not paired:
                raise ValueError("Initial state pairing failed; retain evidence without a paired claim")
        state.update(status="complete", ended_at=datetime.now(timezone.utc).isoformat(),
                     rollouts=plan["maximum_rollouts"], formal_acceptance=False)
        write(status_path, state)
    except Exception as error:
        state.update(status="failed", error=str(error),
                     ended_at=datetime.now(timezone.utc).isoformat())
        write(status_path, state)
        raise


if __name__ == "__main__":
    main()
