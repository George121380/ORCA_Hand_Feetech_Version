"""Continue an existing first pen stage through a bounded, gated curriculum.

Each stage has its own validation and test seed ranges. A failed quality gate
stops the course without silently treating an unfinished skill as learned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from run_training_job import ROOT, now, save


def passed(state):
    return state.get("status") == "complete" and state.get("acceptance_met") is True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run this course on the Linux training host")
    args.initial_job = args.initial_job.resolve()
    args.output = args.output.resolve()
    initial_path = args.initial_job / "job.json"
    initial = json.loads(initial_path.read_text())
    options = initial["arguments"]
    if (options["object"] != "pen" or options["angle"] != 15 or options["turns"] != 1
            or options["randomization"] != .3 or options["selection_seed"] != 30000
            or options["test_seed"] != 20000 or options["test_episodes"] != 100):
        parser.error("Initial job must be the standard pen 15deg/r0.3 stage with 100 test episodes")
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status="waiting", started_at=now(), initial_job=str(args.initial_job), stages=[])
    state_path = args.output / "curriculum.json"
    try:
        deadline = time.monotonic() + 24 * 3600
        while initial.get("status") == "running":
            save(state_path, state)
            if time.monotonic() > deadline:
                raise TimeoutError("Initial stage did not finish within 24 hours")
            time.sleep(20)
            initial = json.loads(initial_path.read_text())
        if not passed(initial):
            state.update(status="quality_gate_failed", failed_stage=str(args.initial_job))
            return
        previous = args.initial_job
        stages = [("15-r1", 15, 1, 240, 3000000), ("45-r1", 45, 1, 240, 3000000),
                  ("90-r1", 90, 1, 240, 3000000), ("90x4-r1", 90, 4, 800, 10000000)]
        for index, (name, angle, turns, horizon, steps) in enumerate(stages, 1):
            output = args.output / name
            command = [sys.executable, str(ROOT / "scripts/run_training_job.py"),
                       "--object", "pen", "--angle", str(angle), "--turns", str(turns),
                       "--horizon", str(horizon), "--randomization", "1", "--steps", str(steps),
                       "--num-envs", str(options["num_envs"]), "--device", options["device"],
                       "--seed", str(options["seed"]), "--resume", str(previous / "selected/policy.zip"),
                       "--selection-seed", str(30000 + index*1000),
                       "--test-seed", str(20000 + index*1000), "--video", "--output", str(output)]
            for key in ("device_config", "mass_kg", "friction"):
                if options.get(key) is not None:
                    command += ["--" + key.replace("_", "-"), str(options[key])]
            entry = dict(name=name, output=str(output), argv=command, started_at=now())
            state["stages"].append(entry)
            state.update(status="running", phase=name)
            save(state_path, state)
            with (args.output / (name + ".log")).open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output / "job.json").read_text())
            entry.update(ended_at=now(), acceptance_met=result["acceptance_met"], test=result["test"])
            if not passed(result):
                state.update(status="quality_gate_failed", failed_stage=str(output))
                return
            previous = output
        state.update(status="complete", selected_checkpoint=str(previous / "selected/policy.zip"))
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        state["updated_at"] = now()
        save(state_path, state)
        print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
