"""Reproduce the three-seed PPO benchmark, baseline and random-policy comparison."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"runs/benchmark-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = []

    def run(name, arguments):
        print(name, flush=True)
        with (output / (name + ".log")).open("w") as log:
            subprocess.run([sys.executable, "-m", "orca_feetech.cli", *arguments],
                           check=True, stdout=log, stderr=subprocess.STDOUT)

    for seed in range(3):
        policy = output / f"seed{seed}"
        evaluation = output / f"eval{seed}"
        run(f"train{seed}", ["train", "--output", str(policy), "--seed", str(seed),
                             "--steps", str(args.steps), "--num-envs", str(args.num_envs)])
        run(f"eval{seed}", ["evaluate", "--checkpoint", str(policy / "policy.zip"),
                            "--episodes", str(args.episodes), "--output", str(evaluation)])
        metrics = json.loads((evaluation / "metrics.json").read_text())
        results.append({"seed": seed, **{k: v for k, v in metrics.items() if k != "results"}})
    for label in ("baseline", "random"):
        evaluation = output / label
        run(label, ["evaluate", "--" + label, "--episodes", str(args.episodes), "--output", str(evaluation)])
        metrics = json.loads((evaluation / "metrics.json").read_text())
        results.append({k: v for k, v in metrics.items() if k != "results"})
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    passed = all(r["success_rate"] >= 0.9 for r in results[:3])
    print(json.dumps({"output": str(output), "three_seed_acceptance_passed": passed}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
