"""Bounded training-seed action search, never checkpoint selection or acceptance.

Each candidate adds one constant, small 16-vector to the observable policy's
actions for a complete canonical rollout. Only baseline horizon failures are
searched. Successful observable trajectories can inform a later imitation
experiment; the search itself is not a deployable policy or success estimate.
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


def residual_candidates(seed, search_seed=0):
    """Sixteen precommitted candidates: four antithetic pairs per amplitude."""
    rng = np.random.default_rng(np.random.SeedSequence([search_seed, seed]))
    candidates = []
    for amplitude in (.15, .35):
        for _ in range(4):
            vector = rng.uniform(-amplitude, amplitude, 16).astype(np.float32)
            candidates.extend([vector, -vector])
    return candidates


def rollout(env, policy, seed, residual):
    """Keep canonical physics, vision, termination and full-pose success gates."""
    residual = np.asarray(residual, dtype=np.float32)
    if residual.shape != (16,) or not np.isfinite(residual).all() or np.max(np.abs(residual)) > .35:
        raise ValueError("Expected a finite bounded 16-action residual")
    obs, info = env.reset(seed=seed)
    initial = dict(position=info["object_position"], rotation=env.pose()[1].tolist(),
                   direction=env.direction, goal=env.goal.tolist())
    observations, actions = [], []
    minimum_error = info["angle_error_deg"]
    for _ in range(env.config.horizon):
        action = np.clip(policy.predict(obs, deterministic=True)[0] + residual, -1, 1)
        observations.append(obs.copy())
        actions.append(action.copy())
        obs, _, done, truncated, info = env.step(action)
        minimum_error = min(minimum_error, info["angle_error_deg"])
        if done or truncated:
            break
    reason = ("success" if info["is_success"] else "dropped" if info["dropped"] else
              "vision_timeout" if info["vision_age"] > .25 else "horizon")
    summary = dict(seed=seed, initial=initial, residual=residual.tolist(), steps=len(actions),
                   end_reason=reason, success=info["is_success"], dropped=info["dropped"],
                   minimum_error_deg=minimum_error, final_error_deg=info["angle_error_deg"],
                   shaft_error_deg=info["shaft_error_deg"])
    return summary, np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def initialize(checkpoint, output, search_seed):
    import torch
    from stable_baselines3 import PPO
    global ENV, POLICY, OUTPUT, SEARCH_SEED
    torch.set_num_threads(1)
    config, _ = load_config(Path(checkpoint))
    ENV, POLICY = ManipulationEnv(config), PPO.load(checkpoint, device="cpu")
    OUTPUT, SEARCH_SEED = Path(output), search_seed


def save_success(label, trial, observations, actions):
    if trial["success"]:
        # The policy's exact raw 296 inputs and bounded 16 commands only.
        # No simulator velocities, contacts or randomization parameters.
        path = OUTPUT/"trajectories"/f"seed-{trial['seed']}-{label}.npz"
        np.savez_compressed(path, observations=observations, actions=actions,
                            seed=np.array(trial["seed"]), residual=np.asarray(trial["residual"]))
        trial["trajectory"] = str(path.relative_to(OUTPUT))


def search_seed(seed):
    baseline, obs, actions = rollout(ENV, POLICY, seed, np.zeros(16, dtype=np.float32))
    save_success("baseline", baseline, obs, actions)
    result = dict(seed=seed, baseline=baseline, candidates=[], rescued=False)
    # A camera timeout is a separate limitation; do not select on its random
    # truncation or spend a reachability search budget on it.
    if baseline["end_reason"] != "horizon":
        return result
    for i, residual in enumerate(residual_candidates(seed, SEARCH_SEED)):
        trial, obs, actions = rollout(ENV, POLICY, seed, residual)
        if trial["initial"] != baseline["initial"]:
            raise RuntimeError("Candidate does not share the baseline initial state and goal")
        trial["candidate_index"] = i
        result["candidates"].append(trial)
        if trial["success"]:
            save_success(f"candidate-{i}", trial, obs, actions)
            result["rescued"] = True
            break
    return result


def persist(output, report):
    temporary = output/"probe.tmp"
    temporary.write_text(json.dumps(report, indent=2)+"\n")
    temporary.replace(output/"probe.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if (not 0 <= args.seed < args.seed+args.episodes <= 20000 or not 1 <= args.episodes <= 32
            or not 1 <= args.workers <= 4 or args.search_seed < 0):
        parser.error("Use 1..32 training seeds below20000, 1..4 workers and nonnegative search seed")
    config, metadata = load_config(args.checkpoint)
    if (config.object != "pen" or config.angle != 35 or config.turns != 1
            or config.randomization != 1 or config.horizon != 240):
        parser.error("This frozen diagnostic is limited to canonical pen35/r1")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"trajectories").mkdir()
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), status="running",
                  purpose="Training-seed local action search; not a policy or acceptance estimate",
                  is_formal_evaluation=False, checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256((args.checkpoint.parent/"config.json").read_bytes()).hexdigest(),
                  fingerprint=metadata["fingerprint"], seed=args.seed, episodes=args.episodes,
                  search_seed=args.search_seed, candidates_per_horizon_failure=16,
                  amplitudes=[.15, .35], maximum_control_steps=args.episodes*17*config.horizon,
                  episodes_completed=0, trials=[])
    persist(args.output, report)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(str(args.checkpoint), str(args.output), args.search_seed)) as pool:
            futures = [pool.submit(search_seed, seed) for seed in range(args.seed, args.seed+args.episodes)]
            for future in as_completed(futures):
                report["trials"].append(future.result())
                report["trials"].sort(key=lambda t: t["seed"])
                report["episodes_completed"] = len(report["trials"])
                report["baseline_successes"] = sum(t["baseline"]["success"] for t in report["trials"])
                report["horizon_failures_searched"] = sum(t["baseline"]["end_reason"] == "horizon"
                                                         for t in report["trials"])
                report["rescued_training_initials"] = sum(t["rescued"] for t in report["trials"])
                report["candidate_rollouts"] = sum(len(t["candidates"]) for t in report["trials"])
                persist(args.output, report)
                print(json.dumps({k:v for k,v in report.items() if k != "trials"}), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        persist(args.output, report)
        raise
    report["ended_at"] = datetime.now(timezone.utc).isoformat()
    persist(args.output, report)


if __name__ == "__main__":
    main()
