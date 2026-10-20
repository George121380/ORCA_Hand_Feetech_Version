"""Bounded training-only audit of physical first-to-second subgoal handoff.

Compare the new actor alone with a frozen first-skill/new-actor composition.
Search sixteen precommitted residuals only after the physical first completion.
The switching gate and per-seed residual lookup are diagnostic oracles, not a
deployable policy or independent success estimate. Canonical gates stay intact.
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
from orca_feetech.prefix_reset import validate_prefix_checkpoint
from probe_action_residuals import residual_candidates


def handoff_signature(env, observation):
    digest = hashlib.sha256()
    for array in (observation, env.data.qpos, env.data.qvel, env.codec.q, env.codec.velocity,
                  env.goal, env.origin, np.asarray(env.queue), np.array([env.steps, env.data.time])):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps(env.np_random.bit_generator.state, sort_keys=True).encode())
    return digest.hexdigest()


def rollout(env, candidate, seed, *, first=None, residual=None):
    residual = np.zeros(16, np.float32) if residual is None else np.asarray(residual, np.float32)
    if residual.shape != (16,) or not np.isfinite(residual).all() or np.max(np.abs(residual)) > .35:
        raise ValueError("Expected a finite bounded 16-action residual")
    obs, info = env.reset(seed=seed)
    initial = dict(position=info["object_position"], rotation=env.pose()[1].tolist(),
                   direction=env.direction, goal=env.goal.tolist())
    handoff = None
    second_minimum = second_shaft_minimum = None
    second_steps = longest = streak = within_angle = 0
    for step in range(env.config.horizon):
        second = env.completed >= 1
        actor = first if first is not None and not second else candidate
        action = actor.predict(obs, deterministic=True)[0]
        # Do not change the first-skill trajectory, history, RNG, or controller.
        if second:
            action = np.clip(action + residual, -1, 1)
        obs, _, done, truncated, info = env.step(action)
        if not second and env.completed == 1:
            handoff = dict(step=step+1, signature=handoff_signature(env, obs))
        if second:
            second_steps += 1
            error, shaft = info["angle_error_deg"], info["shaft_error_deg"]
            second_minimum = error if second_minimum is None else min(second_minimum, error)
            second_shaft_minimum = shaft if second_shaft_minimum is None else min(second_shaft_minimum, shaft)
            within_angle += error < 8
            streak = streak+1 if info["stable"] and not info["dropped"] else 0
            longest = max(longest, streak)
        if done or truncated:
            break
    reason = ("success" if info["is_success"] else "dropped" if info["dropped"] else
              "vision_timeout" if info["vision_age"] > .25 else "horizon")
    return dict(seed=seed, initial=initial, handoff=handoff, steps=step+1,
                end_reason=reason, success=info["is_success"], dropped=info["dropped"],
                completed_subgoals=info["completed_turns"], second_steps=second_steps,
                second_minimum_error_deg=second_minimum, second_minimum_shaft_error_deg=second_shaft_minimum,
                second_steps_within_angle=within_angle, second_longest_stable_steps=longest,
                final_error_deg=info["angle_error_deg"], residual=residual.tolist())


def initialize(first_checkpoint, candidate_checkpoint):
    import torch
    from stable_baselines3 import PPO
    global FIRST, CANDIDATE, ENV
    torch.set_num_threads(1)
    FIRST, CANDIDATE = (PPO.load(path, device="cpu") for path in (first_checkpoint, candidate_checkpoint))
    ENV = ManipulationEnv(load_config(Path(candidate_checkpoint))[0])


def audit_seed(seed):
    native = rollout(ENV, CANDIDATE, seed)
    composed = rollout(ENV, CANDIDATE, seed, first=FIRST)
    if native["initial"] != composed["initial"]:
        raise RuntimeError("Native and composed rollouts changed initial state or goal")
    result = dict(seed=seed, native=native, composed=composed, candidates=[], rescued=False)
    if composed["end_reason"] != "horizon" or composed["completed_subgoals"] != 1:
        return result
    for index, residual in enumerate(residual_candidates(seed)):
        trial = rollout(ENV, CANDIDATE, seed, first=FIRST, residual=residual)
        if trial["initial"] != composed["initial"] or trial["handoff"] != composed["handoff"]:
            raise RuntimeError("Residual changed the physical first-skill handoff")
        trial["candidate_index"] = index
        result["candidates"].append(trial)
        if trial["success"]:
            result["rescued"] = True
            break
    return result


def summarize(trials):
    return dict(initials_completed=len(trials),
                native_successes=sum(t["native"]["success"] for t in trials),
                composed_successes=sum(t["composed"]["success"] for t in trials),
                composed_first_completions=sum(t["composed"]["handoff"] is not None for t in trials),
                horizon_second_failures_searched=sum(bool(t["candidates"]) for t in trials),
                rescued_training_initials=sum(t["rescued"] for t in trials),
                candidate_rollouts=sum(len(t["candidates"]) for t in trials))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=160)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 32 or not 0 <= args.seed < args.seed+args.episodes <= 20000:
        parser.error("Use1..32 training seeds below20000")
    config, meta = load_config(args.candidate)
    teacher_proof = validate_prefix_checkpoint(args.first_checkpoint, config)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), status="running",
                  purpose="Training-only physical second-subgoal reachability audit; no learning or acceptance",
                  is_formal_evaluation=False, acceptance_met=False, first_skill=teacher_proof,
                  candidate=str(args.candidate.resolve()),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256((args.candidate.parent/"config.json").read_bytes()).hexdigest(),
                  fingerprint=meta["fingerprint"], seed=args.seed, episodes=args.episodes,
                  maximum_rollouts=args.episodes*18, maximum_control_steps=args.episodes*18*config.horizon,
                  candidates_per_horizon_second_failure=16, residual_amplitudes=[.15, .35],
                  switching_gate_is_diagnostic_oracle=True, trials=[])
    def persist():
        report["summary"] = summarize(report["trials"])
        temporary = args.output/"probe.tmp"
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(args.output/"probe.json")
    persist()
    try:
        with ProcessPoolExecutor(max_workers=4, initializer=initialize,
                                 initargs=(str(args.first_checkpoint), str(args.candidate))) as pool:
            futures = [pool.submit(audit_seed, seed) for seed in range(args.seed, args.seed+args.episodes)]
            for future in as_completed(futures):
                report["trials"].append(future.result())
                report["trials"].sort(key=lambda t:t["seed"])
                persist()
                print(json.dumps(report["summary"]), flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        persist()
        raise
    report.update(status="complete", ended_at=datetime.now(timezone.utc).isoformat())
    persist()


if __name__ == "__main__":
    main()
