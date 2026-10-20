"""Offline actor-only imitation of successful observable training trajectories."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor

from .checkpoint import load_config
from .training_supervision import checkpoint_steps


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def censored_action_loss(mean, target):
    """Distance from mean to the preimage of a clipped demonstrated action.

    A target +1 requires mean>=1, rather than arbitrarily pulling an already
    saturated mean back to exactly1. Interior targets retain a useful gradient
    even when the current mean is outside the action limits.
    """
    difference = mean-target
    difference = torch.where(target >= 1, difference.clamp(max=0), difference)
    difference = torch.where(target <= -1, difference.clamp(min=0), difference)
    return difference.square().mean()


def load_demonstrations(reports, source, fingerprint):
    source = Path(source)
    source_hash, config_hash = digest(source), digest(source.parent/"config.json")
    groups = {"baseline": [], "rescue": []}
    proof, seeds = [], set()
    for report_path in reports:
        report_path = Path(report_path).resolve()
        report = json.loads(report_path.read_text())
        if (report.get("status") != "complete" or report.get("is_formal_evaluation") is not False
                or report.get("checkpoint_sha256") != source_hash
                or report.get("config_sha256") != config_hash or report.get("fingerprint") != fingerprint):
            raise ValueError("Demonstrations do not match the completed training-seed source protocol")
        expected = list(range(report["seed"], report["seed"]+report["episodes"]))
        if (not expected or min(expected) < 0 or max(expected) >= 20000
                or [t["seed"] for t in report["trials"]] != expected or seeds.intersection(expected)):
            raise ValueError("Use disjoint, complete training seed blocks below20000")
        seeds.update(expected)
        if len(seeds) > 256:
            raise ValueError("Offline imitation is limited to256 initial states")
        entry = dict(report=str(report_path), sha256=digest(report_path), trajectories=[])
        for trial in report["trials"]:
            examples = [("baseline", trial["baseline"])]
            examples += [("rescue", candidate) for candidate in trial["candidates"]]
            for label, example in examples:
                if not example["success"]:
                    continue
                if example.get("end_reason") != "success" or example.get("dropped"):
                    raise ValueError("A demonstration lacks canonical physical success")
                path = (report_path.parent/example["trajectory"]).resolve()
                if not path.is_relative_to(report_path.parent) or path.suffix != ".npz":
                    raise ValueError("Trajectory must be a local numeric NPZ")
                with np.load(path, allow_pickle=False) as data:
                    obs, actions = data["observations"], data["actions"]
                    if (obs.shape != (example["steps"], 296) or actions.shape != (len(obs), 16)
                            or not 10 <= len(obs) <= 240 or not np.isfinite(obs).all()
                            or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1
                            or int(data["seed"]) != trial["seed"]
                            or not np.array_equal(data["residual"], np.asarray(example["residual"]))):
                        raise ValueError("Malformed observable trajectory or label")
                    groups[label].append((obs.astype(np.float32), actions.astype(np.float32)))
                entry["trajectories"].append(dict(path=str(path), sha256=digest(path), group=label,
                                                  seed=trial["seed"], samples=len(obs)))
        proof.append(entry)
    if not all(groups.values()):
        raise ValueError("Need both baseline preservation examples and successful rescue examples")
    arrays = {label: (np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs]))
              for label, pairs in groups.items()}
    return arrays, proof


def imitate_actor(model, groups, *, updates=512, batch_size=256, learning_rate=3.75e-5, seed=0):
    if (not 1 <= updates <= 512 or not 2 <= batch_size <= 256 or batch_size % 2
            or not 0 < learning_rate <= 3.75e-5 or seed < 0):
        raise ValueError("Offline update/batch/learning-rate budget exceeded")
    policy = model.policy
    if (type(policy) is not ActorCriticPolicy or type(policy.features_extractor) is not FlattenExtractor
            or type(policy.action_dist) is not DiagGaussianDistribution or model.use_sde
            or policy.squash_output or model.observation_space.shape != (296,)
            or model.action_space.shape != (16,) or not np.all(model.action_space.low == -1)
            or not np.all(model.action_space.high == 1)):
        raise ValueError("Expected the canonical unregularized Gaussian policy")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy.set_training_mode(False)
    params = dict(policy.named_parameters())
    actor = {n:p for n,p in params.items() if n.startswith(("mlp_extractor.policy_net.", "action_net."))}
    unchanged = {n:p.detach().clone() for n,p in params.items() if n not in actor}
    values = {label: (torch.as_tensor(obs, device=model.device), torch.as_tensor(actions, device=model.device))
              for label, (obs, actions) in groups.items()}
    if set(values) != {"baseline", "rescue"} or any(len(v[0]) == 0 for v in values.values()):
        raise ValueError("Need both nonempty observable demonstration groups")
    def loss_for(obs, actions):
        return censored_action_loss(policy.get_distribution(obs).distribution.mean, actions)
    def metrics():
        with torch.no_grad():
            return {label: float(torch.stack([loss_for(obs[i:i+256], a[i:i+256])
                    *len(obs[i:i+256]) for i in range(0,len(obs),256)]).sum()/len(obs))
                    for label,(obs,a) in values.items()}
    before = metrics()
    optimizer = torch.optim.Adam(actor.values(), lr=learning_rate, eps=1e-5)
    for _ in range(updates):
        losses = []
        for obs, actions in values.values():
            indices = rng.integers(0, len(obs), size=batch_size//2)
            losses.append(loss_for(obs[indices], actions[indices]))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite imitation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor.values()), .5, error_if_nonfinite=True)
        optimizer.step()
    if not all(torch.isfinite(p).all() for p in params.values()):
        raise ValueError("Nonfinite imitation parameter")
    if not all(torch.equal(params[n], value) for n,value in unchanged.items()):
        raise ValueError("Offline imitation modified critic or exploration parameters")
    # Previous actor moments no longer describe this offline update. Keep all
    # critic/log_std optimizer history and PPO environment step counters.
    reset_count = sum(p in policy.optimizer.state for p in actor.values())
    for parameter in actor.values():
        policy.optimizer.state.pop(parameter, None)
    return dict(updates=updates, batch_size=batch_size, learning_rate=learning_rate, seed=seed,
                group_samples={k:len(v[0]) for k,v in values.items()}, loss_before=before, loss_after=metrics(),
                actor_parameter_tensors=len(actor), reset_actor_optimizer_states=reset_count,
                unchanged_critic_and_exploration=True, additional_environment_steps=0)


def clone_from_demonstrations(source, destination, reports, *, device="cpu"):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    config, metadata = load_config(source)
    checkpoint_steps(source, check_weights=True)
    if config.object != "pen" or config.angle != 35 or config.turns != 1 or config.randomization != 1:
        raise ValueError("This offline protocol is limited to pen35/r1")
    arrays, data_proof = load_demonstrations(reports, source, metadata["fingerprint"])
    torch.set_num_threads(1)
    model = PPO.load(source, device=device)
    counters = model.num_timesteps, model._n_updates
    proof = imitate_actor(model, arrays)
    assert (model.num_timesteps, model._n_updates) == counters
    proof.update(source=str(source), source_sha256=digest(source), datasets=data_proof,
                 timestamp=datetime.now(timezone.utc).isoformat(), source_environment_steps=counters[0],
                 source_ppo_updates=counters[1], is_formal_evaluation=False)
    metadata.update(offline_imitation=proof)
    destination.mkdir(parents=True, exist_ok=False)
    model.save(destination/"policy.zip")
    (destination/"config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (destination/"imitation.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof
