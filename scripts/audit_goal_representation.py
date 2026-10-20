"""Development-only audit of nominal pen symmetry and policy action sensitivity."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from orca_feetech.checkpoint import load_config
from orca_feetech.manipulation import ManipulationEnv
from orca_feetech.objects import collision_parts


def main():
    import torch
    from stable_baselines3 import PPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()
    config, _ = load_config(args.checkpoint)
    if config.object != "pen" or args.seed < 20000 or args.episodes < 1 or args.output.exists():
        parser.error("Use pen, development seeds >=20000, positive episodes, and a new output")
    torch.set_num_threads(1)
    policy = PPO.load(args.checkpoint, device="cpu")
    S = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    geometry_errors = []
    for vertices, faces in collision_parts("pen"):
        rotated = vertices @ S.T
        distances = np.linalg.norm(rotated[:, None, :] - vertices[None, :, :], axis=-1)
        geometry_errors.append(float(np.max(np.min(distances, axis=1))))
    env = ManipulationEnv(config)
    rows = []
    try:
        for seed in range(args.seed, args.seed+args.episodes):
            obs, _ = env.reset(seed=seed)
            altered = obs.reshape(4, 74).copy()
            R = altered[:, 38:47].reshape(4, 3, 3).copy()
            G = altered[:, 47:56].reshape(4, 3, 3).copy()
            altered[:, 38:47] = (R @ S).reshape(4, 9)
            altered[:, 47:56] = (G @ S).reshape(4, 9)
            a = policy.predict(obs, deterministic=True)[0]
            b = policy.predict(altered.ravel(), deterministic=True)[0]
            error_difference = G @ R.transpose(0, 2, 1) - (G @ S) @ (R @ S).transpose(0, 2, 1)
            rows.append(dict(seed=seed, mean_abs_action_difference=float(np.mean(np.abs(a-b))),
                             max_abs_action_difference=float(np.max(np.abs(a-b))),
                             world_error_matrix_difference=float(np.max(np.abs(error_difference)))))
    finally:
        env.close()
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(args.checkpoint.resolve()),
                  source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  purpose="Development-only nominal symmetry audit; not causal proof or hardware assumption",
                  nominal_geometry_vertex_hausdorff_m=geometry_errors,
                  mean_abs_action_difference=float(np.mean([r["mean_abs_action_difference"] for r in rows])),
                  max_abs_action_difference=max(r["max_abs_action_difference"] for r in rows),
                  max_world_error_matrix_difference=max(r["world_error_matrix_difference"] for r in rows),
                  trials=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
