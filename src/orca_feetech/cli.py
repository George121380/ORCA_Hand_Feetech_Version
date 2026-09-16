from __future__ import annotations

import argparse
import json


def parser():
    root = argparse.ArgumentParser(prog="orca-hand", description="ORCA v1 / STS3215 tools")
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect environment, pinned sources and serial ports")
    doctor.add_argument("--json", action="store_true")
    commands.add_parser("poses", help="List model-space gesture presets")
    scan = commands.add_parser("scan", help="Read-only STS bus scan; does not enable torque")
    scan.add_argument("--port", required=True)
    scan.add_argument("--baudrate", type=int, default=1_000_000)
    scan.add_argument("--ids", default="1-17")
    calibrate = commands.add_parser("calibrate", help="Guided, per-joint STS calibration")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--joint", required=True)
    calibrate.add_argument("--output", default="local/hand.yaml")
    for name in ("demo", "replay", "teleop"):
        p = commands.add_parser(name)
        p.add_argument("--backend", choices=["mock", "sim", "hardware"], default="sim")
        p.add_argument("--config", help="Device-specific hardware calibration YAML")
        p.add_argument("--headless", action="store_true")
        p.add_argument("--rate", type=int, default=20)
        p.add_argument("--output", default=f"runs/{name}")
        p.add_argument("--video", action="store_true", help="Save simulated view as MP4")
        if name == "demo":
            p.add_argument("--poses", default="open,fist,pinch,open")
            p.add_argument("--duration", type=float, default=2.5, help="Minimum transition duration in seconds")
            p.add_argument("--fast", action="store_true", help="Run simulation without wall-clock pacing")
        elif name == "replay":
            p.add_argument("trajectory")
            p.add_argument("--fast", action="store_true")
        else:
            p.add_argument("--source", default="camera:0", help="camera:N or video path")
            p.add_argument("--confidence", type=float, default=0.7)
            p.add_argument("--mirror-input", action="store_true", help="Unmirror prerecorded mirrored input")
            p.add_argument("--max-frames", type=int, default=0)
            p.add_argument("--show-video", action="store_true")
    train = commands.add_parser("train", help="PPO gesture tracking in official MuJoCo v1")
    train.add_argument("--output", default="runs/ppo/seed0")
    train.add_argument("--steps", type=int, default=300_000)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--num-envs", type=int, default=4)
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume")
    train.add_argument("--moving-goal", action="store_true")
    for name in ("evaluate", "infer"):
        p = commands.add_parser(name, help="Load a checkpoint and run separate simulated episodes")
        mode = p.add_mutually_exclusive_group(required=True)
        mode.add_argument("--checkpoint")
        mode.add_argument("--baseline", action="store_true")
        mode.add_argument("--random", dest="random_policy", action="store_true")
        p.add_argument("--output", default=f"runs/{name}")
        p.add_argument("--episodes", type=int, default=100 if name == "evaluate" else 3)
        p.add_argument("--seed", type=int, default=20_000)
        p.add_argument("--video", action="store_true")
        p.add_argument("--moving-goal", action="store_true")
    return root


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    try:
        if command == "doctor":
            from .diagnostics import doctor
            result = doctor()
        elif command == "poses":
            import numpy as np
            from .model import JOINT_NAMES, poses
            result = {name: dict(zip(JOINT_NAMES, np.rad2deg(q).round(2).tolist())) for name, q in poses().items()}
        elif command == "scan":
            from .hardware import scan
            result = scan(**args)
        elif command == "calibrate":
            from .calibration import calibrate
            result = calibrate(**args)
        elif command == "demo":
            from .runners import demo
            result = demo(**args)
        elif command == "replay":
            from .runners import replay
            result = replay(**args)
        elif command == "teleop":
            from .vision import teleop
            result = teleop(**args)
        elif command == "train":
            from .learning import train
            result = str(train(**args))
        else:
            from .learning import evaluate
            result = evaluate(**args)
        if result is not None:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise SystemExit(f"{command}: {exc}") from exc


if __name__ == "__main__":
    main()
