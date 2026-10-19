"""Checkpoint and Linux process checks shared by the training watchdog.

No CUDA context is created here. Process actions must be limited to the exact
job command, rather than matching every Python or GPU process on the host.
"""
from __future__ import annotations

import io
import json
import math
import os
import pickle
import re
from pathlib import Path
import zipfile


def checkpoint_steps(path, *, check_weights=False):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Checkpoint CRC mismatch")
        if not {"data", "policy.pth", "policy.optimizer.pth"}.issubset(archive.namelist()):
            raise ValueError("Incomplete PPO checkpoint")
        steps = json.loads(archive.read("data"))["num_timesteps"]
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("Invalid checkpoint step count")
        if check_weights:
            import torch
            torch.set_num_threads(1)
            weights = torch.load(io.BytesIO(archive.read("policy.pth")),
                                 map_location="cpu", weights_only=True)
            if not isinstance(weights, dict) or not weights or not all(isinstance(v, torch.Tensor) for v in weights.values()):
                raise ValueError("Invalid policy state dictionary")
            if not all(torch.isfinite(value).all().item() for value in weights.values()):
                raise ValueError("Non-finite policy weights")
    return steps


def latest_checkpoint(directories, fallback=None):
    from .checkpoint import load_config
    candidates = []
    for directory in directories:
        for path in Path(directory).glob("*.zip"):
            if path.stem in ("policy", "interrupted", "best_model") or path.stem.startswith("checkpoint_"):
                candidates.append(path)
    if fallback:
        candidates.append(Path(fallback))
    valid = []
    for path in candidates:
        try:
            load_config(path)
            valid.append((checkpoint_steps(path), path.stat().st_mtime_ns, path))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
            continue
    for steps, _, path in sorted(valid, reverse=True):
        try:
            checkpoint_steps(path, check_weights=True)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError, RuntimeError, pickle.UnpicklingError):
            continue
        return path, steps
    return None, 0


def process_argv(pid):
    try:
        path = Path("/proc") / str(int(pid))
        # A zombie is no longer executing, even though kill(pid, 0) succeeds.
        stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z" or (path.stat().st_uid != os.getuid()):
            return []
        return [part.decode() for part in (path / "cmdline").read_bytes().split(b"\0") if part]
    except (OSError, ValueError, UnicodeError):
        return []


def wrapper_alive(state, output):
    argv = process_argv(state.get("pid", 0))
    try:
        return (any(Path(a).name == "run_training_job.py" for a in argv)
                and Path(argv[argv.index("--output") + 1]).resolve() == Path(output).resolve())
    except (ValueError, IndexError):
        return False


def running_processes(state, output):
    result = [state["pid"]] if wrapper_alive(state, output) else []
    for command in state.get("commands", []):
        if command.get("status") == "running" and process_argv(command.get("pid", 0)) == command["argv"]:
            result.append(command["pid"])
    return list(dict.fromkeys(result))


def training_target(arguments):
    initial = checkpoint_steps(arguments["resume"]) if arguments.get("resume") else 0
    return initial + int(arguments["steps"])


def failure_kind(text):
    text = text.lower()
    if any(s in text for s in ("out of memory", "cannot allocate memory", "exit -9", "exited -9")):
        return "memory"
    if any(s in text for s in ("no space left", "disk quota exceeded")):
        return "disk"
    if any(s in text for s in ("cuda error", "cuda unavailable", "nvml", "device lost")):
        return "gpu"
    if re.search(r"\bnan\b|non-finite|nonfinite", text):
        return "numerical"
    return "process"


def progress_sample(job, state):
    model = Path(job) / state.get("current_model", "model")
    progress = model / "progress.json"
    try:
        data = json.loads(progress.read_text())
        steps = data.get("total_steps", 0)
        if not isinstance(steps, (int, float)) or not math.isfinite(steps):
            raise ValueError("Non-finite training progress")
    except (OSError, json.JSONDecodeError):
        steps = None
    log = Path(job) / (state.get("phase", "train") + ".log")
    size = log.stat().st_size if log.exists() else 0
    return [state.get("current_model", "model"), state.get("phase"), steps, size]
