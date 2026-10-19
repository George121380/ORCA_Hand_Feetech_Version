import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

from orca_feetech import training_supervision as supervisor


def checkpoint(path, steps, weight=1.):
    import torch
    stream = io.BytesIO()
    torch.save({"weight": torch.tensor([weight])}, stream)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(dict(num_timesteps=steps)))
        archive.writestr("policy.pth", stream.getvalue())
        archive.writestr("policy.optimizer.pth", b"optimizer")


def test_recovery_uses_actual_steps_and_skips_corrupt_or_nan_weights(monkeypatch, tmp_path):
    from orca_feetech import checkpoint as contract
    monkeypatch.setattr(contract, "load_config", lambda path: None)
    good = tmp_path / "checkpoint_100_steps.zip"
    checkpoint(good, 100)
    checkpoint(tmp_path / "checkpoint_200_steps.zip", 200, float("nan"))
    (tmp_path / "checkpoint_999_steps.zip").write_bytes(b"interrupted ZIP write")
    path, steps = supervisor.latest_checkpoint([tmp_path])
    assert path == good and steps == 100
    assert supervisor.training_target(dict(resume=str(good), steps=900)) == 1000


def test_process_matching_never_adopts_an_unrelated_python_job(monkeypatch, tmp_path):
    state = dict(pid=123, commands=[])
    monkeypatch.setattr(supervisor, "process_argv", lambda pid: ["python", "other_training.py"])
    assert not supervisor.running_processes(state, tmp_path)
    monkeypatch.setattr(supervisor, "process_argv", lambda pid: ["python", "/repo/run_training_job.py",
                                                              "--output", str(tmp_path)])
    assert supervisor.running_processes(state, tmp_path) == [123]
    assert not supervisor.running_processes(state, tmp_path / "other")


@pytest.mark.parametrize("message,expected", [
    ("CUDA out of memory", "memory"), ("train exited -9", "memory"),
    ("No space left on device", "disk"), ("CUDA error: device lost", "gpu"),
    ("non-finite action", "numerical"), ("NaN policy weights", "numerical"),
    ("filename finance.py: worker exited 1", "process"),
])
def test_failure_classification(message, expected):
    assert supervisor.failure_kind(message) == expected


@pytest.mark.parametrize("job_status,acceptance,expected", [
    ("complete", False, "quality_gate_failed"), ("complete", True, "complete"),
    ("failed", False, "recovery_exhausted"),
])
def test_watchdog_distinguishes_quality_from_process_failure(monkeypatch, tmp_path,
                                                           job_status, acceptance, expected):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    spec = importlib.util.spec_from_file_location("watch", Path(__file__).parents[1] / "scripts/watch_training.py")
    watch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watch)
    monkeypatch.setattr(watch.platform, "system", lambda: "Linux")
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "job.json").write_text(json.dumps(dict(status=job_status, acceptance_met=acceptance,
                                                     phase="train", pid=99999999, commands=[],
                                                     arguments=dict(device="cpu", steps=100))))
    output = tmp_path / "watch"
    monkeypatch.setattr(watch.sys, "argv", ["watch", "--initial-job", str(initial), "--state-dir", str(output),
                       "--max-restarts", "0"])
    watch.main()
    state = json.loads((output / "watchdog.json").read_text())
    assert state["status"] == expected
    assert (initial / "job.json").is_file()
