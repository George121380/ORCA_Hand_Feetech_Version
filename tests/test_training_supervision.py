import importlib.util
import io
import json
from pathlib import Path
import zipfile
from types import SimpleNamespace

import pytest

from orca_feetech import training_supervision as supervisor


def checkpoint(path, steps, weight=1., optimizer_value=1.):
    import torch
    stream = io.BytesIO()
    torch.save({"weight": torch.tensor([weight])}, stream)
    optimizer = io.BytesIO()
    torch.save({"state": {0: {"exp_avg": torch.tensor([optimizer_value])}}, "param_groups": []}, optimizer)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(dict(num_timesteps=steps)))
        archive.writestr("policy.pth", stream.getvalue())
        archive.writestr("policy.optimizer.pth", optimizer.getvalue())


def test_recovery_uses_actual_steps_and_skips_corrupt_or_nan_weights(monkeypatch, tmp_path):
    from orca_feetech import checkpoint as contract
    monkeypatch.setattr(contract, "load_config", lambda path: None)
    good = tmp_path / "checkpoint_100_steps.zip"
    checkpoint(good, 100)
    checkpoint(tmp_path / "checkpoint_200_steps.zip", 200, float("nan"))
    checkpoint(tmp_path / "checkpoint_300_steps.zip", 300, optimizer_value=float("nan"))
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


def test_quality_pilot_stays_on_failed_skill_uses_new_test_and_stops_at_budget(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    spec = importlib.util.spec_from_file_location("watch", Path(__file__).parents[1] / "scripts/watch_training.py")
    watch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watch)
    monkeypatch.setattr(watch.platform, "system", lambda: "Linux")
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)
    initial = tmp_path / "initial"
    initial.mkdir()
    options = dict(device="cpu", steps=100, num_envs=2, object="pen", angle=45, turns=1,
                   horizon=240, randomization=.3, selection_seed=50000, test_seed=60000,
                   learning_rate=.0003, entropy_coef=.01)
    original = dict(status="complete", acceptance_met=False, arguments=options)
    (initial / "job.json").write_text(json.dumps(original))
    launches = []

    def launch(argv, **kwargs):
        launches.append(argv)
        output = Path(argv[argv.index("--output")+1])
        output.mkdir()
        (output / "job.json").write_text(json.dumps(original))
        return SimpleNamespace(pid=99999999)

    monkeypatch.setattr(watch.subprocess, "Popen", launch)
    output = tmp_path / "watch"
    monkeypatch.setattr(watch.sys, "argv", ["watch", "--initial-job", str(initial),
                       "--state-dir", str(output), "--quality-retries", "1", "--quality-retry-steps", "200"])
    watch.main()
    state = json.loads((output / "watchdog.json").read_text())
    assert state["status"] == "quality_gate_failed"
    assert state["stage_index"] == 0 and state["quality_attempt"] == 1
    assert len(launches) == 1 and len(state["stage_results"]) == 2
    argv = launches[0]
    assert argv[argv.index("--resume")+1] == str(initial / "selected/policy.zip")
    assert argv[argv.index("--angle")+1] == "45"
    assert argv[argv.index("--steps")+1] == "200"
    assert argv[argv.index("--test-seed")+1] == "160000"
    assert argv[argv.index("--selection-seed")+1] == "50000"
    assert "--reset-log-std" not in argv
    assert json.loads((initial / "job.json").read_text()) == original
